"""Render and validate deterministic dataset-distribution chart artifacts."""
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import font_manager, image as matplotlib_image, pyplot  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402


CHART_DPI = 150
CHART_MIN_WIDTH_INCH = 10.0
CHART_HEIGHT_INCH = 7.0
BAR_WIDTH_INCH = 0.65
LABEL_ROTATION_MEDIUM = 45
LABEL_ROTATION_DENSE = 75
SPLITS = ('train', 'val', 'test')


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file_obj:
        while True:
            block = file_obj.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_cjk_font(required_text):
    """Return a Matplotlib font that contains every required non-ASCII glyph."""
    if not isinstance(required_text, str):
        raise TypeError('required_text must be a string')
    required = set(ord(character) for character in required_text if ord(character) > 127)
    candidates = sorted(
        set(font_manager.findSystemFonts(fontext='ttf') +
            font_manager.findSystemFonts(fontext='otf'))
    )
    for font_path in candidates:
        charmap = FT2Font(font_path).get_charmap()
        if required.issubset(charmap):
            return {
                'family_name': font_manager.FontProperties(fname=font_path).get_name(),
                'font_path': str(Path(font_path).resolve()),
            }
    missing = ''.join(sorted(chr(codepoint) for codepoint in required))
    raise RuntimeError('No installed font covers all required CJK characters: {}'.format(missing))


def _assert_font_coverage(font, required_text):
    font_path = Path(font.get('font_path', ''))
    if not font_path.is_file() or not font.get('family_name'):
        raise ValueError('font must identify an existing font file')
    required = set(ord(character) for character in required_text if ord(character) > 127)
    available = set(FT2Font(str(font_path)).get_charmap())
    missing = required - available
    if missing:
        characters = ''.join(sorted(chr(codepoint) for codepoint in missing))
        raise RuntimeError('Selected font lacks required CJK glyphs: {}'.format(characters))


def _validate_distribution(distribution):
    mapping = distribution.get('class_to_idx')
    expected = dict((name, index) for index, name in enumerate(sorted(mapping or {})))
    if mapping != expected or not mapping:
        raise ValueError('class_to_idx must be a nonempty sorted contiguous mapping')
    for split in SPLITS:
        class_counts = distribution.get('class_counts', {}).get(split)
        subclasses = distribution.get('subclass_counts', {}).get(split)
        if list(class_counts or {}) != list(mapping) or list(subclasses or {}) != list(mapping):
            raise ValueError('Distribution classes are incomplete or unstable for {}'.format(split))
        for plant in mapping:
            counts = subclasses[plant]
            if list(counts) != sorted(counts) or any(type(value) is not int or value < 0
                                                    for value in counts.values()):
                raise ValueError('Invalid subclass counts for {}/{}'.format(split, plant))
            if class_counts[plant] != sum(counts.values()):
                raise ValueError('Class/subclass counts disagree for {}/{}'.format(split, plant))
        if distribution.get('split_totals', {}).get(split) != sum(class_counts.values()):
            raise ValueError('Split total disagrees for {}'.format(split))


def _slug(plant):
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', plant).strip('._') or 'class'
    digest = hashlib.sha256(plant.encode('utf-8')).hexdigest()[:10]
    return '{}_{}'.format(safe[:40], digest)


def _draw_chart(output_dir, relative_path, split, level, plant, labels, counts, font):
    path = output_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    rotation = 0 if len(labels) <= 8 else (
        LABEL_ROTATION_MEDIUM if len(labels) <= 15 else LABEL_ROTATION_DENSE
    )
    properties = font_manager.FontProperties(fname=font['font_path'])
    figure, axes = pyplot.subplots(figsize=(
        max(CHART_MIN_WIDTH_INCH, len(labels) * BAR_WIDTH_INCH), CHART_HEIGHT_INCH
    ))
    bars = axes.bar(range(len(labels)), counts)
    axes.set_xticks(range(len(labels)))
    axes.set_xticklabels(labels, rotation=rotation, ha='right' if rotation else 'center',
                         fontproperties=properties)
    title = '{} 数据集植物大类分布'.format(split.upper())
    if level == 'subclass':
        title = '{} 数据集 - {} 疾病子类分布'.format(split.upper(), plant)
    axes.set_title(title, fontproperties=properties)
    axes.set_xlabel('植物大类' if level == 'class' else '疾病子类',
                    fontproperties=properties)
    axes.set_ylabel('图片数量', fontproperties=properties)
    for bar, count in zip(bars, counts):
        axes.annotate(str(count), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                      ha='center', va='bottom')
    figure.tight_layout()
    figure.savefig(str(path), dpi=CHART_DPI, format='png')
    pyplot.close(figure)
    size = path.stat().st_size
    if size <= 0:
        raise RuntimeError('Chart is empty: {}'.format(path))
    return {
        'relative_path': relative_path,
        'split': split,
        'level': level,
        'plant': plant,
        'labels': labels,
        'counts': counts,
        'size_bytes': size,
        'sha256': _sha256_file(path),
    }


def render_distribution_charts(distribution, output_dir, font):
    """Render the exact two class charts and sixty subclass charts."""
    _validate_distribution(distribution)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Chart output directory must be empty: {}'.format(output_dir))
    plants = list(distribution['class_to_idx'])
    required_text = '数据集植物大类分布疾病子类图片数量' + ''.join(plants)
    for split in SPLITS:
        for plant in plants:
            required_text += ''.join(distribution['subclass_counts'][split][plant])
    _assert_font_coverage(font, required_text)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for split in ('train', 'test'):
        artifacts.append(_draw_chart(
            output_dir, '{}/class_distribution.png'.format(split), split, 'class', None,
            plants, [distribution['class_counts'][split][plant] for plant in plants], font,
        ))
    for split in ('val', 'train', 'test'):
        for index, plant in enumerate(plants):
            counts = distribution['subclass_counts'][split][plant]
            artifacts.append(_draw_chart(
                output_dir,
                '{}/subclasses/{:02d}_{}.png'.format(split, index, _slug(plant)),
                split, 'subclass', plant, list(counts), list(counts.values()), font,
            ))
    artifacts.sort(key=lambda item: item['relative_path'])
    if len(artifacts) != 62:
        raise RuntimeError('Expected 62 charts, generated {}'.format(len(artifacts)))
    return artifacts


def validate_distribution_artifacts(statistics, output_dir):
    """Validate chart metadata, bytes, coverage, and the staged statistics JSON."""
    output_dir = Path(output_dir)
    statistics_path = output_dir / 'statistics.json'
    with statistics_path.open('r', encoding='utf-8') as file_obj:
        persisted = json.load(file_obj)
    if persisted != statistics:
        raise ValueError('Persisted statistics JSON differs from supplied statistics')
    distribution = statistics.get('distribution')
    _validate_distribution(distribution)
    charts = statistics.get('charts')
    if not isinstance(charts, list) or len(charts) != 62:
        raise ValueError('Statistics must describe exactly 62 charts')
    paths = [item.get('relative_path') for item in charts]
    if paths != sorted(paths) or len(set(paths)) != 62:
        raise ValueError('Chart paths must be sorted and unique')
    plants = list(distribution['class_to_idx'])
    expected_identities = set()
    for split in ('train', 'test'):
        expected_identities.add((split, 'class', None))
    for split in ('val', 'train', 'test'):
        for plant in plants:
            expected_identities.add((split, 'subclass', plant))
    actual_identities = set(
        (item.get('split'), item.get('level'), item.get('plant')) for item in charts
    )
    if actual_identities != expected_identities:
        raise ValueError('Charts do not provide the required split/level/class coverage')
    expected = set(['statistics.json'])
    for chart in charts:
        relative = chart['relative_path']
        if Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Unsafe chart path: {}'.format(relative))
        path = output_dir / relative
        expected.add(relative)
        if (not path.is_file() or path.stat().st_size != chart['size_bytes'] or
                _sha256_file(path) != chart['sha256']):
            raise ValueError('Chart file differs from metadata: {}'.format(relative))
        with path.open('rb') as file_obj:
            if file_obj.read(8) != b'\x89PNG\r\n\x1a\n':
                raise ValueError('Chart is not a PNG: {}'.format(relative))
        pixels = matplotlib_image.imread(str(path))
        if pixels.size == 0:
            raise ValueError('Chart PNG has no readable pixels: {}'.format(relative))
        split = chart['split']
        if chart['level'] == 'class':
            labels = list(distribution['class_to_idx'])
            counts = [distribution['class_counts'][split][name] for name in labels]
        else:
            values = distribution['subclass_counts'][split][chart['plant']]
            labels, counts = list(values), list(values.values())
        if chart['labels'] != labels or chart['counts'] != counts:
            raise ValueError('Chart values differ from distribution: {}'.format(relative))
    actual = set(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob('*') if path.is_file()
    )
    if actual != expected:
        raise ValueError('Statistics directory contains missing or extra files')

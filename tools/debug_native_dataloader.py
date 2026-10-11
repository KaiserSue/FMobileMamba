"""Audit the native training DataLoaders without starting model training."""
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / 'cmd_scripts' / 'train_native.sh'
EXPECTED_CLASSES = 200
# 直接运行本文件时使用的参数。数据在其他位置时，只需修改这里。
DATA_ROOT = None  # None 表示使用 train_native.sh 生效配置中的数据根目录。
REPORT_PATH = PROJECT_ROOT / 'tmp' / 'native_dataloader_audit.txt'
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class LoaderAudit:
    role: str
    source_dir: Path
    sample_count: int
    batch_count: int
    classes: List[str]
    class_to_idx: Dict[str, int]
    sample_examples: List[Tuple[str, int]]
    image_shape: Optional[Tuple[int, ...]] = None
    target_shape: Optional[Tuple[int, ...]] = None
    target_min_max: Optional[Tuple[int, int]] = None
    error: Optional[str] = None


@dataclass
class AuditResult:
    data_root: Optional[Path]
    dry_run_command: Optional[str] = None
    configured_class_count: Optional[int] = None
    selected_eval_split: Optional[str] = None
    train: Optional[LoaderAudit] = None
    eval: Optional[LoaderAudit] = None
    findings: List[str] = field(default_factory=list)
    status: str = 'FAIL'


def load_native_config(data_root):
    """Resolve the native launch script's effective data configuration."""
    environment = os.environ.copy()
    if data_root is not None:
        environment['DATA_ROOT'] = str(data_root)
    completed = subprocess.run(
        ['bash', str(TRAIN_SCRIPT), '--dry-run'], cwd=str(PROJECT_ROOT),
        env=environment, capture_output=True, text=True, check=True,
    )
    command = completed.stdout.strip()
    tokens = shlex.split(command)
    entry = next((index for index, token in enumerate(tokens)
                  if Path(token).name == 'run.py'), None)
    if entry is None:
        raise ValueError('原生启动命令缺少 run.py: {}'.format(command))
    arguments = tokens[entry + 1:]
    if '-c' not in arguments or '-m' not in arguments:
        raise ValueError('原生启动命令缺少配置或运行模式: {}'.format(command))
    cfg_path = arguments[arguments.index('-c') + 1]
    mode = arguments[arguments.index('-m') + 1]
    if mode != 'train':
        raise ValueError('原生启动模式不是 train: {}'.format(mode))
    opts = arguments[arguments.index('-m') + 2:]
    if any('=' not in opt for opt in opts):
        raise ValueError('原生启动命令包含无法识别的配置项: {}'.format(opts))
    from configs import get_cfg
    cfg = deepcopy(get_cfg(Namespace(cfg_path=cfg_path, mode=mode, sleep=-1, memory=-1,
                                 dist_url='env://', logger_rank=0, opts=opts)))
    if cfg.data.type != 'DefaultCLS':
        raise ValueError('仅支持原生 DefaultCLS 数据集，当前为 {}'.format(cfg.data.type))
    configured_root = Path(cfg.data.root_dir).expanduser().resolve()
    if data_root is not None and configured_root != data_root:
        raise ValueError('生效数据根目录与输入不一致: {}'.format(cfg.data.root_dir))
    if not configured_root.is_dir():
        raise FileNotFoundError('数据集根目录不存在或不是目录: {}'.format(configured_root))
    cfg.dist = False
    cfg.world_size = 1
    cfg.rank = 0
    cfg.local_rank = 0
    cfg.trainer.data.batch_size_per_gpu = cfg.trainer.data.batch_size
    if cfg.trainer.data.batch_size_test:
        cfg.trainer.data.batch_size_per_gpu_test = cfg.trainer.data.batch_size_test
    cfg.trainer.data.num_workers_per_gpu = 0
    cfg.trainer.data.persistent_workers = False
    return cfg, command


def _inspect_loader(loader, role):
    dataset = loader.dataset
    audit = LoaderAudit(
        role=role, source_dir=Path(dataset.root).resolve(),
        sample_count=len(dataset), batch_count=len(loader),
        classes=list(dataset.classes), class_to_idx=dict(dataset.class_to_idx),
        sample_examples=[(str(path), int(label)) for path, label in dataset.samples[:5]],
    )
    if audit.batch_count == 0:
        audit.error = 'DataLoader 没有可读取的 batch；请检查样本数、batch size 和 drop_last'
        return audit
    try:
        batch = next(iter(loader))
        images, targets = batch['img'], batch['target']
        audit.image_shape = tuple(images.shape)
        audit.target_shape = tuple(targets.shape)
        if targets.numel() == 0:
            audit.error = '首批标签为空'
        else:
            audit.target_min_max = (int(targets.min()), int(targets.max()))
    except Exception as error:
        audit.error = '首批取数失败: {}: {}'.format(type(error).__name__, error)
    return audit


def audit_loaders(cfg, data_root):
    """Build both native loaders and inspect their actual first batches."""
    result = AuditResult(data_root=data_root, configured_class_count=cfg.data.nb_classes)
    from data import get_loader
    configured_test_dir = (data_root / cfg.data.test_subdir).resolve()
    val_dir = (data_root / cfg.data.val_subdir).resolve()
    selected_dir = configured_test_dir if configured_test_dir.is_dir() else val_dir
    result.selected_eval_split = selected_dir.name
    try:
        train_loader, eval_loader = get_loader(cfg)
    except Exception as error:
        result.findings.append('DataLoader 构建失败: {}: {}'.format(type(error).__name__, error))
        if selected_dir == configured_test_dir and val_dir.is_dir():
            result.findings.append('当前代码优先选中了无法构建的 test/；val/ 存在，请核查 test/ 标签结构或改用 val/')
        return result
    result.train = _inspect_loader(train_loader, 'train')
    result.eval = _inspect_loader(eval_loader, 'eval')
    expected_train = (data_root / cfg.data.train_subdir).resolve()
    expected_eval = selected_dir
    if result.configured_class_count != EXPECTED_CLASSES:
        result.findings.append('生效配置类别数为 {}，预期为 200'.format(result.configured_class_count))
    if result.train.source_dir != expected_train:
        result.findings.append('训练 loader 来源不符: 预期 {}，实际 {}'.format(expected_train, result.train.source_dir))
    if result.eval.source_dir != expected_eval:
        result.findings.append('评估 loader 来源不符: 预期 {}，实际 {}'.format(expected_eval, result.eval.source_dir))
    for item in (result.train, result.eval):
        if item.sample_count == 0:
            result.findings.append('{} 样本数为 0'.format(item.role))
        if len(item.class_to_idx) != EXPECTED_CLASSES:
            result.findings.append('{} 类别数为 {}，预期为 200'.format(item.role, len(item.class_to_idx)))
        if item.error:
            result.findings.append('{} {}'.format(item.role, item.error))
        if item.target_min_max and (item.target_min_max[0] < 0 or item.target_min_max[1] >= EXPECTED_CLASSES):
            result.findings.append('{} 首批标签超出 [0, 199]: {}'.format(item.role, item.target_min_max))
    if result.train.class_to_idx != result.eval.class_to_idx:
        result.findings.append('训练与评估类别映射不一致')
    if (result.eval.source_dir == configured_test_dir and val_dir.is_dir()
            and (len(result.eval.class_to_idx) != EXPECTED_CLASSES
                 or result.train.class_to_idx != result.eval.class_to_idx)):
        result.findings.append('当前代码优先选中了有问题的 test/；val/ 存在，请核查 test/ 标签结构或改用 val/')
    if not result.findings:
        result.status = 'PASS'
    return result


def format_report(result):
    """Render all collected evidence as a readable UTF-8 report."""
    lines = [
        '原生数据加载器核验',
        '结论: {}'.format(result.status),
        '数据根目录: {}'.format(result.data_root),
        '原生 dry-run 命令: {}'.format(result.dry_run_command or '未取得'),
        '生效配置类别数: {}'.format(result.configured_class_count),
        '实际评估 split: {}'.format(result.selected_eval_split or '未取得'),
        '诊断差异: 单进程、零 worker；训练和评估 batch 大小及图像变换沿用生效配置。多卡训练时首批采样顺序与每卡 batch 可能不同。',
        '异常及提示:',
    ]
    lines.extend('  - {}'.format(item) for item in result.findings)
    if not result.findings:
        lines.append('  无')
    for item in (result.train, result.eval):
        if item is None:
            continue
        lines.extend([
            '', '[{} loader]'.format(item.role),
            '实际目录: {}'.format(item.source_dir),
            '样本数: {}'.format(item.sample_count),
            'batch 数: {}'.format(item.batch_count),
            '类别数: {}'.format(len(item.classes)),
            '完整类别映射:',
        ])
        lines.extend('  {} -> {}'.format(name, index) for name, index in sorted(item.class_to_idx.items(), key=lambda pair: pair[1]))
        lines.append('样本路径及标签:')
        lines.extend('  {} -> {}'.format(path, label) for path, label in item.sample_examples)
        lines.extend([
            '首批图像形状: {}'.format(item.image_shape),
            '首批标签形状: {}'.format(item.target_shape),
            '首批标签范围: {}'.format(item.target_min_max),
            '取数错误: {}'.format(item.error or '无'),
        ])
    lines.extend(['', '核验边界: PASS 仅代表目录、类别映射和首批读取的结构性核验；无法证明每张图片的真实语义标签正确。'])
    return '\n'.join(lines) + '\n'


def validate_native_loaders(data_root, report_path):
    """Audit native train/eval loaders and save a TXT report."""
    if data_root is not None:
        data_root = Path(data_root).expanduser().resolve()
    report_path = Path(report_path).expanduser()
    if report_path.suffix.lower() != '.txt':
        raise ValueError('报告路径必须使用 .txt 扩展名: {}'.format(report_path))
    result = AuditResult(data_root=data_root)
    try:
        cfg, load_command = load_native_config(data_root)
        result.dry_run_command = load_command
        data_root = Path(cfg.data.root_dir).expanduser().resolve()
        result = audit_loaders(cfg, data_root)
        result.dry_run_command = load_command
    except Exception as error:
        result.findings.append('核验失败: {}: {}'.format(type(error).__name__, error))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(format_report(result), encoding='utf-8')
    return result


def main():
    """Run the whole audit by executing this file directly."""
    data_root = Path(DATA_ROOT).expanduser().resolve() if DATA_ROOT is not None else None
    report_path = Path(REPORT_PATH).expanduser().resolve()
    os.chdir(PROJECT_ROOT)
    result = validate_native_loaders(data_root, report_path)
    print('核验结论: {}'.format(result.status))
    print('TXT 报告: {}'.format(report_path))
    for finding in result.findings:
        print('  - {}'.format(finding))
    return 0 if result.status == 'PASS' else 1


if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise SystemExit('此脚本不接受命令行参数；请修改文件顶部的 DATA_ROOT 和 REPORT_PATH')
    sys.exit(main())

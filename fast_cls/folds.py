"""Immutable stratified fold plans and resumable cross-validation state."""
import dataclasses
import datetime
import hashlib
import json
import os
import random
import re
import uuid
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "plant-stratified-five-fold-v1"
FOLDS = ("A", "B", "C", "D", "E")
VALIDATION_ORDER = ("E", "A", "B", "C", "D")
_NAME_PATTERN = re.compile(r"^(?P<plant>.+)-(?P<disease>.+)（(?P<index>[1-9][0-9]*)）[.]jpg$")


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    plant: str
    disease: str
    sha256: str


@dataclass(frozen=True)
class FoldAssignment:
    sample_id: str
    plant: str
    disease: str
    fold: str
    sha256: str


@dataclass(frozen=True)
class FoldPlan:
    schema_version: int
    algorithm_version: str
    seed: int
    folds: tuple
    validation_order: tuple
    train_manifest_digest: str
    class_to_idx: dict
    assignments: tuple
    plan_digest: str

    def to_dict(self):
        value = dataclasses.asdict(self)
        value["folds"] = list(self.folds)
        value["validation_order"] = list(self.validation_order)
        value["assignments"] = [dataclasses.asdict(item) for item in self.assignments]
        return value

    @classmethod
    def from_dict(cls, value):
        return cls(
            schema_version=value["schema_version"], algorithm_version=value["algorithm_version"],
            seed=value["seed"], folds=tuple(value["folds"]),
            validation_order=tuple(value["validation_order"]),
            train_manifest_digest=value["train_manifest_digest"],
            class_to_idx=dict(value["class_to_idx"]),
            assignments=tuple(FoldAssignment(**item) for item in value["assignments"]),
            plan_digest=value["plan_digest"],
        )


@dataclass(frozen=True)
class FoldContext:
    fold_index: int
    fold_count: int
    validation_fold: str
    train_folds: tuple
    fold_dir: str
    plan_digest: str
    config_digest: str


def format_fold_split(context):
    if not isinstance(context, FoldContext):
        raise TypeError("context must be FoldContext")
    if context.validation_fold not in FOLDS:
        raise ValueError("Unknown validation fold: " + context.validation_fold)
    if tuple(context.train_folds) != tuple(fold for fold in FOLDS if fold != context.validation_fold):
        raise ValueError("Training folds do not complement the validation fold")
    return "训练折={}，验证折={}".format("".join(context.train_folds), context.validation_fold)


def scan_training_samples(dataset_root):
    root = Path(dataset_root).resolve()
    manifest_path = root / "train.json"
    if not manifest_path.is_file():
        raise ValueError("Missing train.json: " + str(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError("train.json must be a nonempty list")
    disk_paths = {path.relative_to(root).as_posix() for path in (root / "train").glob("*/*.jpg")}
    samples = []
    declared = set()
    for entry in manifest:
        sample_id = entry.get("file_path")
        plant = entry.get("label")
        if not isinstance(sample_id, str) or not isinstance(plant, str):
            raise ValueError("Invalid train.json entry")
        path = root / sample_id
        if path.parent.name != plant or not path.is_file() or path.suffix != ".jpg":
            raise ValueError("Manifest path/label mismatch: " + sample_id)
        match = _NAME_PATTERN.fullmatch(path.name)
        if match is None or match.group("plant") != plant:
            raise ValueError("Noncanonical training filename: " + sample_id)
        if sample_id in declared:
            raise ValueError("Duplicate training sample: " + sample_id)
        declared.add(sample_id)
        samples.append(TrainingSample(sample_id, plant, match.group("disease"), _sha256_file(path)))
    if declared != disk_paths:
        raise ValueError("train.json and train/ files differ")
    classes = sorted({item.plant for item in samples})
    class_to_idx = {name: index for index, name in enumerate(classes)}
    samples.sort(key=lambda item: item.sample_id)
    digest_value = [{"sample_id": item.sample_id, "sha256": item.sha256} for item in samples]
    return samples, class_to_idx, _digest(digest_value)


def build_fold_plan(samples, class_to_idx, train_manifest_digest, seed,
                    validation_order=VALIDATION_ORDER):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if tuple(validation_order) != VALIDATION_ORDER:
        raise ValueError("validation_order must be E,A,B,C,D")
    groups = {}
    for sample in samples:
        groups.setdefault((sample.plant, sample.disease), []).append(sample)
    assignments = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda item: item.sample_id)
        local_seed = int(hashlib.sha256((str(seed) + "\0" + "\0".join(key)).encode()).hexdigest(), 16)
        rng = random.Random(local_seed)
        rng.shuffle(group)
        offset = rng.randrange(len(FOLDS))
        for index, sample in enumerate(group):
            assignments.append(FoldAssignment(sample.sample_id, sample.plant, sample.disease,
                                              FOLDS[(offset + index) % len(FOLDS)], sample.sha256))
    assignments.sort(key=lambda item: item.sample_id)
    body = dict(schema_version=SCHEMA_VERSION, algorithm_version=ALGORITHM_VERSION,
                seed=seed, folds=list(FOLDS), validation_order=list(validation_order),
                train_manifest_digest=train_manifest_digest, class_to_idx=dict(class_to_idx),
                assignments=[dataclasses.asdict(item) for item in assignments])
    plan = FoldPlan(assignments=tuple(assignments), plan_digest=_digest(body),
                    **{key: (tuple(value) if key in ("folds", "validation_order") else value)
                       for key, value in body.items() if key != "assignments"})
    validate_fold_plan(plan, samples, class_to_idx, train_manifest_digest, seed)
    return plan


def validate_fold_plan(plan, samples, class_to_idx, train_manifest_digest, seed):
    errors = []
    if plan.schema_version != SCHEMA_VERSION or plan.algorithm_version != ALGORITHM_VERSION:
        errors.append("version")
    if plan.seed != seed:
        errors.append("seed")
    if plan.folds != FOLDS or plan.validation_order != VALIDATION_ORDER:
        errors.append("fold_order")
    if plan.train_manifest_digest != train_manifest_digest:
        errors.append("train_manifest_digest")
    if plan.class_to_idx != class_to_idx:
        errors.append("class_to_idx")
    expected = {(item.sample_id, item.plant, item.disease, item.sha256) for item in samples}
    actual = {(item.sample_id, item.plant, item.disease, item.sha256) for item in plan.assignments}
    if expected != actual or len(actual) != len(plan.assignments):
        errors.append("assignments")
    body = plan.to_dict()
    saved_digest = body.pop("plan_digest")
    if _digest(body) != saved_digest:
        errors.append("plan_digest")
    groups = {}
    for item in plan.assignments:
        if item.fold not in FOLDS:
            errors.append("unknown_fold")
        groups.setdefault((item.plant, item.disease), []).append(item.fold)
    for values in groups.values():
        counts = [values.count(fold) for fold in FOLDS]
        if (len(values) < 5 and len(set(values)) != len(values)) or max(counts) - min(counts) > 1:
            errors.append("stratification")
    if errors:
        raise ValueError("Fold plan mismatch: " + ", ".join(sorted(set(errors))))


def load_or_create_fold_plan(dataset_root, manifest_path, seed, fold_order=VALIDATION_ORDER):
    samples, mapping, manifest_digest = scan_training_samples(dataset_root)
    path = Path(manifest_path)
    if path.exists():
        plan = FoldPlan.from_dict(json.loads(path.read_text(encoding="utf-8")))
        validate_fold_plan(plan, samples, mapping, manifest_digest, seed)
        if tuple(fold_order) != plan.validation_order:
            raise ValueError("Fold plan mismatch: fold_order")
        return plan
    plan = build_fold_plan(samples, mapping, manifest_digest, seed, fold_order)
    atomic_write_json(path, plan.to_dict())
    return plan


def fold_sample_ids(plan, validation_fold):
    if validation_fold not in plan.folds:
        raise ValueError("Unknown validation fold: " + str(validation_fold))
    train = tuple(item.sample_id for item in plan.assignments if item.fold != validation_fold)
    val = tuple(item.sample_id for item in plan.assignments if item.fold == validation_fold)
    if not train or not val:
        raise ValueError("Fold views must both be nonempty")
    return train, val


def _normalize(value):
    if isinstance(value, Namespace):
        return {key: _normalize(item) for key, item in sorted(vars(value).items())}
    if isinstance(value, dict):
        return {str(key): _normalize(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def build_config_fingerprint(cfg):
    snapshot = {
        "cfg_path": cfg.cfg_path, "seed": cfg.seed, "size": cfg.size,
        "world_size": cfg.world_size, "model": _normalize(cfg.model),
        "optim": _normalize(cfg.optim), "loss": _normalize(cfg.loss),
        "train_transforms": _normalize(cfg.data.train_transforms),
        "test_transforms": _normalize(cfg.data.test_transforms),
        "nb_classes": cfg.data.nb_classes, "epoch_full": cfg.trainer.epoch_full,
        "scheduler_kwargs": _normalize(cfg.trainer.scheduler_kwargs),
        "batch_size": cfg.trainer.data.batch_size,
        "scaler": cfg.trainer.scaler, "ema": cfg.trainer.ema,
        "fold_order": list(cfg.trainer.fold_order),
    }
    return snapshot, _digest(snapshot)


def load_or_create_run_state(state_path, plan, config_snapshot, config_digest, resume):
    path = Path(state_path)
    config_path = path.with_name("config_fingerprint.json")
    if resume:
        if not path.is_file() or not config_path.is_file():
            raise ValueError("Resume state is incomplete")
        state = json.loads(path.read_text(encoding="utf-8"))
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        errors = []
        if state.get("plan_digest") != plan.plan_digest:
            errors.append("plan_digest")
        if state.get("config_digest") != config_digest:
            changed = sorted(key for key in set(old_config) | set(config_snapshot)
                             if old_config.get(key) != config_snapshot.get(key))
            errors.append("config_digest(" + ",".join(changed) + ")")
        if errors:
            raise ValueError("Resume mismatch: " + ", ".join(errors))
        return state
    if path.exists() or config_path.exists():
        raise ValueError("Refusing to overwrite cross-validation state")
    now = _utc_now()
    state = dict(schema_version=SCHEMA_VERSION, status="running", plan_digest=plan.plan_digest,
                 config_digest=config_digest, seed=plan.seed, fold_order=list(plan.validation_order),
                 active_fold=None, completed_folds=[], fold_results={}, started_at=now,
                 updated_at=now, accumulated_duration_seconds=0.0, last_error=None)
    atomic_write_json(config_path, config_snapshot)
    atomic_write_json(path, state)
    return state


def save_run_state(state_path, state):
    if state["status"] not in ("running", "completed", "failed"):
        raise ValueError("Invalid cross-validation status")
    if set(state["completed_folds"]) != set(state["fold_results"]):
        raise ValueError("Completed folds must have complete results")
    state["updated_at"] = _utc_now()
    atomic_write_json(state_path, state)

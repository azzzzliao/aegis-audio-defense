"""Small dependency-free helpers shared across the AEGIS pipeline stages.

Kept torch-free so importing one stage script does not pull in heavy deps just
to reuse a path/jsonl helper.
"""
import json
import math
import numbers
import os
from pathlib import Path
import tempfile


def jl(path):
    """Read a jsonl file into a list of dicts."""
    with open(path, encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def row_key(row):
    return row.get("id") or row.get("sample_id") or row.get("index")


def record_id(row: dict) -> str:
    """Return the first non-empty stable ID from a manifest or result row."""
    for key in ("id", "sample_id", "index"):
        value = row.get(key)
        if value is not None:
            value = str(value).strip()
            if value:
                return value
    raise ValueError("row is missing a non-empty id, sample_id, or index")


_DEFENSE_ALPHAS = {0.0, 1.0, 1.5, 2.0, 2.5, 3.0}


def valid_defense_route(row: dict, *, require_generated: bool) -> bool:
    """Return whether a generated/reused defense row has a valid hard route."""
    gate = row.get("gate")
    alpha = row.get("chosen_alpha")
    if (not isinstance(gate, numbers.Number) or isinstance(gate, bool)
            or not math.isfinite(gate) or float(gate) not in (0.0, 1.0)):
        return False
    if (not isinstance(alpha, numbers.Number) or isinstance(alpha, bool)
            or not math.isfinite(alpha) or float(alpha) not in _DEFENSE_ALPHAS):
        return False
    gate = float(gate)
    alpha = float(alpha)
    if (gate == 0.0) != (alpha == 0.0):
        return False
    from_baseline = row.get("from_baseline")
    if type(from_baseline) is not bool:
        return False
    if require_generated and from_baseline:
        return False
    if from_baseline and gate != 0.0:
        return False
    return True


def compact_successes(
    path: Path,
    response_key: str | None,
    score_only: bool,
    *,
    require_route: bool = False,
    require_generated: bool = False,
    require_never_fire: bool = False,
) -> dict[str, dict]:
    """Atomically retain the latest successful output row for each record ID.

    Failed rows are intentionally dropped so their manifest entries are retried
    on the next ``--resume`` invocation.
    """
    path = Path(path)
    if not path.exists():
        return {}
    if not score_only and not response_key:
        raise ValueError("response_key is required unless score_only is set")

    kept = {}
    for row in jl(path):
        rid = record_id(row)
        error_free = not row.get("eval_error")
        successful = (
            error_free
            and isinstance(row.get("gate"), numbers.Number)
            and not isinstance(row.get("gate"), bool)
            if score_only
            else error_free and bool(row.get(response_key))
        )
        if successful and (require_route or require_generated or require_never_fire):
            successful = valid_defense_route(
                row, require_generated=require_generated or require_never_fire)
        if successful and require_never_fire:
            gate = row.get("gate")
            alpha = row.get("chosen_alpha")
            successful = (
                isinstance(gate, numbers.Number) and not isinstance(gate, bool)
                and float(gate) == 0.0
                and isinstance(alpha, numbers.Number) and not isinstance(alpha, bool)
                and float(alpha) == 0.0
                and row.get("from_baseline") is False
            )
        if successful:
            kept[rid] = row

    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fout:
            for rid in sorted(kept):
                fout.write(json.dumps(kept[rid], ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(temp_name, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return {rid: kept[rid] for rid in sorted(kept)}


def compact_judgments(
    source_path: Path, output_path: Path, response_key: str,
) -> dict[str, dict]:
    """Retain only current, successful Llama-Guard judgments for resume."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    if not output_path.exists():
        return {}
    source_rows = jl(source_path) if source_path.exists() else []
    source = {record_id(row): row for row in source_rows}
    kept = {}
    provenance = (response_key, "gate", "chosen_alpha", "from_baseline", "eval_error")
    for row in jl(output_path):
        rid = record_id(row)
        original = source.get(rid)
        if (original is not None
                and row.get("llamaguard_label") in ("safe", "unsafe")
                and not row.get("llamaguard_error")
                and not row.get("judge_error")
                and all(row.get(key) == original.get(key) for key in provenance)):
            kept[rid] = row

    fd, temp_name = tempfile.mkstemp(
        prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fout:
            for original in source_rows:
                rid = record_id(original)
                if rid in kept:
                    fout.write(json.dumps(kept[rid], ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(temp_name, output_path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise
    return kept


def resolve_audio(path):
    """Return ``path`` if it exists, else try it relative to ``$AEGIS_DATA_ROOT``.

    Manifests written by scripts/prepare_data.py hold absolute paths; the fallback
    lets a manifest built on one machine run on another with the data elsewhere.
    """
    if not path or os.path.exists(path):
        return path
    root = os.environ.get("AEGIS_DATA_ROOT")
    if root and os.path.exists(os.path.join(root, path)):
        return os.path.join(root, path)
    return path


def audio_path_of(row):
    """First populated audio-path field on a manifest/response row."""
    for k in ("local_audio", "qwen2_audio_input_audio", "audio", "output_wav"):
        if row.get(k):
            return row[k]
    return None

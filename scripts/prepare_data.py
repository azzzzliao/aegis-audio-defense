#!/usr/bin/env python3
"""Build the evaluation and training manifests from the downloaded datasets.

Expected layout under --data-root (see data/README.md for where to get each set):

  AudioJailbreak/convert/question/combined_output.jsonl, AudioJailbreak/audio/...
  JALM-Bench/HarmfulQuery/ADiv.parquet                (ADiv_wav/ is extracted from it)
  JALM-Bench/Audio_Originated_Jailbreak/SSJ.parquet   (SSJ_wav/ is extracted from it)
  SACRED-Bench/Multi-speaker_Dialogue/test/{json/multi-speaker_dialogue_test.json, audio/}
  XSTest/{metadata.csv, *.wav}
  Benign_train/{benign_helpful,jbb_benign}_manifest.jsonl, {benign_helpful,jbb_benign}_audio/

Writes to --out-dir (expected row counts in brackets):

  audiojailbreak.jsonl [1490]   AudioJailbreak, Origin subset
  jalm.jsonl           [946]    JALMBench ADiv (700) + SSJ (246)
  sacred_msd.jsonl     [1364]   SACRED-Bench, multi-speaker dialogue
  xstest.jsonl         [250]    TTS XSTest (benign; over-refusal only)
  benign_train.jsonl   [215]    benign training audio

  <bench>_test.jsonl            in-domain test splits (data/splits/<bench>_indomain.json)

Every row has an ``id``, an absolute ``local_audio`` path, and ``prompt``: the request
text, which the Llama-Guard judge reads (the model hears the audio plus its fixed
prompt). As in the paper's manifests, the 250 English-voice JALMBench ADiv clips carry
an empty ``prompt``; pass --fill-jalm-english-prompts to fill in their request text.
"""
import argparse
import csv
import json
import random
from pathlib import Path, PurePosixPath

EXPECTED = {"audiojailbreak": 1490, "jalm": 946, "sacred_msd": 1364, "xstest": 250,
            "benign_train": 215}


def jl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def audiojailbreak(root):
    base = root / "AudioJailbreak"
    rows = []
    for o in jl(base / "convert/question/combined_output.jsonl"):
        sp = o.get("speech_path", "")          # e.g. "./audio/jailbreakbench/..."
        rows.append({"id": f"audiojailbreak/{o.get('index')}",
                     "local_audio": (base / sp.lstrip("./")).as_posix() if sp else "",
                     "category": o.get("category"), "source": o.get("source"),
                     "attack_type": o.get("attack_type"),
                     "prompt": o.get("goal") or o.get("prompt")})
    return rows


def adiv_stem(r):
    """ADiv voice folder name: the language, or "en-<accent>[-<gender>]" for English."""
    if r["language"] != "en":
        return f"{r['language']}_{r['id']}"
    voice = f"en-{r['accent']}" + ("" if r["gender"] == "Neutral" else f"-{r['gender']}")
    return f"{voice}_{r['id']}"


def extract_wavs(parquet, out_dir, stem):
    """Write the audio column of a JALMBench parquet to <out_dir>/<stem(row)>.wav, once."""
    if out_dir.is_dir() and any(out_dir.glob("*.wav")):
        return
    import pyarrow.parquet as pq
    import soundfile as sf
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(parquet).to_pylist()
    for r in table:
        audio = r["audio"]
        sf.write(out_dir / f"{stem(r)}.wav", audio["array"], audio["sampling_rate"])
    print(f"extracted {len(table)} wavs -> {out_dir}")


def jalm(root, fill_english_prompts=False):
    import pandas as pd
    base = root / "JALM-Bench"
    extract_wavs(base / "HarmfulQuery/ADiv.parquet", base / "HarmfulQuery/ADiv_wav", adiv_stem)
    extract_wavs(base / "Audio_Originated_Jailbreak/SSJ.parquet",
                 base / "Audio_Originated_Jailbreak/SSJ_wav", lambda r: f"{r['source']}_{r['id']}")
    rows = []
    # ADiv wavs are "<language/voice>_<query id>.wav"; the unmasked request is per query id.
    adiv = pd.read_parquet(base / "HarmfulQuery/ADiv.parquet", columns=["id", "original_text"])
    adiv_text = dict(zip(adiv["id"].astype(str), adiv["original_text"]))
    for wav in sorted((base / "HarmfulQuery/ADiv_wav").glob("*.wav")):
        rows.append({"id": f"jalm_adiv/{wav.stem}", "local_audio": wav.as_posix(),
                     "source": "JALMBench-ADiv", "subset": "ADiv",
                     "prompt": ("" if wav.stem.startswith("en-") and not fill_english_prompts
                                else adiv_text.get(wav.stem.rsplit("_", 1)[-1], ""))})
    # SSJ wavs are "<source>_<id>.wav" and spell out one masked keyword; the carrier
    # sentence with "[word]" in it is kept as spoken_prompt for reference.
    ssj = pd.read_parquet(base / "Audio_Originated_Jailbreak/SSJ.parquet",
                          columns=["id", "prompt", "original_text", "source"])
    ssj_rows = {f"{r.source}_{r.id}": r for r in ssj.itertuples()}
    for wav in sorted((base / "Audio_Originated_Jailbreak/SSJ_wav").glob("*.wav")):
        r = ssj_rows.get(wav.stem)
        carrier = str(r.prompt) if r is not None else ""
        rows.append({"id": f"jalm_ssj/{wav.stem}", "local_audio": wav.as_posix(),
                     "source": "JALMBench-SSJ", "subset": "SSJ",
                     "prompt": str(r.original_text) if r is not None else "",
                     "spoken_prompt": carrier, "ssj_masked": "[word]" in carrier})
    return rows


def sacred_msd(root):
    base = root / "SACRED-Bench/Multi-speaker_Dialogue/test"
    with open(base / "json/multi-speaker_dialogue_test.json", encoding="utf-8") as f:
        items = json.load(f)
    rows = []
    for o in items:
        # audio_path is the upstream authors' absolute path; keep "<category>/<file>".
        # Upstream omits the category folder for the 97 "01-Illegal_Activitiy" rows
        # (".../audio/<n>_tts.wav"), the only category missing from the paths.
        p = PurePosixPath(o["audio_path"])
        category = p.parent.name if p.parent.name != "audio" else "01-Illegal_Activitiy"
        rows.append({"id": f"sacred_msd/{category}_{o['index']}",
                     "local_audio": (base / "audio" / category / p.name).as_posix(),
                     "source": "SACRED-MSD", "category": category,
                     "prompt": o.get("Question") or o.get("Changed Question") or ""})
    return sorted(rows, key=lambda r: r["id"])


def xstest(root):
    base = root / "XSTest"
    with open(base / "metadata.csv", encoding="utf-8") as f:
        return [{"id": f"xstest/{o['id']}", "local_audio": (base / o["file_name"]).as_posix(),
                 "category": o.get("type"), "source": "xstest", "label_intent": o.get("label"),
                 "prompt": o.get("text")} for o in csv.DictReader(f)]


def benign_train(root):
    base = root / "Benign_train"
    rows = []
    for subset in ("benign_helpful", "jbb_benign"):
        for o in jl(base / f"{subset}_manifest.jsonl"):
            rows.append({"id": o["sample_id"], "sample_id": o["sample_id"],
                         "local_audio": (base / f"{subset}_audio" / f"{o['sample_id']}.wav").as_posix(),
                         "prompt": o.get("text", ""), "dataset": o.get("dataset"),
                         "category": o.get("category") or o.get("topic")})
    return rows


def group_split(rows, train_frac=0.5, seed=42):
    """Deterministic split grouped by prompt string (rows sharing a prompt stay together;
    an empty prompt is its own group). This produced data/splits/*.json."""
    groups = {}
    for o in rows:
        key = (o.get("prompt") or "").strip() or f"__id__{o['id']}"
        groups.setdefault(key, []).append(str(o["id"]))
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n_train = round(len(keys) * train_frac)
    train = sorted(i for k in keys[:n_train] for i in groups[k])
    test = sorted(i for k in keys[n_train:] for i in groups[k])
    return train, test


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out-dir", default="data/manifests", type=Path)
    ap.add_argument("--splits", default="data/splits", type=Path,
                    help="in-domain split id lists (shipped with the repo)")
    ap.add_argument("--fill-jalm-english-prompts", action="store_true",
                    help="give the English-voice ADiv clips their request text (the paper's "
                         "manifests left it empty)")
    ap.add_argument("--make-splits", action="store_true",
                    help="regenerate the in-domain splits instead of using the shipped ones")
    a = ap.parse_args()
    root = a.data_root.resolve()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    builders = {"audiojailbreak": audiojailbreak,
                "jalm": lambda r: jalm(r, a.fill_jalm_english_prompts), "sacred_msd": sacred_msd,
                "xstest": xstest, "benign_train": benign_train}
    ok = True
    written = {}
    for name, build in builders.items():
        rows = build(root)
        kept = [r for r in rows if r["local_audio"] and Path(r["local_audio"]).exists()]
        write(a.out_dir / f"{name}.jsonl", kept)
        written[name] = kept
        flag = "" if len(kept) == EXPECTED[name] else f"  <-- expected {EXPECTED[name]}"
        ok &= not flag
        print(f"{name:15s} {len(kept):5d} rows (missing audio: {len(rows) - len(kept)}){flag}")

    # in-domain test manifests
    for bench, name in (("audiojailbreak", "audiojailbreak"), ("jalm", "jalm"), ("sacred", "sacred_msd")):
        split_path = a.splits / f"{bench}_indomain.json"
        if a.make_splits:
            train, test = group_split(written[name])
            split_path.parent.mkdir(parents=True, exist_ok=True)
            split_path.write_text(json.dumps({"benchmark": bench, "seed": 42, "train_frac": 0.5,
                                              "train_ids": train, "test_ids": test}, indent=1))
        test_ids = set(json.loads(split_path.read_text())["test_ids"])
        test_rows = [r for r in written[name] if r["id"] in test_ids]
        write(a.out_dir / f"{bench}_test.jsonl", test_rows)
        print(f"{bench + '_test':15s} {len(test_rows):5d} rows (in-domain test split)")
    if not ok:
        print("Row counts differ from the paper; check the dataset layout (data/README.md).")


if __name__ == "__main__":
    main()

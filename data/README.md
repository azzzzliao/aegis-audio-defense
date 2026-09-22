# Data

AEGIS is evaluated on three attack benchmarks (AJail, JALM, SACRED). XSTest measures over-refusal, and a small
benign set is used for training. Download them into `data/` with this layout:

| Folder | Source | Role | Rows |
|---|---|---|---:|
| `AudioJailbreak/` | [`MBZUAI/AudioJailbreak`](https://huggingface.co/datasets/MBZUAI/AudioJailbreak) (`audio/`, `convert/`) | attack, *Origin* subset | 1,490 |
| `JALM-Bench/` | [`AnonymousUser000/JALMBench`](https://huggingface.co/datasets/AnonymousUser000/JALMBench) (`HarmfulQuery/ADiv.parquet`, `Audio_Originated_Jailbreak/SSJ.parquet`) | attack, ADiv + SSJ | 946 |
| `SACRED-Bench/` | [`tsinghua-ee/SACRED-Bench`](https://huggingface.co/datasets/tsinghua-ee/SACRED-Bench) (`Multi-speaker_Dialogue/test/`) | attack, multi-speaker dialogue | 1,364 |
| `XSTest/` | [`audio-safety-group/XSTest-Benige`](https://huggingface.co/datasets/audio-safety-group/XSTest-Benige) | benign, over-refusal | 250 |
| `Benign_train/` | [`audio-safety-group/Train_Benign`](https://huggingface.co/datasets/audio-safety-group/Train_Benign) | benign, training only | 215 |

```bash
huggingface-cli download MBZUAI/AudioJailbreak --repo-type dataset --local-dir data/AudioJailbreak --include "audio/*" "convert/*"
huggingface-cli download AnonymousUser000/JALMBench --repo-type dataset --local-dir data/JALM-Bench \
  --include "HarmfulQuery/ADiv.parquet" "Audio_Originated_Jailbreak/SSJ.parquet"
huggingface-cli download tsinghua-ee/SACRED-Bench --repo-type dataset --local-dir data/SACRED-Bench --include "Multi-speaker_Dialogue/test/*"
huggingface-cli download audio-safety-group/XSTest-Benige --repo-type dataset --local-dir data/XSTest
huggingface-cli download audio-safety-group/Train_Benign --repo-type dataset --local-dir data/Benign_train
python scripts/prepare_data.py --data-root data --out-dir data/manifests
```

`prepare_data.py` extracts the JALMBench audio from the parquet files into `ADiv_wav/` and
`SSJ_wav/`. It then writes one manifest per set to `data/manifests/`, plus the in-domain
test splits (`<bench>_test.jsonl`) from the id lists in `data/splits/`.

Notes:

- AudioJailbreak lists 1,495 prompts; 5 have no audio upstream and are skipped.
- `prompt` in every attack manifest is the unmasked request text, which the Llama-Guard judge
  reads. As in the paper's manifests, the 250 English-voice ADiv clips carry an empty
  `prompt`; `--fill-jalm-english-prompts` fills in their request text.
- `data/splits/<bench>_indomain.json` are the in-domain train/test ids used in the paper
  (50/50, seed 42, grouped by prompt). `--make-splits` regenerates them from the manifests.
- JALMBench SSJ audio spells out one masked keyword. The SSJ rows also carry
  `spoken_prompt`, the carrier sentence with `[word]` in it, for reference.
- XSTest audio is TTS (Qwen3-TTS) of the 250 safe XSTest prompts.

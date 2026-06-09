# speech-cutter-with-silero

**Cut long recordings into per-utterance clips using the Silero voice-activity
detector (VAD).**

This repository is a small batch tool for preparing speech corpora. It walks a
directory tree of audio files, runs [Silero VAD](https://github.com/snakers4/silero-vad)
on each one, and writes every detected speech segment longer than a configurable
threshold as its own WAV file — mirroring the input folder structure under the
output directory. It is handy for slicing audiobooks, podcasts, or interview
recordings into individual utterances.

---

## How it works

For each audio file the pipeline is:

1. **Decode → mono → resample** to 16 kHz in a single `librosa.load` call (Silero
   VAD operates at 16 kHz; 8 kHz is also supported).
2. **Detect speech** with `get_speech_timestamps`, governed by:
   - `--threshold` — speech-probability threshold in `[0, 1]` (default `0.5`); higher
     is stricter.
   - `--min-silence-ms` — the minimum silence (in ms) that splits one utterance from
     the next (default `20`).
3. **Filter short fragments** — segments shorter than `--min-audio-length` seconds
   (default `2.0`) are discarded.
4. **Save** each surviving segment to `OUT/<mirrored/dirs>/<stem>_<index>.wav`.

Decoding is done directly from the source file, so **no intermediate files are
written into the input tree**.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.8+. The Silero VAD weights ship inside the `silero-vad` pip
package, so there is no `torch.hub` download at runtime. Decoding some compressed
formats (e.g. certain `.mp3` files) may additionally require **ffmpeg** on your
`PATH`, which librosa uses through its audioread backend.

## Usage

### Quick check (synthetic data)

Run the whole pipeline end-to-end on a tiny synthesised audio tree — no real
recordings and no Weights & Biases account needed. This is a good way to confirm
your install works:

```bash
python main.py --demo
```

This writes a couple of synthetic WAV files under `demo_data/` and segments them
into `demo_data/_out/`. Silero is trained on real speech, so the synthetic tones
may yield few or zero detections — the demo is a smoke test of the *pipeline*, not
of detection quality.

### Run on your own audio

```bash
python main.py --input path/to/audio_dir --output path/to/results
```

The input may be any folder tree; its relative structure is reproduced under the
output directory. CLI options:

`--input`, `--output`, `--min-audio-length`, `--threshold`, `--min-silence-ms`,
`--sample-rate`, `--extensions`, `--wandb`, `--wandb-project`, `--wandb-name`,
`--demo`, `--demo-dir`.

Weights & Biases logging is **off by default**; pass `--wandb` to enable per-file
and summary logging (`pip install wandb` first).

### Library use

```python
from main import run, load_model

# Optional: load the model once and reuse it across calls.
model = load_model()

stats = run("audio_dir", "results", model=model,
            min_audio_length=2.0, threshold=0.5, min_silence_ms=20)

print(stats["total_fragments"], stats["mean_fragment_length"])
```

`segment_file(path, out_dir, model, ...)` exposes the same logic for a single file
and returns `(num_fragments, lengths_in_seconds)`.

## Input / output data contract

- **Input:** any directory tree containing audio files. Recognised extensions
  default to `.mp3, .wav, .flac, .m4a, .ogg, .opus` (override with `--extensions`).
  Channels and sample rate are arbitrary — everything is converted to mono at the
  target sample rate.
- **Output:** the input's relative folder structure is mirrored under `--output`.
  Each detected utterance is written as `<source-stem>_<index>.wav`, mono, PCM at
  `--sample-rate`. A folder is created only when at least one fragment is produced.
- **Return value (library):** `run(...)` returns a dict with `total_files`,
  `total_fragments`, `mean_fragment_length` (0.0 when nothing is found), and the
  per-file records.

## Notes & limitations

This is a small research/utility script. A few things to be aware of:

- The `--demo` mode uses synthetic tones, which are not speech; expect few or no
  detections. Use real recordings to see meaningful segmentation.
- VAD behaviour is sensitive to `--threshold` and `--min-silence-ms`. Lower the
  threshold to catch quieter speech; raise `--min-silence-ms` to merge segments
  separated by short pauses.
- Long files are loaded fully into memory before segmentation.

## Acknowledgements

Built on **Silero VAD** by Silero Team — <https://github.com/snakers4/silero-vad>.
If you use this tool in academic work, please cite Silero VAD accordingly.

## License

Released under the [MIT License](LICENSE).

---

If you like this contribution, please give it a ⭐. Comments and ideas are
welcome — find my contact details at <https://www.mateocamara.com>.

"""Cut long recordings into per-utterance clips with Silero VAD.

This tool walks an input directory tree, runs the Silero voice-activity-detection
(VAD) model on every audio file, and writes each detected speech segment longer
than ``min_audio_length`` seconds as its own WAV file. The input's folder
structure is mirrored under the output directory, so a recording at
``IN/<sub>/<dirs>/talk.mp3`` produces ``OUT/<sub>/<dirs>/talk_0.wav``,
``talk_1.wav``, ... -- one file per utterance.

Two ways to use this file:

* As a library -- ``from main import run, load_model`` and call
  ``run(input_dir, output_dir)`` on your own folder of audio.
* As a script -- ``python main.py --input IN --output OUT`` to process a folder,
  or ``python main.py --demo`` to synthesise a tiny audio tree and confirm the
  whole load -> VAD -> save pipeline runs end-to-end (no real data, no W&B).
"""

import argparse
import os

import librosa
import numpy as np
import soundfile as sf
import torch
from silero_vad import get_speech_timestamps, load_silero_vad
from tqdm import tqdm

# Default parameters (overridable via the CLI or the ``run``/``segment_file`` args).
MIN_AUDIO_LENGTH = 2.0       # seconds; speech segments shorter than this are dropped
THRESHOLD = 0.5              # Silero speech-probability threshold in [0, 1]
MIN_SILENCE_MS = 20          # minimum silence (ms) that separates two segments
TARGET_SAMPLE_RATE = 16000   # Silero VAD operates at 16 kHz (8 kHz also supported)

# Audio file extensions picked up while walking the input tree.
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus")


def load_model():
    """Load and return the Silero VAD model (weights ship with the pip package)."""
    return load_silero_vad()


def segment_file(file_path, output_dir, model, *, min_audio_length=MIN_AUDIO_LENGTH,
                 threshold=THRESHOLD, min_silence_ms=MIN_SILENCE_MS,
                 sample_rate=TARGET_SAMPLE_RATE):
    """Detect speech in one audio file and save the long-enough segments.

    The file is decoded to mono and resampled to ``sample_rate`` in a single
    ``librosa.load`` call, run through Silero VAD, and every segment whose
    duration is at least ``min_audio_length`` seconds is written to
    ``output_dir`` as ``<stem>_<index>.wav``.

    Returns:
        ``(num_fragments, lengths)`` where ``lengths`` is the list of saved
        segment durations in seconds.
    """
    # Decode -> mono -> resample in one step (broad format support via the
    # libsndfile / audioread backends). No intermediate files touch the input tree.
    audio, _ = librosa.load(file_path, sr=sample_rate, mono=True)
    wav = torch.from_numpy(audio)

    timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=sample_rate,
        threshold=threshold,
        min_silence_duration_ms=min_silence_ms,
    )

    # Keep only segments at least ``min_audio_length`` seconds long.
    fragments = [
        t for t in timestamps
        if (t["end"] - t["start"]) / sample_rate >= min_audio_length
    ]

    stem = os.path.splitext(os.path.basename(file_path))[0]
    lengths = []
    if fragments:
        os.makedirs(output_dir, exist_ok=True)
    for idx, t in enumerate(fragments):
        fragment = audio[t["start"]:t["end"]]
        out_file = os.path.join(output_dir, f"{stem}_{idx}.wav")
        sf.write(out_file, fragment, sample_rate)
        lengths.append((t["end"] - t["start"]) / sample_rate)

    return len(fragments), lengths


def iter_audio_files(input_dir, extensions=AUDIO_EXTS):
    """Yield ``(file_path, relative_dir)`` for every audio file under ``input_dir``.

    ``relative_dir`` is the file's directory relative to ``input_dir`` (``"."``
    for files in the root), used to mirror the tree under the output directory.
    """
    exts = tuple(e.lower() for e in extensions)
    for root, _dirs, files in os.walk(input_dir):
        rel_dir = os.path.relpath(root, input_dir)
        for name in sorted(files):
            if name.lower().endswith(exts):
                yield os.path.join(root, name), rel_dir


def run(input_dir, output_dir, *, model=None, min_audio_length=MIN_AUDIO_LENGTH,
        threshold=THRESHOLD, min_silence_ms=MIN_SILENCE_MS,
        sample_rate=TARGET_SAMPLE_RATE, extensions=AUDIO_EXTS,
        use_wandb=False, wandb_project="audio_segmentation",
        wandb_name="silero_vad_audio_cutting"):
    """Segment every audio file under ``input_dir`` into a mirror of ``output_dir``.

    Walks ``input_dir`` recursively; for each audio file the relative folder
    structure is reproduced under ``output_dir`` and the detected speech segments
    are written there. Optionally logs per-file and summary statistics to Weights
    & Biases (disabled by default -- ``wandb`` is only imported when ``use_wandb``).

    Returns:
        A stats dict with ``total_files``, ``total_fragments``,
        ``mean_fragment_length`` (0.0 when nothing was found) and per-file records.
    """
    if model is None:
        model = load_model()

    wandb = None
    if use_wandb:
        import wandb as _wandb  # imported lazily so the dependency stays optional
        wandb = _wandb
        wandb.init(project=wandb_project, name=wandb_name)
        wandb.config.update({
            "min_audio_length": min_audio_length,
            "threshold": threshold,
            "min_silence_ms": min_silence_ms,
            "target_sample_rate": sample_rate,
        })

    files = list(iter_audio_files(input_dir, extensions))
    file_stats = []
    for file_path, rel_dir in tqdm(files, desc="Segmenting"):
        out_dir = output_dir if rel_dir == "." else os.path.join(output_dir, rel_dir)
        num_fragments, lengths = segment_file(
            file_path, out_dir, model,
            min_audio_length=min_audio_length, threshold=threshold,
            min_silence_ms=min_silence_ms, sample_rate=sample_rate,
        )
        file_stats.append({"file": file_path, "num_fragments": num_fragments,
                           "fragment_lengths": lengths})
        if wandb is not None:
            wandb.log({"num_fragments": num_fragments,
                       "fragment_lengths": wandb.Histogram(lengths) if lengths else 0})

    all_lengths = [length for s in file_stats for length in s["fragment_lengths"]]
    total_fragments = sum(s["num_fragments"] for s in file_stats)
    # Guard against an empty result set (no speech found -> no division by zero).
    mean_length = sum(all_lengths) / len(all_lengths) if all_lengths else 0.0

    summary = {
        "total_files": len(file_stats),
        "total_fragments": total_fragments,
        "mean_fragment_length": mean_length,
        "files": file_stats,
    }

    if wandb is not None:
        log = {"total_fragments": total_fragments, "mean_fragment_length": mean_length}
        if all_lengths:
            log["all_fragment_lengths"] = wandb.Histogram(all_lengths)
        wandb.log(log)
        wandb.finish()

    print(f"Processed {summary['total_files']} file(s); {total_fragments} fragment(s); "
          f"mean length {mean_length:.2f}s.")
    return summary


def make_synthetic_dataset(root, *, sample_rate=TARGET_SAMPLE_RATE, seed=0):
    """Create a small synthetic audio tree under ``root`` and return its path.

    Writes a couple of WAV files (in a nested sub-tree) made of tone/noise bursts
    separated by silence, so ``--demo`` can exercise the full load -> VAD -> save
    pipeline without real recordings. Silero is trained on speech, so synthetic
    tones may yield few or zero detections; the demo is a smoke test of the
    pipeline, not of detection quality.
    """
    rng = np.random.default_rng(seed)
    target = os.path.join(root, "speaker_1", "book_1")
    os.makedirs(target, exist_ok=True)

    def burst_track(n_bursts):
        silence = np.zeros(int(0.6 * sample_rate), dtype=np.float32)
        chunks = [silence]
        for _ in range(n_bursts):
            dur = int(2.5 * sample_rate)
            t = np.arange(dur) / sample_rate
            tone = (0.3 * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)
            noise = (0.05 * rng.standard_normal(dur)).astype(np.float32)
            chunks.append(tone + noise)
            chunks.append(silence)
        return np.concatenate(chunks)

    for i, n in enumerate((2, 3)):
        sf.write(os.path.join(target, f"sample_{i}.wav"), burst_track(n), sample_rate)
    return root


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cut long recordings into per-utterance clips with Silero VAD.")
    parser.add_argument("--input", help="input directory tree of audio files")
    parser.add_argument("--output", help="output directory (input structure is mirrored)")
    parser.add_argument("--min-audio-length", type=float, default=MIN_AUDIO_LENGTH,
                        help="drop speech segments shorter than this (seconds)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Silero speech-probability threshold in [0, 1]")
    parser.add_argument("--min-silence-ms", type=int, default=MIN_SILENCE_MS,
                        help="minimum silence (ms) that separates two segments")
    parser.add_argument("--sample-rate", type=int, default=TARGET_SAMPLE_RATE,
                        help="target sample rate; Silero supports 16000 or 8000")
    parser.add_argument("--extensions", default=",".join(AUDIO_EXTS),
                        help="comma-separated audio extensions to process")
    parser.add_argument("--wandb", action="store_true",
                        help="log statistics to Weights & Biases (off by default)")
    parser.add_argument("--wandb-project", default="audio_segmentation",
                        help="W&B project name (used with --wandb)")
    parser.add_argument("--wandb-name", default="silero_vad_audio_cutting",
                        help="W&B run name (used with --wandb)")
    parser.add_argument("--demo", action="store_true",
                        help="synthesise a tiny audio tree and run the pipeline end-to-end")
    parser.add_argument("--demo-dir", default="demo_data",
                        help="where to create the synthetic --demo input tree")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extensions = tuple(
        e if e.startswith(".") else "." + e
        for e in (x.strip() for x in args.extensions.split(",")) if e
    )

    if args.demo:
        demo_input = make_synthetic_dataset(args.demo_dir, sample_rate=args.sample_rate)
        demo_output = args.output or os.path.join(args.demo_dir, "_out")
        print(f"[demo] synthetic audio under {demo_input!r} -> {demo_output!r}")
        run(demo_input, demo_output,
            min_audio_length=args.min_audio_length, threshold=args.threshold,
            min_silence_ms=args.min_silence_ms, sample_rate=args.sample_rate,
            extensions=extensions, use_wandb=args.wandb,
            wandb_project=args.wandb_project, wandb_name=args.wandb_name)
    elif args.input and args.output:
        run(args.input, args.output,
            min_audio_length=args.min_audio_length, threshold=args.threshold,
            min_silence_ms=args.min_silence_ms, sample_rate=args.sample_rate,
            extensions=extensions, use_wandb=args.wandb,
            wandb_project=args.wandb_project, wandb_name=args.wandb_name)
    else:
        raise SystemExit(
            "Nothing to do. Provide --input INPUT_DIR --output OUTPUT_DIR to process "
            "a folder of audio, or run with --demo to synthesise a tiny audio tree "
            "and confirm the pipeline works end-to-end."
        )

"""
Adaptive noise cancellation using the padasip library (LMS and RLS).

Dependencies:
    pip install padasip numpy matplotlib  (matplotlib optional for plots)

Inputs (defaults):
    --noisy aud/audio.wav           noisy speech
    --noise aud/audio_noise.wav     noise reference

Outputs (under outputs_padasip/):
    outputs_padasip/audio_lms_pada.wav   (if LMS or both)
    outputs_padasip/audio_rls_pada.wav   (if RLS or both)
    outputs_padasip/signals_pada.png     (if matplotlib available and not --skip-plot)
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import padasip as pa
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "padasip is required. Install with: pip install padasip"
    ) from exc


def read_wav(path: Path) -> Tuple[int, np.ndarray]:
    """Return sample_rate and mono float samples in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16-bit PCM supported, got {sampwidth * 8} bits")

    samples = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if channels == 2:
        samples = samples.reshape(-1, 2).mean(axis=1)

    norm = 32768.0
    return rate, samples / norm


def write_wav(path: Path, rate: int, samples: np.ndarray) -> None:
    """Write mono 16-bit PCM wav from normalized floats."""
    if samples.size == 0:
        raise ValueError("No samples to write.")
    peak = np.max(np.abs(samples))
    if peak > 1.0:
        samples = samples / peak
    int_samples = np.clip(samples, -1.0, 1.0)
    int_samples = (int_samples * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(int_samples.tobytes())


def preprocess(noisy: np.ndarray, noise: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """DC removal and peak normalization."""
    def _proc(sig: np.ndarray) -> np.ndarray:
        sig = sig - np.mean(sig)
        peak = np.max(np.abs(sig))
        return sig / peak if peak > 0 else sig

    length = min(len(noisy), len(noise))
    return _proc(noisy[:length]), _proc(noise[:length])


def lms_padasip(
    noisy: np.ndarray, noise: np.ndarray, order: int, mu: float
) -> np.ndarray:
    X = pa.preprocess.input_from_history(noise, order)
    d = noisy[order - 1 :]
    f = pa.filters.FilterLMS(n=order, mu=mu, w="random")
    y, e, _ = f.run(d, X)
    return e  # error is the cleaned speech


def rls_padasip(
    noisy: np.ndarray, noise: np.ndarray, order: int, lam: float, delta: float
) -> np.ndarray:
    X = pa.preprocess.input_from_history(noise, order)
    d = noisy[order - 1 :]
    # padasip FilterRLS uses mu as forgetting factor and eps as initial delta.
    f = pa.filters.FilterRLS(n=order, mu=lam, eps=delta, w="random")
    y, e, _ = f.run(d, X)
    return e


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return math.sqrt(float(np.mean(x * x)))


def plot_signals(rate: int, original, lms_clean=None, rls_clean=None) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    duration_s = min(2.0, len(original) / rate)
    n = int(duration_s * rate)
    t = np.arange(n) / rate

    rows = 1 + int(lms_clean is not None) + int(rls_clean is not None)
    plt.figure(figsize=(10, 2 * rows + 1))
    idx = 1

    plt.subplot(rows, 1, idx)
    plt.plot(t, original[:n], linewidth=0.7)
    plt.title("Original noisy speech")
    plt.ylabel("Amplitude")
    idx += 1

    if lms_clean is not None:
        plt.subplot(rows, 1, idx)
        plt.plot(t[: len(lms_clean[:n])], lms_clean[:n], color="green", linewidth=0.7)
        plt.title("LMS (padasip) cleaned")
        plt.ylabel("Amplitude")
        idx += 1

    if rls_clean is not None:
        plt.subplot(rows, 1, idx)
        plt.plot(t[: len(rls_clean[:n])], rls_clean[:n], color="orange", linewidth=0.7)
        plt.title("RLS (padasip) cleaned")
        plt.ylabel("Amplitude")

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig("outputs_padasip/signals_pada.png", dpi=150)
    plt.close()
    print("Saved plot to outputs_padasip/signals_pada.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive noise cancellation via padasip")
    parser.add_argument("--noisy", default="aud/audio.wav", help="Path to noisy speech")
    parser.add_argument("--noise", default="aud/audio_noise.wav", help="Path to noise ref")
    parser.add_argument("--algo", choices=["lms", "rls", "both"], default="both")
    parser.add_argument("--lms-order", type=int, default=32)
    parser.add_argument("--lms-mu", type=float, default=0.01)
    parser.add_argument("--rls-order", type=int, default=16)
    parser.add_argument("--rls-lam", type=float, default=0.995)
    parser.add_argument("--rls-delta", type=float, default=0.01)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    base = Path(".")
    out_dir = base / "outputs_padasip"
    out_dir.mkdir(exist_ok=True)

    rate_noisy, noisy = read_wav(base / args.noisy)
    rate_noise, noise = read_wav(base / args.noise)
    if rate_noisy != rate_noise:
        raise ValueError("Sample rates must match between noisy and noise files.")

    noisy, noise = preprocess(noisy, noise)
    orig_rms = rms(noisy)

    lms_out = None
    rls_out = None

    if args.algo in ("lms", "both"):
        lms_out = lms_padasip(noisy, noise, order=args.lms_order, mu=args.lms_mu)
        write_wav(out_dir / "audio_lms_pada.wav", rate_noisy, lms_out)
        print(
            f"LMS done: order={args.lms_order} mu={args.lms_mu}, "
            f"RMS={rms(lms_out):.4f}"
        )

    if args.algo in ("rls", "both"):
        rls_out = rls_padasip(
            noisy, noise, order=args.rls_order, lam=args.rls_lam, delta=args.rls_delta
        )
        write_wav(out_dir / "audio_rls_pada.wav", rate_noisy, rls_out)
        print(
            f"RLS done: order={args.rls_order} lam={args.rls_lam} delta={args.rls_delta}, "
            f"RMS={rms(rls_out):.4f}"
        )

    if not args.skip_plot:
        plot_signals(rate_noisy, noisy, lms_out, rls_out)

    print(f"Input RMS: {orig_rms:.4f}")


if __name__ == "__main__":
    main()


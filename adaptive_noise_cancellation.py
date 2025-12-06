"""
Adaptive noise cancellation using LMS and RLS filters without external DSP
libraries. Requires only the Python standard library. Matplotlib is used
optionally for visualization; if it is unavailable the script will skip plots.

Input files expected by default:
    - aud/audio.wav          : noisy speech
    - aud/audio_noise.wav    : noise reference signal

Outputs (written under outputs_basic/):
    - outputs_basic/audio_lms.wav      : speech cleaned with LMS
    - outputs_basic/audio_rls.wav      : speech cleaned with RLS
    - outputs_basic/signals.png        : visualization of original vs cleaned
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path
from typing import Iterable, List, Tuple


def read_wav(path: Path) -> Tuple[int, List[float]]:
    """Return sample_rate and mono float samples in [-1, 1]."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        frames = wf.getnframes()
        raw = wf.readframes(frames)

    if sampwidth != 2:
        raise ValueError(f"Only 16‑bit PCM supported, got {sampwidth * 8} bits")

    samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if channels == 2:
        # Average the two channels to mono.
        mono = [(samples[i] + samples[i + 1]) / 2 for i in range(0, len(samples), 2)]
    else:
        mono = list(samples)

    norm = 32768.0
    return rate, [s / norm for s in mono]


def write_wav(path: Path, rate: int, samples: List[float]) -> None:
    """Write mono 16-bit PCM wav from normalized floats."""
    # Avoid clipping.
    peak = max(max((abs(s) for s in samples), default=0.0), 1e-12)
    if peak > 1.0:
        scale = 1.0 / peak
        samples = [s * scale for s in samples]
    int_samples = [int(max(-1.0, min(1.0, s)) * 32767) for s in samples]
    raw = struct.pack("<" + "h" * len(int_samples), *int_samples)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(raw)


def lms_filter(
    noise: List[float], target: List[float], order: int = 12, mu: float = 0.005
) -> List[float]:
    """Least-mean-square adaptive filter returning the error (cleaned output)."""
    w = [0.0] * order
    out: List[float] = []
    for n in range(len(target)):
        x = [noise[n - k] if n - k >= 0 else 0.0 for k in range(order)]
        y = sum(w[i] * x[i] for i in range(order))
        e = target[n] - y
        for i in range(order):
            w[i] += 2 * mu * e * x[i]
        out.append(e)
    return out


def rls_filter(
    noise: List[float],
    target: List[float],
    order: int = 8,
    lam: float = 0.99,
    delta: float = 0.1,
) -> List[float]:
    """Recursive-least-square adaptive filter returning the error output."""
    w = [0.0] * order
    # Initialize inverse correlation matrix.
    P = [[0.0] * order for _ in range(order)]
    for i in range(order):
        P[i][i] = 1.0 / delta

    out: List[float] = []
    for n in range(len(target)):
        x = [noise[n - k] if n - k >= 0 else 0.0 for k in range(order)]
        # k = P x / (lambda + x^T P x)
        Px = [sum(P[i][j] * x[j] for j in range(order)) for i in range(order)]
        xPx = sum(x[i] * Px[i] for i in range(order))
        denom = lam + xPx
        k = [Px[i] / denom for i in range(order)]

        y = sum(w[i] * x[i] for i in range(order))
        e = target[n] - y
        for i in range(order):
            w[i] += k[i] * e

        # P = (P - k x^T P) / lambda
        xTP = [sum(x[j] * P[j][i] for j in range(order)) for i in range(order)]
        for i in range(order):
            for j in range(order):
                P[i][j] = (P[i][j] - k[i] * xTP[j]) / lam

        out.append(e)
    return out


def rms(signal: List[float]) -> float:
    """Root-mean-square level."""
    if not signal:
        return 0.0
    return math.sqrt(sum(s * s for s in signal) / len(signal))


def remove_dc(signal: List[float]) -> List[float]:
    """Remove DC component by subtracting mean."""
    if not signal:
        return signal
    mean = sum(signal) / len(signal)
    return [s - mean for s in signal]


def normalize(signal: List[float], peak_target: float = 0.99) -> List[float]:
    """Scale to a target peak magnitude."""
    if not signal:
        return signal
    peak = max(abs(s) for s in signal)
    if peak == 0:
        return signal
    scale = peak_target / peak
    return [s * scale for s in signal]


def plot_signals(rate: int, original, lms_clean, rls_clean) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    segment = min(len(original), rate * 2)  # first 2 seconds
    t = [i / rate for i in range(segment)]
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t, original[:segment], linewidth=0.7)
    plt.title("Original noisy speech")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 2)
    plt.plot(t, lms_clean[:segment], color="green", linewidth=0.7)
    plt.title("LMS cleaned output")
    plt.ylabel("Amplitude")

    plt.subplot(3, 1, 3)
    plt.plot(t, rls_clean[:segment], color="orange", linewidth=0.7)
    plt.title("RLS cleaned output")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig("outputs_basic/signals.png", dpi=150)
    plt.close()
    print("Saved plot to outputs_basic/signals.png")


def run_once(
    noisy: List[float],
    noise: List[float],
    lms_order: int,
    lms_mu: float,
    rls_order: int,
    rls_lam: float,
    rls_delta: float,
) -> Tuple[List[float], List[float]]:
    lms_clean = lms_filter(noise, noisy, order=lms_order, mu=lms_mu)
    rls_clean = rls_filter(noise, noisy, order=rls_order, lam=rls_lam, delta=rls_delta)
    return lms_clean, rls_clean


def grid_test(
    noisy: List[float],
    noise: List[float],
    lms_grid: Iterable[Tuple[int, float]],
    rls_grid: Iterable[Tuple[int, float, float]],
) -> Tuple[Tuple[int, float], float, Tuple[int, float, float], float]:
    """Return best LMS and RLS parameter sets by RMS of output."""
    best_lms = (None, float("inf"))  # type: ignore
    best_rls = (None, float("inf"))  # type: ignore

    for order, mu in lms_grid:
        out = lms_filter(noise, noisy, order=order, mu=mu)
        val = rms(out)
        if val < best_lms[1]:
            best_lms = ((order, mu), val)

    for order, lam, delta in rls_grid:
        out = rls_filter(noise, noisy, order=order, lam=lam, delta=delta)
        val = rms(out)
        if val < best_rls[1]:
            best_rls = ((order, lam, delta), val)

    return best_lms[0], best_lms[1], best_rls[0], best_rls[1]  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive noise cancellation (LMS/RLS)")
    parser.add_argument("--noisy", default="aud/audio.wav", help="Path to noisy speech")
    parser.add_argument("--noise", default="aud/audio_noise.wav", help="Path to noise ref")
    parser.add_argument("--lms-order", type=int, default=12)
    parser.add_argument("--lms-mu", type=float, default=0.0025)
    parser.add_argument("--rls-order", type=int, default=8)
    parser.add_argument("--rls-lam", type=float, default=0.995)
    parser.add_argument("--rls-delta", type=float, default=0.01)
    parser.add_argument("--skip-plot", action="store_true", help="Disable plot output")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run a small parameter grid and report best RMS",
    )
    args = parser.parse_args()

    base = Path(".")
    out_dir = base / "outputs_basic"
    out_dir.mkdir(exist_ok=True)

    noisy_path = base / args.noisy
    noise_path = base / args.noise
    rate_noisy, noisy = read_wav(noisy_path)
    rate_noise, noise = read_wav(noise_path)
    if rate_noisy != rate_noise:
        raise ValueError("Sample rates must match between noisy and noise files.")

    length = min(len(noisy), len(noise))
    noisy = noisy[:length]
    noise = noise[:length]

    # Preprocess to improve convergence.
    noisy = normalize(remove_dc(noisy))
    noise = normalize(remove_dc(noise))

    if args.grid:
        lms_grid = [
            (16, 0.004),
            (24, 0.006),
            (24, 0.010),
            (32, 0.004),
            (32, 0.006),
            (32, 0.010),
            (48, 0.0035),
            (48, 0.0050),
        ]
        rls_grid = [
            (8, 0.995, 0.01),
            (12, 0.995, 0.01),
            (16, 0.995, 0.01),
            (16, 0.992, 0.01),
            (16, 0.990, 0.01),
            (24, 0.990, 0.01),
            (24, 0.995, 0.020),
            (16, 0.995, 0.005),
        ]
        best_lms, best_lms_rms, best_rls, best_rls_rms = grid_test(
            noisy, noise, lms_grid, rls_grid
        )
        print(f"Grid best LMS (order, mu): {best_lms}, RMS={best_lms_rms:.4f}")
        print(f"Grid best RLS (order, lam, delta): {best_rls}, RMS={best_rls_rms:.4f}")
        # Use best settings for output.
        args.lms_order, args.lms_mu = best_lms
        args.rls_order, args.rls_lam, args.rls_delta = best_rls

    lms_clean, rls_clean = run_once(
        noisy=noisy,
        noise=noise,
        lms_order=args.lms_order,
        lms_mu=args.lms_mu,
        rls_order=args.rls_order,
        rls_lam=args.rls_lam,
        rls_delta=args.rls_delta,
    )

    write_wav(out_dir / "audio_lms.wav", rate_noisy, lms_clean)
    write_wav(out_dir / "audio_rls.wav", rate_noisy, rls_clean)

    if not args.skip_plot:
        plot_signals(rate_noisy, noisy, lms_clean, rls_clean)

    print(f"Input RMS: {rms(noisy):.4f}")
    print(
        f"LMS output RMS (order={args.lms_order}, mu={args.lms_mu}): {rms(lms_clean):.4f}"
    )
    print(
        f"RLS output RMS (order={args.rls_order}, lam={args.rls_lam}, delta={args.rls_delta}): {rms(rls_clean):.4f}"
    )


if __name__ == "__main__":
    main()


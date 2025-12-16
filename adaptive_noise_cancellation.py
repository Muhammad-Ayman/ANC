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
    - outputs_basic/convergence.png    : LMS vs RLS running-RMS convergence
    - outputs_basic/convergence.csv    : numeric convergence curves
"""

from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

try:
    from convergence_diagnostics import (
        check_stability,
        recommend_lms_params,
        recommend_rls_params,
        print_diagnostics,
        plot_enhanced_convergence,
    )
    HAS_DIAGNOSTICS = True
except ImportError:
    HAS_DIAGNOSTICS = False


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
        # Stability check
        if math.isnan(e) or math.isinf(e):
            print(f"WARNING: LMS instability detected at sample {n}")
            return out  # Return partial result
        for i in range(order):
            w[i] += 2 * mu * e * x[i]
        out.append(e)
    return out


def nlms_filter(
    noise: List[float], target: List[float], order: int = 12, mu: float = 0.5, eps: float = 1e-6
) -> List[float]:
    """Normalized LMS adaptive filter with automatic step-size normalization.
    
    NLMS is more robust than LMS as it normalizes the step size by input power,
    preventing divergence when input power varies.
    
    Args:
        noise: Noise reference signal
        target: Noisy target signal
        order: Filter order (number of taps)
        mu: Normalized step size (typically 0.1 to 1.0)
        eps: Small constant to prevent division by zero
    """
    w = [0.0] * order
    out: List[float] = []
    for n in range(len(target)):
        x = [noise[n - k] if n - k >= 0 else 0.0 for k in range(order)]
        y = sum(w[i] * x[i] for i in range(order))
        e = target[n] - y
        # Stability check
        if math.isnan(e) or math.isinf(e):
            print(f"WARNING: NLMS instability detected at sample {n}")
            return out
        # Normalized update: step size divided by input power
        x_power = sum(xi * xi for xi in x) + eps
        alpha = mu / x_power
        for i in range(order):
            w[i] += 2 * alpha * e * x[i]
        out.append(e)
    return out


def rls_filter(
    noise: List[float],
    target: List[float],
    order: int = 12,
    lam: float = 0.95,
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
        # Stability check for RLS
        if abs(denom) < 1e-10:
            print(f"WARNING: RLS numerical instability at sample {n} (denom={denom})")
            return out
        k = [Px[i] / denom for i in range(order)]

        y = sum(w[i] * x[i] for i in range(order))
        e = target[n] - y
        # Stability check
        if math.isnan(e) or math.isinf(e):
            print(f"WARNING: RLS instability detected at sample {n}")
            return out
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


def running_rms(signal: List[float], window: int) -> List[float]:
    """Running RMS with a sliding window (inclusive of current sample)."""
    if window <= 0:
        raise ValueError("window must be positive")
    if not signal:
        return []
    sq = [s * s for s in signal]
    cumsum = []
    total = 0.0
    for v in sq:
        total += v
        cumsum.append(total)
    out: List[float] = []
    for i in range(len(signal)):
        start = i - window + 1
        if start <= 0:
            total_window = cumsum[i]
            count = i + 1
        else:
            total_window = cumsum[i] - cumsum[start - 1]
            count = window
        out.append(math.sqrt(total_window / count))
    return out


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

    segment = min(len(original), rate * 10)  # first 2 seconds
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


def plot_convergence(rate: int, lms_curve, rls_curve) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping convergence plot.")
        return

    length = min(len(lms_curve), len(rls_curve))
    t = [i / rate for i in range(length)]
    plt.figure(figsize=(8, 4))
    plt.semilogy(t, lms_curve[:length], label="LMS", linewidth=0.9)
    plt.semilogy(t, rls_curve[:length], label="RLS", linewidth=0.9)
    plt.xlabel("Time [s]")
    plt.ylabel("Running RMS (linear, log scale)")
    plt.title("Convergence comparison")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("outputs_basic/convergence.png", dpi=150)
    plt.close()
    print("Saved plot to outputs_basic/convergence.png")


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
    parser.add_argument("--rls-order", type=int, default=15)
    parser.add_argument("--rls-lam", type=float, default=0.999)
    parser.add_argument("--rls-delta", type=float, default=0.06)
    parser.add_argument("--skip-plot", action="store_true", help="Disable plot output")
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Run a small parameter grid and report best RMS",
    )
    parser.add_argument(
        "--conv-window-ms",
        type=float,
        default=50.0,
        help="Running RMS window (ms) for convergence curves",
    )
    parser.add_argument(
        "--use-nlms",
        action="store_true",
        help="Use Normalized LMS instead of standard LMS (more stable)",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Show parameter diagnostics and recommendations",
    )
    parser.add_argument(
        "--use-recommended",
        action="store_true",
        help="Use recommended parameters based on signal analysis",
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

    # Show diagnostics if requested or available
    if HAS_DIAGNOSTICS and (args.diagnostics or args.use_recommended):
        print_diagnostics(
            noisy, noise,
            (args.lms_order, args.lms_mu),
            (args.rls_order, args.rls_lam, args.rls_delta)
        )
    
    # Use recommended parameters if requested
    if args.use_recommended and HAS_DIAGNOSTICS:
        rec_lms = recommend_lms_params(noise, noisy, args.lms_order)
        rec_rls = recommend_rls_params(noise, noisy, args.rls_order)
        args.lms_order, args.lms_mu = rec_lms
        args.rls_order, args.rls_lam, args.rls_delta = rec_rls
        print(f"\nUsing recommended parameters:")
        print(f"  LMS: order={args.lms_order}, μ={args.lms_mu:.6f}")
        print(f"  RLS: order={args.rls_order}, λ={args.rls_lam:.6f}, δ={args.rls_delta:.6f}\n")

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

    # Run filters
    if args.use_nlms:
        print("Using Normalized LMS (NLMS) filter...")
        lms_clean = nlms_filter(noise, noisy, order=args.lms_order, mu=args.lms_mu)
    else:
        lms_clean = lms_filter(noise, noisy, order=args.lms_order, mu=args.lms_mu)
    
    rls_clean = rls_filter(noise, noisy, order=args.rls_order, lam=args.rls_lam, delta=args.rls_delta)
    
    # Check stability
    if HAS_DIAGNOSTICS:
        lms_stable, lms_msg = check_stability(lms_clean)
        rls_stable, rls_msg = check_stability(rls_clean)
        if not lms_stable:
            print(f"⚠️  LMS STABILITY ISSUE: {lms_msg}")
        if not rls_stable:
            print(f"⚠️  RLS STABILITY ISSUE: {rls_msg}")

    write_wav(out_dir / "audio_lms.wav", rate_noisy, lms_clean)
    write_wav(out_dir / "audio_rls.wav", rate_noisy, rls_clean)

    # Convergence curves (running RMS).
    window = max(1, int(rate_noisy * (args.conv_window_ms / 1000.0)))
    lms_curve = running_rms(lms_clean, window)
    rls_curve = running_rms(rls_clean, window)
    length = min(len(lms_curve), len(rls_curve))
    times = [i / rate_noisy for i in range(length)]
    conv_csv = out_dir / "convergence.csv"
    with conv_csv.open("w", encoding="utf-8") as f:
        f.write("time_sec,lms_running_rms,rls_running_rms\n")
        for t, a, b in zip(times, lms_curve[:length], rls_curve[:length]):
            f.write(f"{t},{a},{b}\n")
    print(f"Saved convergence data to {conv_csv}")

    if not args.skip_plot:
        plot_signals(rate_noisy, noisy, lms_clean, rls_clean)
        plot_convergence(rate_noisy, lms_curve, rls_curve)
        
        # Generate enhanced plot if diagnostics available
        if HAS_DIAGNOSTICS:
            plot_enhanced_convergence(
                rate_noisy, lms_curve, rls_curve,
                output_path="outputs_basic/convergence_enhanced.png",
                lms_params=(args.lms_order, args.lms_mu),
                rls_params=(args.rls_order, args.rls_lam, args.rls_delta)
            )

    print(f"Input RMS: {rms(noisy):.4f}")
    print(
        f"LMS output RMS (order={args.lms_order}, mu={args.lms_mu}): {rms(lms_clean):.4f}"
    )
    print(
        f"RLS output RMS (order={args.rls_order}, lam={args.rls_lam}, delta={args.rls_delta}): {rms(rls_clean):.4f}"
    )


if __name__ == "__main__":
    main()


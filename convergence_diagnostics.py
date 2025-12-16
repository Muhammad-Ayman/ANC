"""
Convergence diagnostics and parameter optimization for adaptive filters.

This module provides tools to:
- Analyze filter stability and convergence behavior
- Recommend optimal parameters based on signal statistics
- Detect issues like NaN, Inf, and divergence
- Visualize convergence with enhanced plots
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def signal_power(signal: List[float]) -> float:
    """Compute average power of a signal."""
    if not signal:
        return 0.0
    return sum(s * s for s in signal) / len(signal)


def check_stability(signal: List[float]) -> Tuple[bool, str]:
    """
    Check if signal contains NaN, Inf, or shows divergence.
    
    Returns:
        (is_stable, message): True if stable, False otherwise with diagnostic message
    """
    if not signal:
        return True, "Empty signal"
    
    # Check for NaN or Inf
    for i, val in enumerate(signal):
        if math.isnan(val):
            return False, f"NaN detected at sample {i}"
        if math.isinf(val):
            return False, f"Inf detected at sample {i}"
    
    # Check for explosive growth (divergence)
    # If latter half has power > 100x initial power, likely diverging
    n = len(signal)
    if n > 100:
        initial_power = signal_power(signal[:n//4])
        final_power = signal_power(signal[3*n//4:])
        if initial_power > 0 and final_power > 100 * initial_power:
            return False, f"Divergence detected: power grew {final_power/initial_power:.1f}x"
    
    return True, "Signal is stable"


def recommend_lms_params(
    noise: List[float],
    target: List[float],
    order: int = 32
) -> Tuple[int, float]:
    """
    Recommend stable LMS parameters based on signal statistics.
    
    Args:
        noise: Noise reference signal
        target: Target noisy signal
        order: Optional filter order (will use this if provided)
    
    Returns:
        (recommended_order, recommended_mu)
    """
    # Compute signal power
    noise_power = signal_power(noise)
    
    # Recommend order based on signal characteristics
    # For audio, typically 16-64 taps
    if order < 8:
        order = 16
    elif order > 128:
        order = 64
    
    # LMS stability: mu < 2 / (order * trace(R))
    # where R is autocorrelation. Use signal power as approximation
    if noise_power > 0:
        mu_max = 1.5 / (order * noise_power)
        # Use 10-30% of max for safe convergence
        recommended_mu = 0.2 * mu_max
        # Clamp to reasonable range
        recommended_mu = max(0.001, min(0.05, recommended_mu))
    else:
        recommended_mu = 0.01
    
    return order, recommended_mu


def recommend_rls_params(
    noise: List[float],
    target: List[float],
    order: int = 16
) -> Tuple[int, float, float]:
    """
    Recommend stable RLS parameters.
    
    Args:
        noise: Noise reference signal
        target: Target noisy signal
        order: Optional filter order
    
    Returns:
        (recommended_order, recommended_lambda, recommended_delta)
    """
    # RLS typically needs fewer taps than LMS
    if order < 4:
        order = 8
    elif order > 64:
        order = 32
    
    # Lambda (forgetting factor): 0.95 - 0.995
    # Use 0.98 for good balance of adaptation speed and stability
    recommended_lambda = 0.98
    
    # Delta (initial covariance): related to signal power
    noise_power = signal_power(noise)
    if noise_power > 0:
        # Start with moderate initial uncertainty
        recommended_delta = 0.1
    else:
        recommended_delta = 0.01
    
    return order, recommended_lambda, recommended_delta


def analyze_convergence(
    curve: List[float],
    sample_rate: int,
    window_ms: float = 50.0
) -> dict:
    """
    Analyze convergence curve and provide statistics.
    
    Returns:
        Dictionary with convergence statistics
    """
    if not curve:
        return {"valid": False, "message": "Empty curve"}
    
    n = len(curve)
    
    # Check stability
    stable, msg = check_stability(curve)
    
    # Compute statistics
    initial_rms = curve[0] if curve else 0
    final_rms = sum(curve[-min(100, n):]) / min(100, n) if n > 0 else 0
    min_rms = min(curve) if curve else 0
    max_rms = max(curve) if curve else 0
    
    # Check for convergence (final < initial)
    converged = final_rms < initial_rms * 0.9  # 10% improvement threshold
    
    # Estimate convergence time (when curve reaches within 10% of final value)
    convergence_sample = 0
    threshold = final_rms * 1.1
    for i, val in enumerate(curve):
        if val <= threshold:
            convergence_sample = i
            break
    convergence_time_ms = (convergence_sample / sample_rate) * 1000
    
    return {
        "valid": stable,
        "message": msg,
        "initial_rms": initial_rms,
        "final_rms": final_rms,
        "min_rms": min_rms,
        "max_rms": max_rms,
        "improvement_db": 20 * math.log10(final_rms / initial_rms) if initial_rms > 0 else 0,
        "converged": converged,
        "convergence_time_ms": convergence_time_ms,
    }


def plot_enhanced_convergence(
    rate: int,
    lms_curve: List[float],
    rls_curve: List[float],
    output_path: str = "outputs_basic/convergence_enhanced.png",
    lms_params: Optional[Tuple[int, float]] = None,
    rls_params: Optional[Tuple[int, float, float]] = None,
) -> None:
    """
    Create enhanced convergence plot with diagnostics.
    
    Args:
        rate: Sample rate
        lms_curve: LMS running RMS curve
        rls_curve: RLS running RMS curve
        output_path: Where to save the plot
        lms_params: (order, mu) for annotation
        rls_params: (order, lambda, delta) for annotation
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed; skipping enhanced plot.")
        return
    
    length = min(len(lms_curve), len(rls_curve))
    t = [i / rate for i in range(length)]
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Main convergence plot (log scale)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.semilogy(t, lms_curve[:length], label="LMS", linewidth=1.0, alpha=0.8)
    ax1.semilogy(t, rls_curve[:length], label="RLS", linewidth=1.0, alpha=0.8)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Running RMS (log scale)")
    ax1.set_title("Convergence Comparison (Enhanced)")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    
    # Add parameter annotations
    if lms_params:
        ax1.text(0.02, 0.95, f"LMS: order={lms_params[0]}, μ={lms_params[1]:.4f}",
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    if rls_params:
        ax1.text(0.02, 0.85, f"RLS: order={rls_params[0]}, λ={rls_params[1]:.4f}, δ={rls_params[2]:.3f}",
                transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Linear scale for initial convergence
    ax2 = fig.add_subplot(gs[1, 0])
    max_samples = min(length, int(0.5 * rate))  # First 0.5 seconds
    t_zoom = t[:max_samples]
    ax2.plot(t_zoom, lms_curve[:max_samples], label="LMS", linewidth=1.0)
    ax2.plot(t_zoom, rls_curve[:max_samples], label="RLS", linewidth=1.0)
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Running RMS (linear)")
    ax2.set_title("Initial Convergence (First 0.5s)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    # Histogram of RMS values
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(lms_curve[:length], bins=50, alpha=0.5, label="LMS", density=True)
    ax3.hist(rls_curve[:length], bins=50, alpha=0.5, label="RLS", density=True)
    ax3.set_xlabel("RMS Value")
    ax3.set_ylabel("Density")
    ax3.set_title("RMS Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.4)
    
    # Convergence statistics
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    lms_stats = analyze_convergence(lms_curve[:length], rate)
    rls_stats = analyze_convergence(rls_curve[:length], rate)
    
    stats_text = "Convergence Statistics:\n\n"
    stats_text += f"LMS: {lms_stats['message']}\n"
    stats_text += f"  Initial RMS: {lms_stats['initial_rms']:.6f} → Final RMS: {lms_stats['final_rms']:.6f}\n"
    stats_text += f"  Improvement: {lms_stats['improvement_db']:.2f} dB, "
    stats_text += f"Converged: {lms_stats['converged']}, "
    stats_text += f"Time: {lms_stats['convergence_time_ms']:.1f} ms\n\n"
    
    stats_text += f"RLS: {rls_stats['message']}\n"
    stats_text += f"  Initial RMS: {rls_stats['initial_rms']:.6f} → Final RMS: {rls_stats['final_rms']:.6f}\n"
    stats_text += f"  Improvement: {rls_stats['improvement_db']:.2f} dB, "
    stats_text += f"Converged: {rls_stats['converged']}, "
    stats_text += f"Time: {rls_stats['convergence_time_ms']:.1f} ms"
    
    ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved enhanced convergence plot to {output_path}")


def print_diagnostics(
    noisy: List[float],
    noise: List[float],
    current_lms_params: Tuple[int, float],
    current_rls_params: Tuple[int, float, float],
) -> None:
    """Print diagnostic information and parameter recommendations."""
    print("\n" + "="*60)
    print("CONVERGENCE DIAGNOSTICS")
    print("="*60)
    
    # Signal statistics
    noisy_power = signal_power(noisy)
    noise_power = signal_power(noise)
    print(f"\nSignal Statistics:")
    print(f"  Noisy signal power: {noisy_power:.6f}")
    print(f"  Noise reference power: {noise_power:.6f}")
    print(f"  Signal length: {len(noisy)} samples")
    
    # Current parameters
    print(f"\nCurrent Parameters:")
    print(f"  LMS: order={current_lms_params[0]}, μ={current_lms_params[1]:.6f}")
    print(f"  RLS: order={current_rls_params[0]}, λ={current_rls_params[1]:.6f}, δ={current_rls_params[2]:.6f}")
    
    # Recommended parameters
    rec_lms = recommend_lms_params(noise, noisy, current_lms_params[0])
    rec_rls = recommend_rls_params(noise, noisy, current_rls_params[0])
    
    print(f"\nRecommended Parameters:")
    print(f"  LMS: order={rec_lms[0]}, μ={rec_lms[1]:.6f}")
    print(f"  RLS: order={rec_rls[0]}, λ={rec_rls[1]:.6f}, δ={rec_rls[2]:.6f}")
    
    # Parameter comparison
    lms_mu_ratio = current_lms_params[1] / rec_lms[1] if rec_lms[1] > 0 else 0
    if lms_mu_ratio > 2:
        print(f"\n⚠️  WARNING: LMS step size is {lms_mu_ratio:.1f}x larger than recommended!")
        print(f"   This may cause oscillations or divergence.")
    elif lms_mu_ratio < 0.1:
        print(f"\n⚠️  WARNING: LMS step size is {1/lms_mu_ratio:.1f}x smaller than recommended!")
        print(f"   Convergence may be very slow.")
    
    if current_rls_params[1] > 0.998:
        print(f"\n⚠️  WARNING: RLS forgetting factor λ={current_rls_params[1]} is very close to 1.0!")
        print(f"   This can cause numerical instability and divergence.")
    
    print("="*60 + "\n")

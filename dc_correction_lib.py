"""
DC Offset Correction Library
Core functions for DC removal, symmetry correction, and diagnostics.
"""
import numpy as np
from scipy.signal import hilbert
from scipy.optimize import minimize_scalar
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Data Classes for structured results
# =============================================================================

@dataclass
class AsymmetryStats:
    """Statistics for positive/negative amplitude asymmetry."""
    pos_count: int = 0
    neg_count: int = 0
    pos_meanabs: float = 0.0
    neg_meanabs: float = 0.0
    pos_rms: float = 0.0
    neg_rms: float = 0.0
    pos_peak: float = 0.0
    neg_peak: float = 0.0
    ratio_rms: float = 1.0
    ratio_rms_db: float = 0.0
    ratio_peak: float = 1.0
    ratio_peak_db: float = 0.0
    ratio_meanabs: float = 1.0
    ratio_meanabs_db: float = 0.0


@dataclass
class ChannelStats:
    """Statistics for a single channel."""
    mean: float = 0.0
    peak: float = 0.0
    asymmetry: AsymmetryStats = field(default_factory=AsymmetryStats)


@dataclass
class ProcessingResult:
    """Complete result from audio processing."""
    audio: np.ndarray
    sample_rate: Optional[int] = None

    # Processing parameters used
    threshold_db: float = -70.0
    method: str = 'median'
    symmetry: str = 'none'
    symmetry_strength: float = 1.0
    smoothing: float = 0.02
    use_gate: bool = True

    # Per-channel diagnostics
    dc: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    gated_samples: np.ndarray = field(default_factory=lambda: np.array([0]))
    gated_percent: np.ndarray = field(default_factory=lambda: np.array([0.0]))
    total_samples: int = 0

    # Pre/post stats per channel
    pre_stats: list[ChannelStats] = field(default_factory=list)
    post_stats: list[ChannelStats] = field(default_factory=list)

    @property
    def num_channels(self) -> int:
        return len(self.pre_stats)

    @property
    def is_mono(self) -> bool:
        return self.num_channels == 1


# =============================================================================
# Core Processing Functions
# =============================================================================

def rms(x: np.ndarray) -> float:
    """Compute RMS of array."""
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def safe_ratio(a: float, b: float) -> float:
    """Compute ratio a/b safely."""
    if b > 0:
        return a / b
    elif a > 0:
        return float('inf')
    return 1.0


def to_db(x: float) -> float:
    """Convert linear ratio to dB."""
    return 20.0 * np.log10(x) if x > 0 and np.isfinite(x) else float('nan')


def compute_gate_mask(audio: np.ndarray, threshold_db: float, use_gate: bool = True) -> np.ndarray:
    """
    Compute a boolean mask for samples above the gate threshold.

    Args:
        audio: 2D array (samples, channels)
        threshold_db: Gate threshold in dB (e.g., -70)
        use_gate: If False, return all-True mask

    Returns:
        Boolean mask same shape as audio
    """
    if not use_gate:
        return np.ones_like(audio, dtype=bool)
    threshold_lin = 10.0 ** (threshold_db / 20.0)
    return np.abs(audio) >= threshold_lin


def estimate_dc(audio: np.ndarray, mask: np.ndarray, method: str = 'median') -> np.ndarray:
    """
    Estimate DC offset using gated samples.

    Args:
        audio: 2D array (samples, channels)
        mask: Boolean mask for gated samples
        method: 'mean' or 'median'

    Returns:
        DC offset per channel (1D array)
    """
    import warnings
    masked = np.where(mask, audio, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        if method == 'median':
            dc = np.nanmedian(masked, axis=0)
        else:
            dc = np.nanmean(masked, axis=0)
    return np.nan_to_num(dc, nan=0.0).astype('float32')


def compute_asymmetry_stats(audio: np.ndarray, mask: np.ndarray, ch: int = 0) -> AsymmetryStats:
    """
    Compute asymmetry statistics for a single channel.

    Args:
        audio: 2D array (samples, channels)
        mask: Boolean mask
        ch: Channel index

    Returns:
        AsymmetryStats dataclass
    """
    col = audio[:, ch]
    col_mask = mask[:, ch]
    vals = col[col_mask]
    pos = vals[vals > 0.0]
    neg = -vals[vals < 0.0]

    pos_meanabs = float(np.mean(np.abs(pos))) if pos.size else 0.0
    neg_meanabs = float(np.mean(np.abs(neg))) if neg.size else 0.0
    pos_rms_val = rms(pos)
    neg_rms_val = rms(neg)
    pos_peak = float(pos.max()) if pos.size else 0.0
    neg_peak = float(neg.max()) if neg.size else 0.0

    ratio_rms = safe_ratio(pos_rms_val, neg_rms_val)
    ratio_peak = safe_ratio(pos_peak, neg_peak)
    ratio_meanabs = safe_ratio(pos_meanabs, neg_meanabs)

    return AsymmetryStats(
        pos_count=int(pos.size),
        neg_count=int(neg.size),
        pos_meanabs=pos_meanabs,
        neg_meanabs=neg_meanabs,
        pos_rms=pos_rms_val,
        neg_rms=neg_rms_val,
        pos_peak=pos_peak,
        neg_peak=neg_peak,
        ratio_rms=ratio_rms,
        ratio_rms_db=to_db(ratio_rms),
        ratio_peak=ratio_peak,
        ratio_peak_db=to_db(ratio_peak),
        ratio_meanabs=ratio_meanabs,
        ratio_meanabs_db=to_db(ratio_meanabs),
    )


def compute_channel_stats(audio: np.ndarray, mask: np.ndarray, ch: int = 0) -> ChannelStats:
    """Compute full statistics for a single channel."""
    col = audio[:, ch]
    return ChannelStats(
        mean=float(np.mean(col)),
        peak=float(np.max(np.abs(col))) if col.size else 0.0,
        asymmetry=compute_asymmetry_stats(audio, mask, ch),
    )


def compute_adaptive_blend(
    signal: np.ndarray,
    smoothing: float = 0.5,
    sample_rate: int = 48000,
    crossing_sensitivity: float = 6.0,
    min_adaptive_factor: float = 0.25,
) -> np.ndarray:
    """
    Compute per-sample blend factor with adaptive smoothing based on zero-crossing rate.

    At low frequencies (few zero crossings), applies full smoothing for gradual transitions.
    At high frequencies (many zero crossings), reduces smoothing to preserve waveform shape.

    Args:
        signal: 1D audio signal array
        smoothing: Base smoothing factor. Window = smoothing * sample_rate / 1000 samples.
                   At 48kHz: 0.5 -> 24 samples, 1.0 -> 48 samples.
        sample_rate: Audio sample rate in Hz
        crossing_sensitivity: How much zero crossings reduce smoothing (default 6.0).
                              Higher values = more aggressive reduction at high frequencies.
        min_adaptive_factor: Minimum blend factor towards smoothed signal (default 0.25).
                             At high frequencies, blends this much smoothed + rest hard.

    Returns:
        Per-sample blend factor (0.0 to 1.0):
        - 1.0 = apply positive scale factor
        - 0.0 = apply negative scale factor
    """
    from scipy.ndimage import uniform_filter1d

    if smoothing <= 0 or sample_rate <= 0:
        # Hard switching: 1 for positive, 0 for negative/zero
        return (signal > 0).astype(np.float32)

    # Calculate window size from smoothing parameter
    window = max(3, int(smoothing * sample_rate / 1000))
    # Make window odd for symmetry
    if window % 2 == 0:
        window += 1

    # Create polarity signal: +1 for positive, -1 for negative, 0 at zero
    sign_signal = np.sign(signal).astype(np.float32)

    # Detect zero crossings (where sign changes)
    sign_diff = np.abs(np.diff(sign_signal))
    crossings = sign_diff > 0
    crossings = np.concatenate([[False], crossings])  # Pad to original length

    # Compute local crossing density (crossings per sample in window)
    crossing_density = uniform_filter1d(
        crossings.astype(np.float32), size=window, mode='nearest'
    )

    # Adaptive factor: reduce smoothing when crossing rate is high
    # density=0 → factor=1.0 (full smoothing)
    # density=0.5 (Nyquist-like) → factor≈0.25 (mostly hard switching)
    adaptive_factor = 1.0 / (1.0 + crossing_density * crossing_sensitivity)
    adaptive_factor = np.clip(adaptive_factor, min_adaptive_factor, 1.0)

    # Compute smoothed polarity via uniform filter
    smoothed_sign = uniform_filter1d(sign_signal, size=window, mode='nearest')

    # Convert to blend factors (0-1 range)
    hard_blend = (signal > 0).astype(np.float32)
    smooth_blend = (smoothed_sign + 1.0) / 2.0

    # Mix based on adaptive factor: more smoothing at low freq, less at high freq
    blend = adaptive_factor * smooth_blend + (1.0 - adaptive_factor) * hard_blend

    return blend.astype(np.float32)


def apply_symmetry_correction(
    audio: np.ndarray,
    mask: np.ndarray,
    mode: str = 'rms',
    strength: float = 1.0,
    smoothing: float = 0.5,
    sample_rate: int = 48000
) -> np.ndarray:
    """
    Apply symmetry correction to balance positive/negative amplitudes.

    Args:
        audio: 2D array (samples, channels), DC already removed
        mask: Boolean mask for gated samples
        mode: 'rms' or 'peak'
        strength: Correction strength 0..1
        smoothing: Smoothing factor for blending near zero crossings.
                   Window size = smoothing * sample_rate / 1000 samples.
                   At 48kHz: 0.5 -> 24 samples, 1.0 -> 48 samples.
                   Set to 0 for hard switching.
        sample_rate: Audio sample rate

    Returns:
        Corrected audio (same shape)
    """
    if mode == 'none' or strength <= 0:
        return audio.copy()

    arr = audio.copy()
    nch = arr.shape[1]

    for ch in range(nch):
        ch_arr = arr[:, ch]
        ch_mask = mask[:, ch]
        pos_samples = ch_arr[ch_mask & (ch_arr > 0.0)]
        neg_samples = -ch_arr[ch_mask & (ch_arr < 0.0)]

        if mode == 'rms':
            pos_metric = rms(pos_samples)
            neg_metric = rms(neg_samples)
        else:  # peak
            pos_metric = float(pos_samples.max()) if pos_samples.size else 0.0
            neg_metric = float(neg_samples.max()) if neg_samples.size else 0.0

        if pos_metric <= 0 or neg_metric <= 0:
            continue

        scale = pos_metric / neg_metric

        # Compute effective scale factors with strength applied
        eff_scale_neg = 1.0 + strength * (scale - 1.0)
        eff_scale_pos = 1.0 + strength * ((1.0 / scale) - 1.0)

        # Check for clipping; decide which side to adjust
        max_neg = float(neg_samples.max()) if neg_samples.size else 0.0
        max_pos = float(pos_samples.max()) if pos_samples.size else 0.0

        if scale > 1.0:
            if max_neg * eff_scale_neg >= 1.0 - 1e-9:
                scale_pos, scale_neg = eff_scale_pos, 1.0
            else:
                scale_pos, scale_neg = 1.0, eff_scale_neg
        elif scale < 1.0:
            if max_pos * eff_scale_pos >= 1.0 - 1e-9:
                scale_pos, scale_neg = 1.0, eff_scale_neg
            else:
                scale_pos, scale_neg = eff_scale_pos, 1.0
        else:
            scale_pos, scale_neg = 1.0, 1.0

        # Scale smoothing by correction magnitude: small correction = less smoothing needed
        scale_diff = abs(scale_pos - scale_neg)
        effective_smoothing = smoothing * min(1.0, scale_diff * 2.0)  # Full at diff >= 0.5

        # Compute per-sample blend factor with adaptive zero-crossing smoothing
        blend = compute_adaptive_blend(ch_arr, effective_smoothing, sample_rate)

        per_sample_scale = scale_pos * blend + scale_neg * (1.0 - blend)
        arr[:, ch] = ch_arr * per_sample_scale

    return arr


def apply_phase_rotation(
    audio: np.ndarray,
    mask: np.ndarray,
    mode: str = 'rms',
    strength: float = 1.0,
) -> np.ndarray:
    """
    Apply phase rotation to balance positive/negative amplitudes.

    This uses the Hilbert transform to rotate the signal's phase,
    which changes the waveform shape without adding harmonic distortion.

    Args:
        audio: 2D array (samples, channels), DC already removed
        mask: Boolean mask for gated samples (used for measuring asymmetry)
        mode: 'rms' or 'peak' - metric to optimize
        strength: Correction strength 0..1 (interpolate between 0 and optimal angle)

    Returns:
        Phase-rotated audio (same shape)
    """
    if strength <= 0:
        return audio.copy()

    arr = audio.copy()
    nch = arr.shape[1]

    for ch in range(nch):
        ch_arr = arr[:, ch]
        ch_mask = mask[:, ch]

        # Compute analytic signal (Hilbert transform)
        analytic = hilbert(ch_arr)
        hilbert_component = np.imag(analytic)

        def asymmetry_metric(theta: float) -> float:
            """
            Compute asymmetry after rotating by theta radians.
            We want to minimize | metric_pos - metric_neg |.
            """
            rotated = ch_arr * np.cos(theta) - hilbert_component * np.sin(theta)
            gated = rotated[ch_mask]
            pos = gated[gated > 0]
            neg = -gated[gated < 0]

            if pos.size == 0 or neg.size == 0:
                return float('inf')

            if mode == 'rms':
                pos_metric = rms(pos)
                neg_metric = rms(neg)
            else:  # peak
                pos_metric = float(pos.max())
                neg_metric = float(neg.max())

            if neg_metric <= 0:
                return float('inf')

            # Minimize deviation from ratio=1
            ratio = pos_metric / neg_metric
            return abs(ratio - 1.0)

        # Search for optimal angle in range [-pi/2, pi/2]
        # (full rotation would be -pi to pi, but symmetry means we only need half)
        result = minimize_scalar(
            asymmetry_metric,
            bounds=(-np.pi / 2, np.pi / 2),
            method='bounded',
            options={'xatol': 1e-4}
        )

        optimal_theta = result.x if result.success else 0.0

        # Apply strength (interpolate between 0 and optimal angle)
        theta = optimal_theta * strength

        # Apply rotation: y = x*cos(θ) - H(x)*sin(θ)
        arr[:, ch] = ch_arr * np.cos(theta) - hilbert_component * np.sin(theta)

    return arr


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_audio(
    audio: np.ndarray,
    threshold_db: float = -70.0,
    method: str = 'median',
    symmetry: str = 'none',
    symmetry_strength: float = 1.0,
    smoothing: float = 0.02,
    use_gate: bool = True,
    sample_rate: Optional[int] = None,
) -> ProcessingResult:
    """
    Full DC correction pipeline with comprehensive diagnostics.

    Args:
        audio: 1D or 2D audio array
        threshold_db: Gate threshold in dB
        method: DC estimation method ('mean' or 'median')
        symmetry: Symmetry correction mode:
            - 'none': No symmetry correction
            - 'rms': Amplitude scaling to balance RMS (can add harmonic distortion)
            - 'peak': Amplitude scaling to balance peaks (can add harmonic distortion)
            - 'phase': Phase rotation via Hilbert transform (no harmonic distortion)
        symmetry_strength: Strength of symmetry correction (0..1)
        smoothing: Transition width for smooth blending (amplitude modes only)
        use_gate: Whether to use gating
        sample_rate: Optional sample rate for metadata

    Returns:
        ProcessingResult with corrected audio and all diagnostics
    """
    # Ensure 2D
    mono = audio.ndim == 1
    if mono:
        audio = audio[:, None]

    nch = audio.shape[1]
    total_samples = audio.shape[0]

    # Compute mask
    mask = compute_gate_mask(audio, threshold_db, use_gate)

    # Pre-correction stats
    pre_stats = [compute_channel_stats(audio, mask, ch) for ch in range(nch)]

    # Estimate and remove DC
    dc = estimate_dc(audio, mask, method)
    corrected = audio - dc[None, :]

    # Apply symmetry correction
    if symmetry != 'none' and symmetry_strength > 0:
        if symmetry == 'phase':
            # Phase rotation - no harmonic distortion
            corrected = apply_phase_rotation(
                corrected, mask, mode='rms', strength=symmetry_strength
            )
        else:
            # Amplitude scaling (rms/peak modes)
            corrected = apply_symmetry_correction(
                corrected, mask, symmetry, symmetry_strength, smoothing,
                sample_rate=sample_rate or 48000
            )

    # Post-correction stats (recompute asymmetry on corrected audio)
    post_stats = [compute_channel_stats(corrected, mask, ch) for ch in range(nch)]

    # Clip
    corrected = np.clip(corrected, -1.0, 1.0)

    # Update post-stats mean/peak after clipping
    for ch in range(nch):
        col = corrected[:, ch]
        post_stats[ch].mean = float(np.mean(col))
        post_stats[ch].peak = float(np.max(np.abs(col))) if col.size else 0.0

    # Convert back to 1D if input was mono
    output = corrected[:, 0] if mono else corrected

    # Compute gated stats
    gated_samples = mask.sum(axis=0)
    gated_percent = (gated_samples / float(total_samples)) * 100.0

    return ProcessingResult(
        audio=output,
        sample_rate=sample_rate,
        threshold_db=threshold_db,
        method=method,
        symmetry=symmetry,
        symmetry_strength=symmetry_strength,
        smoothing=smoothing,
        use_gate=use_gate,
        dc=dc,
        gated_samples=gated_samples,
        gated_percent=gated_percent,
        total_samples=total_samples,
        pre_stats=pre_stats,
        post_stats=post_stats,
    )


# =============================================================================
# Formatting / Reporting
# =============================================================================

def format_value(v: float, decimals: int = 6) -> str:
    """Format a single float value."""
    return f"{v:.{decimals}f}"


def format_array(a: np.ndarray, decimals: int = 6) -> str:
    """Format array values for printing."""
    if np.ndim(a) == 0:
        return format_value(float(a), decimals)
    return ", ".join(format_value(float(x), decimals) for x in np.atleast_1d(a))


def format_int_array(a: np.ndarray) -> str:
    """Format array values as integers for printing."""
    if np.ndim(a) == 0:
        return str(int(a))
    return ", ".join(str(int(x)) for x in np.atleast_1d(a))


def format_report(result: ProcessingResult, path: str = "") -> str:
    """
    Generate a human-readable report from processing result.

    Args:
        result: ProcessingResult from process_audio
        path: Optional file path to include in report

    Returns:
        Formatted string report
    """
    lines = []

    if path:
        lines.append(f"File: {path}")
    if result.sample_rate:
        lines.append(f" SR: {int(result.sample_rate)}")

    # Pre/post summary
    pre_means = [s.mean for s in result.pre_stats]
    pre_peaks = [s.peak for s in result.pre_stats]
    post_means = [s.mean for s in result.post_stats]
    post_peaks = [s.peak for s in result.post_stats]

    lines.append(f" Pre-mean: {format_array(np.array(pre_means))}  Pre-peak: {format_array(np.array(pre_peaks))}")
    lines.append(f" Gated-samples: {format_int_array(result.gated_samples)} ({format_array(result.gated_percent)}%)  DC used: {format_array(result.dc)}")
    lines.append(f" Post-mean: {format_array(np.array(post_means))}  Post-peak: {format_array(np.array(post_peaks))}")

    gate_str = 'on' if result.use_gate else 'off'
    lines.append(f" DC method: {result.method}  Symmetry correction: {result.symmetry}  Gate: {gate_str} ({result.threshold_db} dB)")

    # Per-channel asymmetry
    for i, (post, pre) in enumerate(zip(result.post_stats, result.pre_stats)):
        ch_label = "mono" if result.is_mono else f"ch{i}"
        post_a, pre_a = post.asymmetry, pre.asymmetry

        lines.append(f" Asymmetry {ch_label}: pos_count={int(post_a.pos_count)} neg_count={int(post_a.neg_count)} (previously pos={int(pre_a.pos_count)} neg={int(pre_a.neg_count)})")
        lines.append(f"  mean-abs pos/neg: {format_value(post_a.pos_meanabs)} / {format_value(post_a.neg_meanabs)}  ratio={format_value(post_a.ratio_meanabs)} ({post_a.ratio_meanabs_db:+.2f} dB) (previously {pre_a.ratio_meanabs_db:+.2f} dB)")
        lines.append(f"  rms      pos/neg: {format_value(post_a.pos_rms)} / {format_value(post_a.neg_rms)}  ratio={format_value(post_a.ratio_rms)} ({post_a.ratio_rms_db:+.2f} dB) (previously {pre_a.ratio_rms_db:+.2f} dB)")
        lines.append(f"  peak     pos/neg: {format_value(post_a.pos_peak)} / {format_value(post_a.neg_peak)}  ratio={format_value(post_a.ratio_peak)} ({post_a.ratio_peak_db:+.2f} dB) (previously {pre_a.ratio_peak_db:+.2f} dB)")

    return "\n".join(lines)

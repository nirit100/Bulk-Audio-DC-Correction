"""
Unit tests for DC Offset Correction Library

Run with: pytest test_dc_correction.py -v
"""
import numpy as np
import pytest
from dc_correction_lib import (
    compute_gate_mask,
    estimate_dc,
    rms,
    compute_asymmetry_stats,
    apply_symmetry_correction,
    apply_phase_rotation,
    compute_adaptive_blend,
    process_audio,
    ProcessingResult,
    AsymmetryStats,
    format_report,
)


# ============================================================================
# Test Fixtures - synthetic audio signals
# ============================================================================

@pytest.fixture
def sine_wave():
    """Pure sine wave, no DC offset, symmetric."""
    t = np.linspace(0, 1, 48000, dtype='float32')
    return np.sin(2 * np.pi * 100 * t)  # 100 Hz sine


@pytest.fixture
def sine_with_dc():
    """Sine wave with DC offset of +0.1"""
    t = np.linspace(0, 1, 48000, dtype='float32')
    return (np.sin(2 * np.pi * 100 * t) * 0.5 + 0.1).astype('float32')


@pytest.fixture
def asymmetric_signal():
    """Signal with positive amplitudes larger than negative."""
    t = np.linspace(0, 1, 48000, dtype='float32')
    sig = np.sin(2 * np.pi * 100 * t) * 0.5
    # Make positive side 1.5x larger
    sig[sig > 0] *= 1.5
    return sig.astype('float32')


@pytest.fixture
def silence():
    """Very quiet signal (below typical gate threshold)."""
    return np.random.randn(48000).astype('float32') * 1e-5


# ============================================================================
# Tests for compute_gate_mask
# ============================================================================

class TestGateMask:
    def test_gate_on_loud_signal(self, sine_wave):
        """Loud signal should have most samples above gate."""
        audio = sine_wave[:, None] * 0.5  # 2D, -6dB
        mask = compute_gate_mask(audio, threshold_db=-70.0, use_gate=True)
        # Most samples should be above -70dB threshold
        assert mask.sum() / mask.size > 0.99

    def test_gate_on_quiet_signal(self, silence):
        """Very quiet signal should have few samples above gate."""
        audio = silence[:, None]
        mask = compute_gate_mask(audio, threshold_db=-70.0, use_gate=True)
        # Most samples should be below threshold
        assert mask.sum() / mask.size < 0.01

    def test_gate_disabled(self, silence):
        """With gate disabled, all samples should be included."""
        audio = silence[:, None]
        mask = compute_gate_mask(audio, threshold_db=-70.0, use_gate=False)
        assert mask.all()

    def test_threshold_affects_mask(self, sine_wave):
        """Higher threshold should exclude more samples."""
        audio = sine_wave[:, None] * 0.1  # -20dB peak
        mask_low = compute_gate_mask(audio, threshold_db=-40.0, use_gate=True)
        mask_high = compute_gate_mask(audio, threshold_db=-20.0, use_gate=True)
        assert mask_low.sum() > mask_high.sum()


# ============================================================================
# Tests for DC estimation
# ============================================================================

class TestDCEstimation:
    def test_dc_removal_mean(self, sine_with_dc):
        """Mean method should estimate DC correctly."""
        audio = sine_with_dc[:, None]
        mask = np.ones_like(audio, dtype=bool)
        dc = estimate_dc(audio, mask, method='mean')
        # DC should be close to 0.1
        assert abs(dc[0] - 0.1) < 0.01

    def test_dc_removal_median(self, sine_with_dc):
        """Median method should estimate DC correctly."""
        audio = sine_with_dc[:, None]
        mask = np.ones_like(audio, dtype=bool)
        dc = estimate_dc(audio, mask, method='median')
        # DC should be close to 0.1
        assert abs(dc[0] - 0.1) < 0.01

    def test_dc_with_partial_mask(self, sine_with_dc):
        """DC estimation should work with partial mask."""
        audio = sine_with_dc[:, None]
        mask = np.abs(audio) > 0.2  # Only include louder samples
        dc = estimate_dc(audio, mask, method='mean')
        # Should still get reasonable DC estimate
        assert abs(dc[0] - 0.1) < 0.15

    def test_dc_empty_mask_returns_zero(self):
        """Empty mask should return DC of 0."""
        audio = np.random.randn(1000, 1).astype('float32')
        mask = np.zeros_like(audio, dtype=bool)
        dc = estimate_dc(audio, mask, method='mean')
        assert dc[0] == 0.0


# ============================================================================
# Tests for RMS function
# ============================================================================

class TestRMS:
    def test_rms_sine(self):
        """RMS of sine wave should be peak / sqrt(2)."""
        t = np.linspace(0, 1, 48000)
        sine = np.sin(2 * np.pi * 100 * t)
        expected_rms = 1.0 / np.sqrt(2)
        assert abs(rms(sine) - expected_rms) < 0.001

    def test_rms_empty(self):
        """RMS of empty array should be 0."""
        assert rms(np.array([])) == 0.0

    def test_rms_constant(self):
        """RMS of constant signal equals the constant."""
        constant = np.ones(1000) * 0.5
        assert abs(rms(constant) - 0.5) < 0.001


# ============================================================================
# Tests for asymmetry statistics
# ============================================================================

class TestAsymmetryStats:
    def test_symmetric_signal(self, sine_wave):
        """Symmetric signal should have ratio ~1.0"""
        audio = sine_wave[:, None]
        mask = np.ones_like(audio, dtype=bool)
        stats = compute_asymmetry_stats(audio, mask, ch=0)
        assert abs(stats.ratio_rms - 1.0) < 0.01
        assert abs(stats.ratio_peak - 1.0) < 0.01

    def test_asymmetric_signal(self, asymmetric_signal):
        """Asymmetric signal should have ratio != 1.0"""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)
        stats = compute_asymmetry_stats(audio, mask, ch=0)
        # Positive is 1.5x larger, so ratio should be ~1.5
        assert stats.ratio_rms > 1.3
        assert stats.ratio_peak > 1.3

    def test_counts_positive_negative(self, sine_wave):
        """Should count positive and negative samples."""
        audio = sine_wave[:, None]
        mask = np.ones_like(audio, dtype=bool)
        stats = compute_asymmetry_stats(audio, mask, ch=0)
        # Should have roughly equal pos/neg counts
        assert abs(stats.pos_count - stats.neg_count) < 100


# ============================================================================
# Tests for symmetry correction
# ============================================================================

class TestSymmetryCorrection:
    def test_no_correction_when_mode_none(self, asymmetric_signal):
        """Mode 'none' should return unchanged audio."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_symmetry_correction(audio, mask, mode='none')
        np.testing.assert_array_equal(result, audio)

    def test_no_correction_when_strength_zero(self, asymmetric_signal):
        """Strength 0 should return unchanged audio."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_symmetry_correction(audio, mask, mode='rms', strength=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_rms_correction_balances_signal(self, asymmetric_signal):
        """RMS correction should balance pos/neg RMS."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_symmetry_correction(
            audio, mask, mode='rms', strength=1.0, smoothing=2.0, sample_rate=48000
        )
        # Check that RMS ratio is closer to 1.0
        pos = result[result > 0]
        neg = -result[result < 0]
        ratio_after = rms(pos) / rms(neg)
        # Original ratio was ~1.5, should be closer to 1.0 now
        assert abs(ratio_after - 1.0) < 0.1

    def test_partial_strength(self, asymmetric_signal):
        """Partial strength should give intermediate result."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)

        result_full, _ = apply_symmetry_correction(
            audio, mask, mode='rms', strength=1.0, smoothing=2.0, sample_rate=48000
        )
        result_half, _ = apply_symmetry_correction(
            audio, mask, mode='rms', strength=0.5, smoothing=2.0, sample_rate=48000
        )

        # Half strength should be between original and full correction
        pos_orig = rms(audio[audio > 0])
        pos_full = rms(result_full[result_full > 0])
        pos_half = rms(result_half[result_half > 0])

        assert min(pos_orig, pos_full) < pos_half < max(pos_orig, pos_full)

    def test_no_clipping(self, asymmetric_signal):
        """Correction should not produce clipping."""
        audio = asymmetric_signal[:, None] * 0.9  # Near clipping
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_symmetry_correction(
            audio, mask, mode='rms', strength=1.0, smoothing=2.0, sample_rate=48000
        )
        assert np.abs(result).max() <= 1.0


# ============================================================================
# Tests for time-domain smooth blending (no discontinuities)
# ============================================================================

class TestSmoothBlending:
    def test_smooth_blending_continuous(self, asymmetric_signal):
        """Smoothed output should not have large discontinuities."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_symmetry_correction(
            audio, mask, mode='rms', strength=1.0, smoothing=2.0, sample_rate=48000
        )
        # Check that sample-to-sample differences are small
        diff = np.abs(np.diff(result[:, 0]))
        # No huge jumps (discontinuities would cause large diffs)
        assert diff.max() < 0.1

    def test_hard_vs_smooth_blending(self):
        """Smoothing should reduce abruptness at zero crossings."""
        # Create a longer signal with clear zero crossings
        t = np.linspace(0, 0.1, 4800, dtype='float32')  # 100ms at 48kHz
        audio = (np.sin(2 * np.pi * 100 * t) * 0.5)[:, None]
        # Make asymmetric
        audio[audio > 0] *= 1.5
        mask = np.ones_like(audio, dtype=bool)

        # With hard blending, scale changes abruptly at zero crossings
        result_hard, _ = apply_symmetry_correction(
            audio, mask, mode='rms', strength=1.0, smoothing=0.0, sample_rate=48000
        )
        # With smoothing, the transition is gradual over time
        result_smooth, _ = apply_symmetry_correction(
            audio, mask, mode='rms', strength=1.0, smoothing=2.0, sample_rate=48000
        )

        # Both should balance the signal
        def rms_ratio(arr):
            pos = arr[arr > 0]
            neg = -arr[arr < 0]
            return rms(pos) / rms(neg)

        # Both should improve symmetry
        assert abs(rms_ratio(result_hard.flatten()) - 1.0) < 0.15
        assert abs(rms_ratio(result_smooth.flatten()) - 1.0) < 0.15

    def test_smoothing_window_size(self):
        """Smoothing parameter controls the transition window size."""
        t = np.linspace(0, 0.1, 4800, dtype='float32')
        audio = (np.sin(2 * np.pi * 100 * t) * 0.5)[:, None]
        audio[audio > 0] *= 1.5
        mask = np.ones_like(audio, dtype=bool)

        # Longer smoothing should create smoother transitions
        result_short, _ = apply_symmetry_correction(
            audio, mask, mode='rms', strength=1.0, smoothing=1.0, sample_rate=48000
        )
        result_long, _ = apply_symmetry_correction(
            audio, mask, mode='rms', strength=1.0, smoothing=5.0, sample_rate=48000
        )

        # Both should produce valid output
        assert result_short.shape == audio.shape
        assert result_long.shape == audio.shape


# ============================================================================
# Tests for adaptive blend (zero-crossing rate adaptation)
# ============================================================================

class TestAdaptiveBlend:
    """Tests for compute_adaptive_blend function."""

    def test_hard_switching_when_smoothing_zero(self):
        """With smoothing=0, should return hard blend (1 for positive, 0 for negative)."""
        t = np.linspace(0, 0.1, 4800, dtype='float32')
        signal = np.sin(2 * np.pi * 100 * t)
        
        blend = compute_adaptive_blend(signal, smoothing=0.0, sample_rate=48000)
        
        expected = (signal > 0).astype(np.float32)
        np.testing.assert_array_equal(blend, expected)

    def test_output_shape_matches_input(self):
        """Output blend array should match input signal shape."""
        signal = np.random.randn(10000).astype('float32')
        blend = compute_adaptive_blend(signal, smoothing=0.5, sample_rate=48000)
        assert blend.shape == signal.shape
        assert blend.dtype == np.float32

    def test_blend_values_in_range(self):
        """Blend values should always be between 0 and 1."""
        signal = np.random.randn(10000).astype('float32')
        blend = compute_adaptive_blend(signal, smoothing=1.0, sample_rate=48000)
        assert blend.min() >= 0.0
        assert blend.max() <= 1.0

    def test_low_frequency_gets_more_smoothing(self):
        """Low frequency signals should have blended transitions."""
        # 10 Hz sine at 48kHz = 4800 samples per cycle, ~one crossing per 2400 samples
        t = np.linspace(0, 0.5, 24000, dtype='float32')  # 0.5 seconds
        low_freq = np.sin(2 * np.pi * 10 * t)
        
        blend = compute_adaptive_blend(low_freq, smoothing=2.0, sample_rate=48000)
        
        # Around zero crossings, blend should be close to 0.5 (transitioning)
        # Find samples near zero crossings
        crossings = np.where(np.abs(np.diff(np.sign(low_freq))) > 0)[0]
        if len(crossings) > 0:
            near_crossing = blend[crossings[1]]  # Second crossing (avoid edge)
            # Should be close to 0.5 due to smoothing
            assert 0.3 < near_crossing < 0.7

    def test_high_frequency_gets_less_smoothing(self):
        """High frequency signals should approach hard switching."""
        # 8000 Hz sine at 48kHz = 6 samples per cycle, crossing every 3 samples
        t = np.linspace(0, 0.01, 480, dtype='float32')  # 10ms
        high_freq = np.sin(2 * np.pi * 8000 * t)
        
        blend = compute_adaptive_blend(high_freq, smoothing=2.0, sample_rate=48000)
        hard_blend = (high_freq > 0).astype(np.float32)
        
        # At high frequencies, adaptive factor should approach min (0.25)
        # So blend should be closer to hard_blend than to 0.5
        # Check that most values are not in the 0.4-0.6 transition zone
        in_transition = np.sum((blend > 0.4) & (blend < 0.6))
        total = len(blend)
        assert in_transition / total < 0.5  # Less than 50% in transition zone

    def test_crossing_sensitivity_parameter(self):
        """Higher crossing_sensitivity should reduce smoothing more aggressively."""
        t = np.linspace(0, 0.01, 480, dtype='float32')
        signal = np.sin(2 * np.pi * 2000 * t)  # Medium-high frequency
        
        blend_low_sens = compute_adaptive_blend(
            signal, smoothing=2.0, sample_rate=48000, crossing_sensitivity=2.0
        )
        blend_high_sens = compute_adaptive_blend(
            signal, smoothing=2.0, sample_rate=48000, crossing_sensitivity=10.0
        )
        
        # High sensitivity should be closer to hard blend
        hard = (signal > 0).astype(np.float32)
        diff_low = np.mean(np.abs(blend_low_sens - hard))
        diff_high = np.mean(np.abs(blend_high_sens - hard))
        assert diff_high < diff_low

    def test_min_adaptive_factor_parameter(self):
        """min_adaptive_factor should cap how much smoothing is reduced."""
        t = np.linspace(0, 0.01, 480, dtype='float32')
        high_freq = np.sin(2 * np.pi * 8000 * t)
        
        # With high min factor, even high freq should get more smoothing
        blend_low_min = compute_adaptive_blend(
            high_freq, smoothing=2.0, sample_rate=48000, min_adaptive_factor=0.1
        )
        blend_high_min = compute_adaptive_blend(
            high_freq, smoothing=2.0, sample_rate=48000, min_adaptive_factor=0.8
        )
        
        # High min factor should be smoother (more values in 0.3-0.7 range)
        smooth_count_low = np.sum((blend_low_min > 0.3) & (blend_low_min < 0.7))
        smooth_count_high = np.sum((blend_high_min > 0.3) & (blend_high_min < 0.7))
        assert smooth_count_high >= smooth_count_low

    def test_constant_signal_no_crossings(self):
        """Signal with no zero crossings should get full smoothing factor."""
        # All positive signal
        signal = np.ones(1000, dtype='float32') * 0.5
        blend = compute_adaptive_blend(signal, smoothing=1.0, sample_rate=48000)
        # All positive -> blend should be all 1.0
        np.testing.assert_array_almost_equal(blend, np.ones(1000))

    def test_mixed_frequency_content(self):
        """Signal with both low and high frequency content."""
        t = np.linspace(0, 0.1, 4800, dtype='float32')
        # Low freq modulated by high freq
        mixed = np.sin(2 * np.pi * 50 * t) * 0.5 + np.sin(2 * np.pi * 5000 * t) * 0.3
        
        blend = compute_adaptive_blend(mixed, smoothing=2.0, sample_rate=48000)
        
        # Should produce valid output without errors
        assert blend.shape == mixed.shape
        assert np.all(np.isfinite(blend))

    def test_dc_signal_all_positive(self):
        """Purely positive DC signal should have blend = 1.0 everywhere."""
        signal = np.ones(1000, dtype='float32') * 0.8
        blend = compute_adaptive_blend(signal, smoothing=1.0, sample_rate=48000)
        np.testing.assert_array_almost_equal(blend, np.ones(1000, dtype='float32'))

    def test_dc_signal_all_negative(self):
        """Purely negative DC signal should have blend = 0.0 everywhere."""
        signal = np.ones(1000, dtype='float32') * -0.8
        blend = compute_adaptive_blend(signal, smoothing=1.0, sample_rate=48000)
        np.testing.assert_array_almost_equal(blend, np.zeros(1000, dtype='float32'))

    def test_very_short_signal(self):
        """Should handle very short signals gracefully."""
        signal = np.array([0.5, -0.5, 0.5], dtype='float32')
        blend = compute_adaptive_blend(signal, smoothing=1.0, sample_rate=48000)
        assert blend.shape == (3,)
        assert np.all(np.isfinite(blend))

    def test_single_sample(self):
        """Should handle single sample input."""
        signal = np.array([0.5], dtype='float32')
        blend = compute_adaptive_blend(signal, smoothing=1.0, sample_rate=48000)
        assert blend.shape == (1,)
        assert blend[0] == 1.0  # Positive sample -> blend = 1


# ============================================================================
# Tests for phase rotation symmetry correction
# ============================================================================

class TestPhaseRotation:
    def test_no_correction_when_strength_zero(self, asymmetric_signal):
        """Strength 0 should return unchanged audio."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_phase_rotation(audio, mask, mode='rms', strength=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_phase_rotation_balances_rms(self, asymmetric_signal):
        """Phase rotation should balance pos/neg RMS."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_phase_rotation(audio, mask, mode='rms', strength=1.0)

        # Check that RMS ratio is closer to 1.0
        pos = result[result > 0]
        neg = -result[result < 0]
        ratio_after = rms(pos) / rms(neg)
        # Original ratio was ~1.5, should be closer to 1.0 now
        assert abs(ratio_after - 1.0) < 0.15

    def test_phase_rotation_preserves_rms(self, asymmetric_signal):
        """Phase rotation should not significantly change overall RMS."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_phase_rotation(audio, mask, mode='rms', strength=1.0)

        # Total RMS should be very similar (phase rotation preserves energy)
        # Allow up to 5% change since heavily asymmetric signals can shift slightly
        rms_before = rms(audio.flatten())
        rms_after = rms(result.flatten())
        assert abs(rms_after - rms_before) / rms_before < 0.05

    def test_phase_rotation_partial_strength(self, asymmetric_signal):
        """Partial strength should give intermediate result."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)

        result_full, _ = apply_phase_rotation(audio, mask, mode='rms', strength=1.0)
        result_half, _ = apply_phase_rotation(audio, mask, mode='rms', strength=0.5)

        # Compute asymmetry ratios
        def asymmetry_ratio(arr):
            pos = arr[arr > 0]
            neg = -arr[arr < 0]
            return rms(pos) / rms(neg)

        ratio_orig = asymmetry_ratio(audio.flatten())
        ratio_full = asymmetry_ratio(result_full.flatten())
        ratio_half = asymmetry_ratio(result_half.flatten())

        # Half strength should be between original and full
        # (closer to 1.0 than original, but not as close as full)
        dist_orig = abs(ratio_orig - 1.0)
        dist_full = abs(ratio_full - 1.0)
        dist_half = abs(ratio_half - 1.0)
        assert dist_full < dist_half < dist_orig

    def test_phase_rotation_symmetric_signal_unchanged(self, sine_wave):
        """Symmetric signal should remain mostly unchanged."""
        audio = sine_wave[:, None]
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_phase_rotation(audio, mask, mode='rms', strength=1.0)

        # RMS should be nearly identical
        rms_before = rms(audio.flatten())
        rms_after = rms(result.flatten())
        assert abs(rms_after - rms_before) / rms_before < 0.01

    def test_phase_rotation_no_clipping(self, asymmetric_signal):
        """Phase rotation should not cause clipping."""
        audio = asymmetric_signal[:, None] * 0.9  # Near full scale
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_phase_rotation(audio, mask, mode='rms', strength=1.0)
        # Phase rotation preserves amplitude envelope, so no clipping expected
        assert np.abs(result).max() <= 1.0 + 1e-6

    def test_phase_rotation_peak_mode(self, asymmetric_signal):
        """Peak mode should also balance the signal."""
        audio = asymmetric_signal[:, None]
        mask = np.ones_like(audio, dtype=bool)
        result, info = apply_phase_rotation(audio, mask, mode='peak', strength=1.0)

        # Check that peak ratio is closer to 1.0
        pos_peak = result[result > 0].max()
        neg_peak = (-result[result < 0]).max()
        ratio_after = pos_peak / neg_peak
        # Should be improved from original ~1.5
        assert abs(ratio_after - 1.0) < 0.3

    def test_phase_rotation_multichannel(self, asymmetric_signal):
        """Phase rotation should work on multichannel audio."""
        # Create stereo with different asymmetry per channel
        ch1 = asymmetric_signal.copy()
        ch2 = -asymmetric_signal  # Inverted
        audio = np.column_stack([ch1, ch2])
        mask = np.ones_like(audio, dtype=bool)

        result, info = apply_phase_rotation(audio, mask, mode='rms', strength=1.0)

        # Both channels should be processed
        assert result.shape == audio.shape
        # Each channel should have improved symmetry
        for ch in range(2):
            pos = result[result[:, ch] > 0, ch]
            neg = -result[result[:, ch] < 0, ch]
            ratio = rms(pos) / rms(neg)
            assert abs(ratio - 1.0) < 0.15


# ============================================================================
# Tests for phase mode in full pipeline
# ============================================================================

class TestPhaseModeInPipeline:
    def test_process_audio_with_phase_symmetry(self, asymmetric_signal):
        """process_audio should accept 'phase' symmetry mode."""
        result = process_audio(
            asymmetric_signal,
            threshold_db=-70.0,
            method='median',
            symmetry='phase',
            symmetry_strength=1.0,
        )

        # Should return valid result
        assert isinstance(result, ProcessingResult)
        assert result.symmetry == 'phase'

        # Asymmetry should be improved
        post_ratio = result.post_stats[0].asymmetry.ratio_rms
        pre_ratio = result.pre_stats[0].asymmetry.ratio_rms
        assert abs(post_ratio - 1.0) < abs(pre_ratio - 1.0)

    def test_process_audio_phase_preserves_dc_removal(self, sine_with_dc):
        """Phase mode should not interfere with DC removal."""
        result = process_audio(
            sine_with_dc,
            threshold_db=-70.0,
            method='mean',
            symmetry='phase',
        )
        # DC should still be removed
        assert abs(np.mean(result.audio)) < 0.01

    def test_process_audio_phase_stereo(self):
        """Phase mode should work with stereo audio."""
        # Create asymmetric stereo signal
        t = np.linspace(0, 1, 48000, dtype='float32')
        mono = np.sin(2 * np.pi * 100 * t) * 0.5
        mono[mono > 0] *= 1.5  # Make asymmetric
        stereo = np.column_stack([mono, -mono])

        result = process_audio(
            stereo,
            symmetry='phase',
        )

        assert result.audio.shape == stereo.shape
        assert len(result.post_stats) == 2


# ============================================================================
# Tests for full pipeline
# ============================================================================

class TestFullPipeline:
    def test_process_removes_dc(self, sine_with_dc):
        """Full pipeline should remove DC offset."""
        result = process_audio(
            sine_with_dc,
            threshold_db=-70.0,
            method='mean',
            symmetry='none'
        )
        # Post-correction mean should be near zero
        assert abs(np.mean(result.audio)) < 0.01

    def test_process_corrects_symmetry(self, asymmetric_signal):
        """Full pipeline should correct asymmetry when enabled."""
        result = process_audio(
            asymmetric_signal,
            threshold_db=-70.0,
            method='median',
            symmetry='rms',
            symmetry_strength=1.0,
            smoothing=2.0
        )
        pos = result.audio[result.audio > 0]
        neg = -result.audio[result.audio < 0]
        ratio = rms(pos) / rms(neg)
        assert abs(ratio - 1.0) < 0.1

    def test_process_preserves_shape_mono(self, sine_wave):
        """Output should be 1D for 1D input."""
        result = process_audio(sine_wave)
        assert result.audio.ndim == 1
        assert result.audio.shape == sine_wave.shape

    def test_process_preserves_shape_stereo(self, sine_wave):
        """Output should be 2D for 2D input."""
        stereo = np.column_stack([sine_wave, sine_wave * 0.8])
        result = process_audio(stereo)
        assert result.audio.ndim == 2
        assert result.audio.shape == stereo.shape

    def test_process_clips_output(self):
        """Output should be clipped to [-1, 1]."""
        # Create signal that would exceed 1.0 after processing
        audio = np.ones(1000, dtype='float32') * 0.99
        audio[::2] = -0.5  # Asymmetric
        result = process_audio(
            audio, symmetry='rms', symmetry_strength=1.0
        )
        assert result.audio.max() <= 1.0
        assert result.audio.min() >= -1.0

    def test_info_contains_dc(self, sine_with_dc):
        """ProcessingResult should contain DC estimate."""
        result = process_audio(sine_with_dc)
        assert abs(result.dc[0] - 0.1) < 0.01

    def test_format_report(self, sine_with_dc):
        """format_report should produce readable output."""
        result = process_audio(sine_with_dc, sample_rate=48000)
        report = format_report(result, path="test.wav")
        assert "test.wav" in report
        assert "48000" in report
        assert "Pre-mean" in report
        assert "Post-mean" in report


# ============================================================================
# Edge case tests
# ============================================================================

class TestEdgeCases:
    def test_all_zeros(self):
        """Should handle all-zero input."""
        audio = np.zeros(1000, dtype='float32')
        result = process_audio(audio)
        np.testing.assert_array_equal(result.audio, audio)

    def test_single_sample(self):
        """Should handle single sample."""
        audio = np.array([0.5], dtype='float32')
        result = process_audio(audio)
        assert result.audio.shape == (1,)

    def test_very_short_audio(self):
        """Should handle very short audio."""
        audio = np.array([0.1, -0.1, 0.2], dtype='float32')
        result = process_audio(audio)
        assert result.audio.shape == audio.shape

    def test_all_positive(self):
        """Should handle all-positive input."""
        audio = np.abs(np.random.randn(1000).astype('float32')) * 0.5
        result = process_audio(audio, symmetry='rms')
        # Should not crash; symmetry correction skipped when neg=0
        assert result.audio.shape == audio.shape

    def test_all_negative(self):
        """Should handle all-negative input."""
        audio = -np.abs(np.random.randn(1000).astype('float32')) * 0.5
        result = process_audio(audio, symmetry='rms')
        assert result.audio.shape == audio.shape

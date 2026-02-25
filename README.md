# DC Offset & Asymmetry Correction

A small Python tool for correcting **DC offset** and **amplitude asymmetry** in audio files.

## The Problem

### DC Offset (Additive Bias)

When the audio waveform isn't centered around zero. This can be caused by:
- Faulty recording equipment or ADC converters with bias voltage errors
- Analog hardware with calibration drift or aging capacitors
- Some synthesizers and audio interfaces
- Improper gain staging in the signal chain

**Consequences:** Reduced headroom, clicks/pops at edit points, asymmetric clipping behavior, issues with some dynamics plugins.

Top: Negative DC offset, Bottom: Same audio after running `python .\dc_offset_correction.py file.wav`
<img width="1024" height="348" alt="image" src="https://github.com/user-attachments/assets/57843f8e-ad60-40d2-9661-8023c43ad98f" />

As long as no clipping is involved, this correction is pretty easy to do and more or less perfect.

### Amplitude Asymmetry (Multiplicative Bias)

When positive amplitudes are consistently larger (or smaller) than negative ones. Common causes:
- **Magnetic pickups** (bass/guitar): The pickup responds differently to string motion toward vs away from the magnet
- **Dynamic microphones**: Voice coil and diaphragm mechanics can have asymmetric excursion characteristics
- **Tube/valve equipment**: Operating point and saturation behavior can favor one polarity
- **Some ribbon microphones**: Asymmetric ribbon tension or magnetic field

**Consequences:** Wasted headroom (peak meters show higher than perceived loudness), potential for asymmetric distortion when pushed, and your mastering engineer will have a bad day.

Top: Bass signal with asymmetric amplitude, Bottom: Same audio after running `python .\dc_offset_correction.py --symmetry phase file.wav`:
<img width="945" height="297" alt="image" src="https://github.com/user-attachments/assets/5ec7aa49-42c9-4e4d-b9e1-6e96d5917a2c" />

## Features

- **DC offset removal** using mean or median estimation
- Optional **phase rotation symmetry correction** — uses Hilbert transform to rotate waveform phase, balancing positive/negative amplitudes without adding harmonic distortion
- Alternative **amplitude scaling** symmetry modes (RMS/peak-based) — simpler but can introduce subtle distortion
- **Gated analysis** — ignores quiet sections for more accurate estimation, parameterized threshold
- **Dry-run mode** — preview diagnostics without writing files
- **Preserves audio format** — maintains sample rate, bit depth, and file format

## Installation & Usage

First, clone the repository.

Then install dependencies:

```bash
pip install -r requirements.txt
```

To run the program:

```bash
python dc_offset_correction.py [options] <files...>
```

I will probably do a complete packaging later. Maybe. Who knows.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | off | Print stats but do not actually write files |
| `--method {mean,median}` | `median` | DC offset estimation method (median is more robust to outliers) |
| `--symmetry {none,phase,rms,peak}` | `none` | Symmetry correction: `none` (disabled), `phase` (Hilbert transform, no distortion), `rms`/`peak` (amplitude scaling, can distort) |
| `--symmetry-strength` | `1.0` | Strength of symmetry correction (0..1) |
| `--smoothing` | `0.02` | Amplitude range for blending near zero (rms/peak modes only) |
| `--threshold-db` | `-70` | Gate threshold in dB |
| `--no-gate` | off | Disable gate and use all samples |

## Examples

### Basic DC offset removal
```bash
python dc_offset_correction.py track.wav
```

### Preview without modifying files
```bash
python dc_offset_correction.py --dry-run *.wav
```

### Full correction with phase rotation (recommended)
```bash
python dc_offset_correction.py --symmetry phase track.wav
```

### Multiple passes for better symmetry
```bash
# Phase rotation finds a local optimum; running multiple times can improve results
python dc_offset_correction.py --symmetry phase bass.wav
python dc_offset_correction.py --symmetry phase "bass (DC-Corrected).wav"
```

### Conservative symmetry correction
```bash
python dc_offset_correction.py --symmetry-strength 0.5 bass.wav
```

### Symmetry correction by amplitude scaling (if phase doesn't suit your material)
```bash
python dc_offset_correction.py --symmetry rms track.wav
```

## Example Output

```
$ python3 ./dc_offset_correction.py --symmetry phase --dry-run "Djembe (Mono).wav" "Drums (Stereo).wav"
File: Djembe (Mono).wav
 SR: 48000
 Pre-mean: -0.000001  Pre-peak: 0.025983
 Gated-samples: 12000138 (80.257745%)  DC used: -0.000000
 Post-mean: 0.000000  Post-peak: 0.026052
 DC method: mean  Symmetry correction: phase  Gate: on (-70.0 dB)
 Asymmetry mono: pos_count=6005700 neg_count=5994438 (previously pos=6005700 neg=5994438)
  mean-abs pos/neg: 0.001838 / 0.001840  ratio=0.998547 (-0.01 dB) (previously -0.02 dB)
  rms      pos/neg: 0.002371 / 0.002376  ratio=0.997787 (-0.02 dB) (previously -0.03 dB)
  peak     pos/neg: 0.026052 / 0.021343  ratio=1.220642 (+1.73 dB) (previously +1.71 dB)
Dry-run: would write Djembe (Mono) (DC-Corrected).wav
File: Drums (Stereo).wav
 SR: 48000
 Pre-mean: -0.000001, -0.000001  Pre-peak: 0.831722, 0.914206
 Gated-samples: 8730364, 8910128 (58.389272, 59.591546%)  DC used: 0.000000, -0.000000
 Post-mean: 0.000093, 0.000057  Post-peak: 0.831722, 0.914205
 DC method: mean  Symmetry correction: phase  Gate: on (-70.0 dB)
 Asymmetry ch0: pos_count=4422471 neg_count=4307893 (previously pos=4422471 neg=4307893)
  mean-abs pos/neg: 0.021578 / 0.021827  ratio=0.988619 (-0.10 dB) (previously -0.23 dB)
  rms      pos/neg: 0.044165 / 0.044177  ratio=0.999719 (-0.00 dB) (previously -0.15 dB)
  peak     pos/neg: 0.676836 / 0.831722  ratio=0.813777 (-1.79 dB) (previously -1.94 dB)
 Asymmetry ch1: pos_count=4498187 neg_count=4411941 (previously pos=4498187 neg=4411941)
  mean-abs pos/neg: 0.024570 / 0.024856  ratio=0.988496 (-0.10 dB) (previously -0.17 dB)
  rms      pos/neg: 0.049589 / 0.049595  ratio=0.999883 (-0.00 dB) (previously -0.08 dB)
  peak     pos/neg: 0.743941 / 0.914205  ratio=0.813757 (-1.79 dB) (previously -1.87 dB)
Dry-run: would write Drums (Stereo) (DC-Corrected).wav
```

## How It Works

1. **Gating**: Samples below the threshold (default -70 dB) are excluded from analysis to avoid noise floor bias
2. **DC Estimation**: Calculates the median (or mean) of gated samples as the DC offset
3. **DC Removal**: Subtracts the estimated DC from all samples
4. **Symmetry Analysis**: Computes RMS and peak of positive vs negative samples
5. **Symmetry Correction** (if enabled, one of):
   - **Phase rotation** (`--symmetry phase`, recommended): Uses the Hilbert transform to create an analytic signal, then searches for the optimal rotation angle θ that minimizes the RMS difference between positive and negative amplitudes. The rotated signal is: `y(t) = x(t)·cos(θ) − H{x(t)}·sin(θ)`. This changes the waveform shape without altering frequency content or adding harmonics.
   - **Amplitude scaling** (`--symmetry rms` or `peak`): Scales positive or negative samples to match, with smooth tanh blending near zero crossings. Simpler but mathematically equivalent to a piecewise-linear transfer function, which adds even-order harmonic distortion.
6. **Clipping**: Output is clipped to [-1, 1] to prevent overflow

### Why Phase Rotation for asymmetry correction?

Waveform asymmetry often originates from phase relationships between harmonics rather than true amplitude differences. For example, a bass guitar's second harmonic might be slightly phase-shifted relative to the fundamental due to pickup mechanics. Phase rotation finds an angle *theta* that "unwinds" this, making the waveform more symmetric without changing what frequencies are present.

Note that the `--symmetry-strength` parameter controls the impact if the symmetry correction. By default, full correction is applied (1.0), but you may pick any coefficient that suits your purpose. In `phase` mode, the parameter is used for interpolation on the angle *theta* (0.0 for no shift, 1.0 for full shift by *theta*).

**Tip:** The optimization finds a local minimum. For severely asymmetric signals, **running the tool multiple times** on the output can yield progressively better results. Listen and experiment!

**Tip no. 2:** You might not need to correct the asymmetry at all.

## Testing

```bash
pip install pytest
python -m pytest test_dc_correction.py -v
```

## License

GPL3

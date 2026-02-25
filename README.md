# DC Offset & Asymmetry Correction

A small Python tool for correcting **DC offset** and **amplitude asymmetry** in audio files.

## The Problem

### DC Offset (Additive Bias)

When the audio waveform isn't centered around zero. This can be caused by:
- Faulty recording equipment or ADC converters
- Analog hardware with calibration drift
- Some synthesizers and audio interfaces
- Magic or bad luck

**Consequences:** Reduced headroom, clicks at edit points, asymmetric clipping, issues with some plugins.

Top: Negative DC offset, Bottom: Same audio after running `python .\dc_offset_correction.py file.wav`
<img width="1024" height="348" alt="image" src="https://github.com/user-attachments/assets/57843f8e-ad60-40d2-9661-8023c43ad98f" />

As long as no clipping is involved, this correction is pretty easy to do and more or less perfect.

### Amplitude Asymmetry (Multiplicative Bias)

When positive amplitudes are consistently larger (or smaller) than negative ones. Common in:
- Bass guitar with asymmetric pickup response
- Recordings through transformer-based preamps
- Some speaker/microphone combinations
- Magic or bad luck

**Consequences:** Wasted headroom (peak meters show higher than perceived loudness), potential for asymmetric distortion, mastering engineer will have a bad day.

Top: Bass signal with asymmatric amplitude, Bottom: Same audio after running `python .\dc_offset_correction.py --symmetry rms file.wav`:
<img width="945" height="297" alt="image" src="https://github.com/user-attachments/assets/5ec7aa49-42c9-4e4d-b9e1-6e96d5917a2c" />

Use this with caution. The correction algorithms I provide are very basic and are good enough for me, but they may introduce subtle artifacts.

## Features

- **DC offset removal** using mean or median estimation
- Optional **Symmetry correction** to balance positive/negative amplitudes (RMS or peak-based)
- **Gated analysis** — ignores quiet sections for more accurate estimation, parameterized threshold
- **Smooth blending** — avoids distortion at zero crossings when correcting asymmetry
- **Dry-run mode** — preview diagnostics without writing files
- **Preserves audio format** — maintains sample rate, bit depth, and file format

## Installation & Usage

First, clone the repository. 

Then isnstall dependencies:

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
| `--symmetry {none,rms,peak}` | `none` | DC asymmetry estimation and correction method (pick "none" to disable symmetry correction) |
| `--symmetry-strength` | `1.0` | Strength of symmetry correction (recommended range is 0..1) |
| `--smoothing` | `0.02` | Amplitude range for symmetry blending near zero, as fraction of full-scale (mitigates kink when symmetry correcting, disable with 0.0, be careful with bigger values on quiet audio) (0..1) |
| `--threshold-db` | `-70` | Gate threshold in dB |
| `--no-gate` | off | Disable gate and use all samples (basically sets threshold-db to -inf) |

## Examples

### Basic DC offset removal
```bash
python dc_offset_correction.py track.wav
```

### Preview without modifying files
```bash
python dc_offset_correction.py --dry-run *.wav
```

### Full correction (DC + symmetry)
```bash
python dc_offset_correction.py --symmetry rms track.wav
```

### Conservative symmetry correction
```bash
python dc_offset_correction.py --symmetry rms --symmetry-strength 0.5 bass.wav
```

## Example Output

```
$ python3 ./dc_offset_correction.py --symmetry rms --method mean --dry-run "Djembe (Mono).wav" "Drums (Stereo).wav"
File: Djembe (Mono).wav
 SR: 48000
 Pre-mean: -0.000001  Pre-peak: 0.025983
 Gated-samples: 12000138 (80.257745%)  DC used: -0.000000
 Post-mean: 0.000000  Post-peak: 0.026052
 DC method: mean  Symmetry correction: rms  Gate: on (-70.0 dB)
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
 DC method: mean  Symmetry correction: rms  Gate: on (-70.0 dB)
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
4. **Symmetry Analysis**: Computes RMS or peak of positive vs negative samples
5. **Symmetry Correction**: Scales one side to match the other, using smooth tanh blending near zero to avoid discontinuities -- this is the part that is not perfect!
6. **Clipping**: Output is clipped to [-1, 1] to prevent overflow

## Testing

```bash
pip install pytest
python -m pytest test_dc_correction.py -v
```

## License

GPL3

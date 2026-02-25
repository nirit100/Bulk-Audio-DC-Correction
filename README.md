# DC Offset & Asymmetry Correction

A Python tool for correcting **DC offset** and **amplitude asymmetry** in audio files — two common issues in bass recordings that can cause problems during mixing and mastering.

## The Problem

### DC Offset (Additive Bias)
When the audio waveform isn't centered around zero. This can be caused by:
- Faulty recording equipment or ADC converters
- Analog hardware with calibration drift
- Some synthesizers and audio interfaces

**Consequences:** Reduced headroom, clicks at edit points, asymmetric clipping, issues with some plugins.

```
  With DC offset:          After correction:
       ~~~~                     ~~~~
    ──────────  0            ────────── 0
       ~~~~                     ~~~~
  (shifted up)               (centered)
```

### Amplitude Asymmetry (Multiplicative Bias)
When positive amplitudes are consistently larger (or smaller) than negative ones. Common in:
- Bass guitar with asymmetric pickup response
- Recordings through transformer-based preamps
- Some speaker/microphone combinations

**Consequences:** Wasted headroom (peak meters show higher than perceived loudness), potential for asymmetric distortion.

```
  Asymmetric:               After correction:
      ▲▲▲                       ▲▲▲
   ────────  0               ────────  0
       ▼                        ▼▼▼
  (pos > neg)               (balanced)
```

## Features

- **DC offset removal** using mean or median estimation
- **Symmetry correction** to balance positive/negative amplitudes (RMS or peak-based)
- **Gated analysis** — ignores quiet sections for more accurate estimation
- **Smooth blending** — avoids distortion at zero crossings
- **Dry-run mode** — preview diagnostics without writing files
- **Preserves audio format** — maintains sample rate, bit depth, and file format

## Installation

```bash
pip install numpy soundfile
```

## Usage

```bash
python dc_offset_correction.py [options] <files...>
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dry-run` | off | Print diagnostics without writing files |
| `--method {mean,median}` | `median` | DC estimation method (median is robust to outliers) |
| `--symmetry {none,rms,peak}` | `none` | Balance pos/neg amplitudes by RMS or peak |
| `--symmetry-strength` | `1.0` | Correction strength (0.0–1.0) |
| `--smoothing` | `0.02` | Transition width near zero crossings |
| `--threshold-db` | `-70` | Gate threshold in dB |
| `--no-gate` | off | Use all samples (disable gating) |

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

### Batch process bass tracks recursively (PowerShell)
```powershell
python dc_offset_correction.py --dry-run (Get-ChildItem -Path ./Cut -Recurse -Filter "*_Bass (Mono).wav").FullName
```

### Batch process (Bash/Zsh)
```bash
python dc_offset_correction.py --dry-run $(find ./Cut -name "*_Bass (Mono).wav")
```

### Conservative symmetry correction
```bash
python dc_offset_correction.py --symmetry rms --symmetry-strength 0.5 bass.wav
```

## Example Output

```
File: Cut/Song_Bass (Mono).wav
 SR: 48000
 Pre-mean: 0.002341  Pre-peak: 0.847623
 Gated-samples: 1847234 (85.23%)  DC used: 0.002298
 Post-mean: 0.000043  Post-peak: 0.845325
 DC method: median  Symmetry correction: none  Gate: on (-70.0 dB)
 Asymmetry mono: pos_count=923617 neg_count=923617 (previously pos=924521 neg=922713)
  mean-abs pos/neg: 0.142851 / 0.142847  ratio=1.000028 (+0.00 dB) (previously +0.12 dB)
  rms      pos/neg: 0.198234 / 0.198229  ratio=1.000025 (+0.00 dB) (previously +0.15 dB)
  peak     pos/neg: 0.845312 / 0.845325  ratio=0.999985 (-0.00 dB) (previously +0.31 dB)
Processed Cut/Song_Bass (Mono).wav -> Cut/Song_Bass (Mono) (DC-Corrected).wav
```

## How It Works

1. **Gating**: Samples below the threshold (default -70 dB) are excluded from analysis to avoid noise floor bias
2. **DC Estimation**: Calculates the median (or mean) of gated samples as the DC offset
3. **DC Removal**: Subtracts the estimated DC from all samples
4. **Symmetry Analysis**: Computes RMS or peak of positive vs negative samples
5. **Symmetry Correction**: Scales one side to match the other, using smooth tanh blending near zero to avoid discontinuities
6. **Clipping**: Output is clipped to [-1, 1] to prevent overflow

## Testing

```bash
pip install pytest
python -m pytest test_dc_correction.py -v
```

## License

MIT

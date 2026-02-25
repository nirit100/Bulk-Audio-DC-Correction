"""
DC Offset Correction Script
Thin CLI wrapper around dc_correction_lib.
"""
import soundfile as sf
import os
import argparse

from dc_correction_lib import process_audio, format_report


def main():
    parser = argparse.ArgumentParser(description='DC-offset correct audio files')
    parser.add_argument('files', nargs='+', help='Input audio files to process')
    parser.add_argument('--dry-run', action='store_true', help='Print stats but do not actually write files')
    parser.add_argument('--threshold-db', type=float, default=-70.0, help='Gate threshold in dB (default: -70)')
    parser.add_argument('--no-gate', action='store_true', help='Disable gate and use all samples (basically sets threshold-db to -inf)')
    parser.add_argument('--method', choices=['mean', 'median'], default='median', help='DC offset estimation method (median is more robust to outliers; default: median)')
    parser.add_argument('--symmetry', choices=['none', 'rms', 'peak'], default='none', help='DC asymmetry estimation and correction method (pick "none" to disable symmetry correction; default: none)')
    parser.add_argument('--symmetry-strength', type=float, default=1.0, help='Strength of symmetry correction (recommended range is 0..1; default: 1.0)')
    parser.add_argument('--smoothing', type=float, default=0.02, help='Amplitude range for symmetry blending near zero, as fraction of full-scale (mitigates non-continuous curves when symmetry correcting) (0..1; default: 0.02)')
    args = parser.parse_args()

    for path in args.files:
        if not os.path.isfile(path):
            print(f"Warning: {path} not found, skipping")
            continue

        # Read audio (I/O)
        info = sf.info(path)
        audio, sr = sf.read(path, dtype='float32')

        # Process (pure computation)
        result = process_audio(
            audio,
            threshold_db=args.threshold_db,
            method=args.method,
            symmetry=args.symmetry,
            symmetry_strength=max(0.0, min(1.0, args.symmetry_strength)),
            smoothing=max(0.0, args.smoothing),
            use_gate=not args.no_gate,
            sample_rate=sr,
        )

        # Report (formatting)
        print(format_report(result, path))

        # Build output filename
        root = os.path.dirname(path) or '.'
        name, ext = os.path.splitext(os.path.basename(path))
        out_path = os.path.join(root, f"{name} (DC-Corrected){ext}")

        # Write (I/O)
        if not args.dry_run:
            sf.write(out_path, result.audio, sr, subtype=info.subtype)
            print(f"Processed {path} -> {out_path}")
        else:
            print(f"Dry-run: would write {out_path}")


if __name__ == '__main__':
    main()

import os
import argparse
import pandas as pd
from model import train_model
from features import featureExtraction
from visualizations import plot_dwt, plot_waveform


real_DIR = "data/Human"
ai_DIR = "data/AI"

def dataframe_build():
    rows = []
    for filename in os.listdir(real_DIR):
        if filename.lower().endswith((".mp3", ".wav")):
            filepath = os.path.join(real_DIR, filename)
            features = featureExtraction(filepath)

            row = {"filename": filename, "label": 0}
            for i, value in enumerate(features):
                row[f"feature_{i}"] = value
            rows.append(row)

    for filename in os.listdir(ai_DIR):
        if filename.lower().endswith((".mp3", ".wav")):
            filepath = os.path.join(ai_DIR, filename)
            features = featureExtraction(filepath)
            row = {"filename": filename, "label": 1}
            for i, value in enumerate(features):
                row[f"feature_{i}"] = value

            rows.append(row)

    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Train model and optionally produce visualizations")
    parser.add_argument("--plot", action="store_true", help="Generate waveform and DWT visualizations for files")
    parser.add_argument("--samples", type=int, default=2, help="Number of example files per class to plot (ignored with --all)")
    parser.add_argument("--all", action="store_true", help="Generate visualizations for all files in both classes")
    args = parser.parse_args()

    df = dataframe_build()
    print("Dataset shape:", df.shape)

    if args.plot:
        print("Generating visualizations...")
        # If --all, set limit to None to process every file; otherwise use samples
        limit = None if args.all else args.samples

        def _plot_from_dir(directory, limit):
            count = 0
            for filename in os.listdir(directory):
                if limit is not None and count >= limit:
                    break
                if filename.lower().endswith((".mp3", ".wav")):
                    filepath = os.path.join(directory, filename)
                    try:
                        plot_waveform(filepath)
                        plot_dwt(filepath)
                        count += 1
                    except Exception as e:
                        print(f"Failed plotting {filepath}: {e}")

        _plot_from_dir(real_DIR, limit)
        _plot_from_dir(ai_DIR, limit)

    train_model(df)

if __name__ == "__main__":
    main()
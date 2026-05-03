import os
import numpy as np
import pandas as pd
from model import train_model
from features import featureExtraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


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
    df = dataframe_build()
    
    print("Dataset shape:", df.shape)
    train_model(df)

if __name__ == "__main__":
    main()
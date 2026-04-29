import os
import numpy as np
import pandas as pd

from features import featureExtraction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


real_DIR = "data/Real"
ai_DIR = "data/AI"

def dataframe_build():
  rows = []
	for filename in os.listdir(real_DIR):
		filepath = os.path.join(real_DIR, filename)
		features = featureExtraction(filepath)
    row = {"filename": filename, "label": 0}
    for i, value in enumerate(features):
      row[f"feature_{i}"] = value

    rows.append(row)

  for filename in os.listdir(ai_DIR):
    if filename.endswith((".mp3"))
      filepath = os.path.join(ai_DIR, filename)
      features = featureExtraction(filepath)
      row = {"filename": filename, "label": 0}
      for i, value in enumerate(features):
        row[f"feature_{i}"] = value

    rows.append(row)

  return pd.DataFrame(rows)



if __name__ = "__main__":
     main()

from pathlib import Path

import pandas as pd
from sklearn import model_selection


RANDOM_SEED = 257
DATA_PATH = Path("data")
SPLITS_PATH = DATA_PATH / "split"

if __name__ == "__main__":
    SPLITS_PATH.mkdir()
    train_data = pd.read_csv(DATA_PATH / "train.csv")
    df_train, df_test = model_selection.train_test_split(train_data, test_size=0.1, random_state=RANDOM_SEED, stratify=train_data["label"])

    df_train.to_csv(SPLITS_PATH / "train.csv")
    df_test.to_csv(SPLITS_PATH / "test.csv")

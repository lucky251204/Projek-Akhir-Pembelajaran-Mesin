import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "pexels_dataset/metadata_clean.csv"
SEED = 42

df = pd.read_csv(INPUT_CSV)
print(f"total: {len(df)}, keywords: {df['keyword'].nunique()}")

train_parts, val_parts, test_parts = [], [], []

for keyword, group in df.groupby('keyword'):
    train, rest = train_test_split(group, test_size=0.2, random_state=SEED)
    val, test   = train_test_split(rest,  test_size=0.5, random_state=SEED)
    train_parts.append(train)
    val_parts.append(val)
    test_parts.append(test)

df_train = pd.concat(train_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_val   = pd.concat(val_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_test  = pd.concat(test_parts).sample(frac=1, random_state=SEED).reset_index(drop=True)

df_train.to_csv("pexels_dataset/train.csv", index=False)
df_val.to_csv("pexels_dataset/val.csv",     index=False)
df_test.to_csv("pexels_dataset/test.csv",   index=False)

total = len(df_train) + len(df_val) + len(df_test)
print(f"train: {len(df_train)} | val: {len(df_val)} | test: {len(df_test)} | total: {total}")
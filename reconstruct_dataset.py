import numpy as np
import pandas as pd
import random
from sklearn.feature_selection import VarianceThreshold

# ─────────────────────────────────────────────
# 重建正例（加入翻转）
# ─────────────────────────────────────────────
positive_samples = []
positive_pairs = set()
df = pd.read_csv("ChCh-Miner_durgbank-chem-chem.tsv\ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None, names=["drug1", "drug2"])

data = np.load("drug_fingerprints.npz", allow_pickle=True)
drug_fingerprints = dict(zip(data["ids"], data["fps"]))

for _, row in df.iterrows():
    d1, d2 = row["drug1"], row["drug2"]
    if d1 not in drug_fingerprints or d2 not in drug_fingerprints:
        continue
    
    fp1 = drug_fingerprints[d1]
    fp2 = drug_fingerprints[d2]
    
    # 原始顺序
    positive_samples.append((np.concatenate([fp1, fp2]), 1))
    # 翻转顺序
    positive_samples.append((np.concatenate([fp2, fp1]), 1))
    
    # 两个方向都记录，用于负例过滤
    positive_pairs.add((d1, d2))
    positive_pairs.add((d2, d1))

print(f"正例数量: {len(positive_samples)}")  # 之前的两倍

# ─────────────────────────────────────────────
# 重建负例（同样加入翻转）
# ─────────────────────────────────────────────
drug_ids = list(drug_fingerprints.keys())
negative_samples = []
target_count = len(positive_samples)  # 保持1:1

while len(negative_samples) < target_count:
    d1 = random.choice(drug_ids)
    d2 = random.choice(drug_ids)
    
    if d1 == d2:
        continue
    if (d1, d2) in positive_pairs:
        continue
    
    fp1 = drug_fingerprints[d1]
    fp2 = drug_fingerprints[d2]
    
    # 原始顺序
    negative_samples.append((np.concatenate([fp1, fp2]), 0))
    # 翻转顺序
    negative_samples.append((np.concatenate([fp2, fp1]), 0))
    
    # 避免重复采样
    positive_pairs.add((d1, d2))
    positive_pairs.add((d2, d1))

print(f"负例数量: {len(negative_samples)}")

# ─────────────────────────────────────────────
# 合并、打乱、划分
# ─────────────────────────────────────────────
all_samples = positive_samples + negative_samples
random.shuffle(all_samples)

X = np.array([s[0] for s in all_samples])
y = np.array([s[1] for s in all_samples])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

selector = VarianceThreshold(threshold=0.0)  # 去掉方差为0的列
X_train = selector.fit_transform(X_train)
X_val   = selector.transform(X_val)
X_test  = selector.transform(X_test)

print(f"Train: {X_train.shape}")
print(f"Val:   {X_val.shape}")
print(f"Test:  {X_test.shape}")

# 保存
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("y_test.npy", y_test)
print("数据集已保存！")
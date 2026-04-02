import pandas as pd
df = pd.read_csv("ChCh-Miner_durgbank-chem-chem.tsv\ChCh-Miner_durgbank-chem-chem.tsv", sep="\t", header=None, names=["drug1", "drug2"])

# 检查正例里有没有对称对
df_pairs = set(zip(df["drug1"], df["drug2"]))

symmetric_count = 0
for d1, d2 in df_pairs:
    if (d2, d1) in df_pairs:
        symmetric_count += 1

print(f"已有对称对数量: {symmetric_count}")
print(f"总正例数量: {len(df_pairs)}")
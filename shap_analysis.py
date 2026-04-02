import shap
import pickle
import matplotlib.pyplot as plt
import numpy as np

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

with open("rf_model.pkl", "rb") as f:
    rf = pickle.load(f)

# ─────────────────────────────────────────────
# Step 1 — 创建SHAP解释器
# ─────────────────────────────────────────────
explainer = shap.TreeExplainer(rf)

# 用测试集的一个子集（500个样本够了，算全部太慢）
X_sample = X_test[:50]
shap_values = explainer.shap_values(X_sample)

# 兼容新旧版SHAP：旧版返回list，新版返回3D array (n_samples, n_features, n_classes)
if isinstance(shap_values, list):
    shap_pos = shap_values[1]
    expected_val = explainer.expected_value[1]
else:
    shap_pos = shap_values[:, :, 1]
    expected_val = explainer.expected_value[1] if hasattr(explainer.expected_value, '__len__') else explainer.expected_value

# ─────────────────────────────────────────────
# Step 2 — 给特征命名（关键！这样图上显示的是
#           有意义的名字而不是feature_0, feature_1）
# ─────────────────────────────────────────────

# MACCS Keys的166个bit都有固定名字
from rdkit.Chem import MACCSkeys
from rdkit import Chem

# 生成MACCS key的名称列表
fp_dim = X_test.shape[1] // 2
maccs_labels = [f"MACCS_{i}" for i in range(fp_dim)]

# Drug A的特征名
feature_names_a = [f"DrugA_{label}" for label in maccs_labels]
# Drug B的特征名
feature_names_b = [f"DrugB_{label}" for label in maccs_labels]
# 合并
feature_names = feature_names_a + feature_names_b  # 长度 = X_test.shape[1]

# ─────────────────────────────────────────────
# Step 3 — 全局解释：哪些特征整体上最重要
# ─────────────────────────────────────────────

# Summary plot：每个特征的SHAP值分布
plt.figure()
shap.summary_plot(
    shap_pos,
    X_sample,
    feature_names=feature_names,
    max_display=20,      # 只显示最重要的20个特征
    show=False
)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("保存：shap_summary.png")

# Bar plot：特征重要性排名（更简洁）
plt.figure()
shap.summary_plot(
    shap_pos,
    X_sample,
    feature_names=feature_names,
    plot_type="bar",
    max_display=20,
    show=False
)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("保存：shap_bar.png")

# ─────────────────────────────────────────────
# Step 4 — 局部解释：具体某个药对为什么被flagged
# ─────────────────────────────────────────────

# 找一个被正确预测为interaction的样本
n_sample = X_sample.shape[0]
correct_positive_idx = np.where(
    (y_test[:n_sample] == 1) & ((shap_pos.sum(axis=1) + expected_val) > 0.5)
)[0][0]

# Waterfall plot：分解这个预测
shap.waterfall_plot(
    shap.Explanation(
        values=shap_pos[correct_positive_idx],
        base_values=expected_val,
        data=X_sample[correct_positive_idx],
        feature_names=feature_names
    ),
    max_display=15,
    show=False
)
plt.tight_layout()
plt.savefig("shap_waterfall.png", dpi=150, bbox_inches="tight")
plt.close()
print("保存：shap_waterfall.png")

# ─────────────────────────────────────────────
# Step 5 — 打印最重要的特征（方便写报告）
# ─────────────────────────────────────────────

mean_abs_shap = np.abs(shap_pos).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:10]

print("\nTop 10 最重要特征：")
print(f"{'排名':<6} {'特征名':<25} {'平均|SHAP|':<12}")
print("-" * 45)
for rank, idx in enumerate(top_indices, 1):
    print(f"{rank:<6} {feature_names[idx]:<25} {mean_abs_shap[idx]:.4f}")
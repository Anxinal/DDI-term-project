import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report

from torch.utils.data import DataLoader, TensorDataset

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

# 转成tensor
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train)
X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.FloatTensor(y_val)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test)

# DataLoader
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader  = DataLoader(train_dataset, batch_size=256, shuffle=True)

class MLPWithAttention(nn.Module):
    def __init__(self, fp_dim=166, hidden_dim=128):
        super().__init__()
        self.fp_dim = fp_dim
        
        # 分别编码Drug A和Drug B
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attention层：Drug A attend to Drug B
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # 最终分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 拆分Drug A和Drug B的指纹
        drug_a = x[:, :self.fp_dim]           # (batch, 166)
        drug_b = x[:, self.fp_dim:]           # (batch, 166)
        
        # 各自编码
        enc_a = self.encoder(drug_a)           # (batch, 128)
        enc_b = self.encoder(drug_b)           # (batch, 128)
        
        # Attention: A attend to B
        Q = self.W_q(enc_a)
        K = self.W_k(enc_b)
        V = self.W_v(enc_b)

        score = torch.sum(Q * K, dim=1, keepdim=True)
        weight = torch.sigmoid(score)

        # 用attention weight重新加权B的表示
        attended_b = weight * V       
    
        # 拼接A和加权后的B
        combined = torch.cat([enc_a, attended_b], dim=1)  # (batch, 256)
        
        return self.classifier(combined).squeeze()
    
# 训练函数
def train_model(model, train_loader, X_val, y_val, epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    best_val_auc = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_probs = model(X_val).numpy()
            val_auc = roc_auc_score(y_val, val_probs)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val AUC: {val_auc:.4f}")
    
    # 加载最好的模型
    model.load_state_dict(best_model_state)
    print(f"\n最佳Val AUC: {best_val_auc:.4f}")
    return model


# 跑起来
fp_dim = X_train.shape[1] // 2
mlp_attn = MLPWithAttention(fp_dim=fp_dim)
mlp_attn = train_model(mlp_attn, train_loader, X_val_t, y_val_t, epochs=50)

# 评估
mlp_attn.eval()
with torch.no_grad():
    y_prob_attn = mlp_attn(X_test_t).numpy()
    y_pred_attn = (y_prob_attn > 0.5).astype(int)

def evaluate(y_true, y_pred, y_prob, model_name):
    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"AUC-ROC:   {roc_auc_score(y_true, y_prob):.4f}")
    print(f"F1:        {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")

evaluate(y_test, y_pred_attn, y_prob_attn, "MLP + Attention")

results = {}
results["Full Attention"] = {
    "AUC-ROC": roc_auc_score(y_test, y_prob_attn),
    "F1": f1_score(y_test, y_pred_attn),
    "Precision": precision_score(y_test, y_pred_attn),
    "Recall": recall_score(y_test, y_pred_attn)
}

torch.save(mlp_attn.state_dict(), "full_mlp_attn.pth")

import json
with open("full_attention_results.json", "w") as f:
    json.dump(results, f, indent=2)
# 时序版本
# 它使用的是VMF3数据，数据本身就包括了模型所需要的输入，我是用的生成的模拟的数据，所以可能不是很准确，也不是很对
# 但是物理公式和损失函数是和他一样的

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# 加在这里
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================================================================
# 1. 生成带时序的模拟 VMF3 数据集（固定维度）
# ==============================================================================
def generate_sequential_vmf3_data(n_steps=1000, save_path="vmf3_timeseries.csv"):
    np.random.seed(42)
    
    t = np.linspace(0, 10, n_steps)
    P = 950 + 10 * np.sin(0.5*t) + np.random.randn(n_steps)*1
    T = 273 + 10 * np.cos(0.3*t) + np.random.randn(n_steps)*0.5
    e = 10 + 5 * np.sin(0.7*t) + np.random.randn(n_steps)*0.3
    ZHD = 2.2 + np.random.randn(n_steps)*0.01
    ZWD = 0.15 + 0.08 * np.sin(0.6*t) + np.random.randn(n_steps)*0.01
    C1 = np.random.uniform(0.001, 0.003, n_steps)
    C2 = np.random.uniform(0.0005, 0.002, n_steps)
    
    lat = np.full(n_steps, 30.0)
    height = np.full(n_steps, 0.5)
    doy = np.linspace(1, 365, n_steps).astype(int)
    hour = np.tile([0,6,12,18], n_steps//4 + 1)[:n_steps]
    
    df = pd.DataFrame({
        "doy": doy, "hour": hour,
        "lat": lat, "height": height,
        "P": P, "T": T, "e": e,
        "ZHD": ZHD, "C1": C1, "C2": C2, "ZWD_true": ZWD
    })
    df.to_csv(save_path, index=False)
    print(f"✅ 时序数据集已保存")
    return df

df = generate_sequential_vmf3_data(n_steps=2000)

# ==============================================================================
# 2. 【修复】严格构建 55 维输入：7×7=49 + 6=55
# ==============================================================================
def build_55d_features(df, window_size=7):
    feature_cols = ["P", "T", "e", "ZHD", "C1", "C2", "ZWD_true"]  # 7个
    static_cols = ["lat", "height", "doy", "hour", "season", "tmp"] # 6个（补到6维）
    
    X, y = [], []
    for i in range(window_size, len(df)):
        # 历史7个时刻 → 7×7=49维
        seq_feat = df[feature_cols].iloc[i-window_size:i].values.flatten()
        
        # 当前6维静态信息
        static_feat = df[static_cols].iloc[i].values
        
        # 拼接 = 49 + 6 = 55 维 ✅
        feat = np.concatenate([seq_feat, static_feat])
        X.append(feat)
        y.append(df["ZWD_true"].iloc[i])
    
    return np.array(X), np.array(y)

# 增加2个虚拟静态列，确保是6维
df["season"] = 1
df["tmp"] = 0

X, y = build_55d_features(df, window_size=7)
print(f"✅ 输入特征形状：{X.shape}  (必须是 [n,55])")

# ==============================================================================
# 3. 训练测试划分
# ==============================================================================
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.reshape(-1,1), dtype=torch.float32)
y_test = torch.tensor(y_test.reshape(-1,1), dtype=torch.float32)

# ==============================================================================
# 4. 物理公式
# ==============================================================================
def f_phi_h(lat_deg, height_km):
    lat_rad = torch.deg2rad(lat_deg)
    return 1.0 - 0.00266 * torch.cos(2*lat_rad) - 0.00028 * height_km

def saastamoinen_zwd(P, T, e, lat, h):
    f = f_phi_h(lat, h)
    return (0.002277 / f) * (1255.0 / T + 0.05) * e

# ==============================================================================
# 5. PINN 网络（输入=55维）
# ==============================================================================
class PINN_ZWD_55D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(55, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 6. 【修复】损失函数：维度严格匹配
# ==============================================================================
def pinn_loss_55d(model, X, y_true, lambda_phy=0.5):
    y_pred = model(X)
    loss_data = torch.mean((y_pred - y_true)**2)
    
    # 55维：[0~48:49维]  [49~54:6维]
    P = X[:, 42]  # 最新时刻 P
    T = X[:, 43]  # 最新时刻 T
    e = X[:, 44]  # 最新时刻 e
    lat = X[:, 49]
    h = X[:, 50]
    
    zwd_phy = saastamoinen_zwd(P, T, e, lat, h)
    loss_phy = torch.mean((y_pred.squeeze() - zwd_phy)**2)
    
    total_loss = loss_data + lambda_phy * loss_phy
    return total_loss, loss_data, loss_phy

# ==============================================================================
# 7. 训练
# ==============================================================================
model = PINN_ZWD_55D()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 300
lambda_phy = 0.5
train_losses, test_losses = [], []

print("\n🚀 开始训练 55维时序 PINN...")

for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()
    loss, loss_d, loss_p = pinn_loss_55d(model, X_train, y_train, lambda_phy)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        test_loss, _, _ = pinn_loss_55d(model, X_test, y_test, lambda_phy)
    
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

print("\n🎉 55维 PINN 训练完成！")

# ==============================================================================
# 8. 测试与可视化
# ==============================================================================
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

y_test_np = y_test.numpy().flatten()
y_pred_np = y_pred.numpy().flatten()
rmse = np.sqrt(np.mean((y_pred_np - y_test_np)**2))

print(f"\n📌 测试集 RMSE = {rmse:.4f} m ({rmse*100:.2f} cm)")

plt.figure(figsize=(10,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("55D PINN Loss Curve")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,8))
plt.scatter(y_test_np, y_pred_np, s=10, alpha=0.7)
plt.plot([y_test_np.min(), y_test_np.max()],
         [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
plt.xlabel("True ZWD")
plt.ylabel("Pred ZWD")
plt.title("55D PINN Prediction")
plt.grid(True)
plt.show()
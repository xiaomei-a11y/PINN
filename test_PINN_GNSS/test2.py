# 非时序

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 加在这里
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ==============================================================================
# 1. 生成模拟 VMF3 数据集（论文输入格式）
# ==============================================================================
def generate_data(save_path="vmf3_synthetic_data.csv"):
    np.random.seed(42)
    n_samples = 2000

    # 模拟真实物理范围的数据
    lat = np.random.uniform(-60, 60, n_samples)
    height = np.random.uniform(0, 3, n_samples)
    P = np.random.uniform(850, 1050, n_samples)
    T = np.random.uniform(260, 290, n_samples)
    e = np.random.uniform(4, 25, n_samples)

    # 生成真实ZWD（观测值）
    ZWD_true = 0.001 * e + 0.0002 * (290 - T) + np.random.randn(n_samples) * 0.01

    df = pd.DataFrame({
        "lat": lat,
        "height": height,
        "P": P,
        "T": T,
        "e": e,
        "ZWD_true": ZWD_true
    })
    df.to_csv(save_path, index=False)
    print(f"✅ 模拟数据已保存：{save_path}")
    return df

# 生成数据
df = generate_data()

# ==============================================================================
# 2. 加载数据 + 标准化（论文式1）
# ==============================================================================
data = df.values
X = data[:, :-1]  # 输入：lat, height, P, T, e
y = data[:, -1:]  # 标签：ZWD_true

# 划分训练集、测试集（8:2）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 转 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ==============================================================================
# 3. 物理公式实现（论文核心）
# ==============================================================================
def f_phi_h(lat_deg, height_km):
    lat_rad = torch.deg2rad(lat_deg)
    return 1.0 - 0.00266 * torch.cos(2 * lat_rad) - 0.00028 * height_km

def saastamoinen_zwd(P, T, e, lat, h):
    f = f_phi_h(lat, h)
    return (0.002277 / f) * (1255.0 / T + 0.05) * e

# ==============================================================================
# 4. PINN 网络结构（论文 MLP）
# ==============================================================================
class PINN_ZWD(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)  # 输出：预测 ZWD
        )

    def forward(self, x):
        return self.net(x)

# ==============================================================================
# 5. PINN 损失函数（论文式2-4）
# ==============================================================================
def pinn_loss(model, X, y_true, lambda_phy=0.5):
    # 从输入中提取物理量
    lat = X[:, 0:1]
    h = X[:, 1:2]
    P = X[:, 2:3]
    T = X[:, 3:4]
    e = X[:, 4:5]

    # 1. 网络预测 ZWD
    y_pred = model(X)

    # 2. 数据损失：预测 ↔ 观测真值
    loss_data = torch.mean((y_pred - y_true) ** 2)

    # 3. 物理损失：预测 ↔ 物理公式
    zwd_phy = saastamoinen_zwd(P, T, e, lat, h)
    loss_phy = torch.mean((y_pred - zwd_phy) ** 2)

    # 4. 总损失
    total_loss = loss_data + lambda_phy * loss_phy

    return total_loss, loss_data, loss_phy

# ==============================================================================
# 6. 训练配置
# ==============================================================================
model = PINN_ZWD(input_dim=5)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 500
lambda_phy = 0.5  # 论文最优权重

# 记录损失
train_loss_list = []
test_loss_list = []

print("\n🚀 开始训练 PINN...")

# ==============================================================================
# 7. 训练循环
# ==============================================================================
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # 前向 + 损失
    loss, loss_d, loss_p = pinn_loss(model, X_train, y_train, lambda_phy)

    # 反向传播
    loss.backward()
    optimizer.step()

    # 测试集损失
    model.eval()
    with torch.no_grad():
        test_loss, _, _ = pinn_loss(model, X_test, y_test, lambda_phy)

    # 保存损失
    train_loss_list.append(loss.item())
    test_loss_list.append(test_loss.item())

    # 打印日志
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] | "
              f"总损失: {loss:.4f} | 数据损失: {loss_d:.4f} | 物理损失: {loss_p:.4f} | "
              f"测试损失: {test_loss:.4f}")

print("\n🎉 PINN 训练完成！")

# ==============================================================================
# 8. 测试集预测 + 可视化
# ==============================================================================
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# 转 numpy 绘图
y_test_np = y_test.numpy().flatten()
y_pred_np = y_pred.numpy().flatten()

# 绘图
plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label="训练损失", linewidth=2)
plt.plot(test_loss_list, label="测试损失", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PINN 训练损失曲线")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(y_test_np, y_pred_np, s=10, alpha=0.6)
plt.plot([y_test_np.min(), y_test_np.max()],
         [y_test_np.min(), y_test_np.max()],
         'r--', linewidth=2)
plt.xlabel("真实 ZWD")
plt.ylabel("预测 ZWD")
plt.title("PINN 预测效果")
plt.grid(True)
plt.show()

# 计算 RMSE
rmse = np.sqrt(np.mean((y_pred_np - y_test_np) ** 2))
print(f"\n📌 测试集 RMSE = {rmse:.4f} m")
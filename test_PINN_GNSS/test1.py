# 参考论文中的
# 核心公式：
# 1 映射函数： f = 1.0 - 0.00266*cos(2*φ) - 0.00028*h
# 2 Saastamoinen ZWD 物理公式： ZWD = (0.002277 / f) * (1225/T + 0.05 ) * e


import torch
import numpy as np
import pandas as pd
import os

# ==============================================================================
# 1. 生成模拟 VMF3 数据集并保存到本地 (完全按论文字段模拟)
# ==============================================================================
def generate_synthetic_vmf3_data(save_path="vmf3_synthetic_data.csv"):
    """
    生成模拟的 VMF3 气象 + 对流层延迟数据
    包含论文需要的所有字段：纬度φ、高度h、气压P、温度T、水汽压e、ZHD、ZWD_true
    """
    np.random.seed(42)  # 固定随机种子，结果可复现
    n_samples = 1000     # 生成1000条模拟数据

    # 模拟数据（符合真实物理范围）
    lat = np.random.uniform(-60, 60, n_samples)    # 纬度 [-60,60] 度
    height = np.random.uniform(0, 4, n_samples)   # 测站高度 0~4 km
    P = np.random.uniform(800, 1050, n_samples)   # 气压 hPa
    T = np.random.uniform(250, 300, n_samples)    # 绝对温度 K
    e = np.random.uniform(5, 30, n_samples)       # 水汽压 hPa
    ZHD = np.random.uniform(2.0, 2.4, n_samples)   # 天顶干延迟 m
    ZWD_true = np.random.uniform(0.1, 0.3, n_samples) # 真实ZWD（观测值）

    # 构造DataFrame并保存
    df = pd.DataFrame({
        "lat_deg": lat,        # 纬度（度）
        "height_km": height,   # 高度（km）
        "P_hPa": P,            # 气压
        "T_K": T,              # 温度
        "e_hPa": e,            # 水汽压
        "ZHD_m": ZHD,          # 干延迟
        "ZWD_true_m": ZWD_true # 真实ZWD（标签）
    })

    df.to_csv(save_path, index=False, encoding="utf-8")
    print(f"✅ 模拟数据集已保存到：{os.path.abspath(save_path)}")
    return df

# 生成数据
df = generate_synthetic_vmf3_data()

# ==============================================================================
# 2. 核心公式实现：f(φ,h) 映射函数 + Saastamoinen ZWD
# ==============================================================================
def calculate_f_phi_h(lat_deg, height_km):
    """
    计算映射函数 f(φ, h)
    公式：f(φ,h) = 1 - 0.00266*cos(2φ) - 0.00028*h
    输入：
        lat_deg: 纬度（度）
        height_km: 高度（km）
    """
    # 角度转弧度（必须！）
    lat_rad = torch.deg2rad(lat_deg)
    f = 1.0 - 0.00266 * torch.cos(2 * lat_rad) - 0.00028 * height_km
    return f

def saastamoinen_zwd(P, T, e, lat_deg, height_km):
    """
    论文核心：Saastamoinen 物理公式计算 ZWD_phy
    ZWD_phy = (0.002277 / f(φ,h)) * (1255/T + 0.05) * e
    所有单位严格遵循论文标准
    """
    # 1. 计算映射函数
    f = calculate_f_phi_h(lat_deg, height_km)

    # 2. 计算 ZWD（单位：米）
    zwd_phy = (0.002277 / f) * (1255.0 / T + 0.05) * e

    return zwd_phy, f

# ==============================================================================
# 3. 加载本地数据 → 转张量 → 计算论文中的 ZWD_phy
# ==============================================================================
# 读取本地数据
df = pd.read_csv("vmf3_synthetic_data.csv")

# 转 PyTorch 张量
lat = torch.tensor(df["lat_deg"].values, dtype=torch.float32)
height = torch.tensor(df["height_km"].values, dtype=torch.float32)
P = torch.tensor(df["P_hPa"].values, dtype=torch.float32)
T = torch.tensor(df["T_K"].values, dtype=torch.float32)
e = torch.tensor(df["e_hPa"].values, dtype=torch.float32)
ZWD_true = torch.tensor(df["ZWD_true_m"].values, dtype=torch.float32)

# 计算物理公式 ZWD_phy
ZWD_phy, f_val = saastamoinen_zwd(P, T, e, lat, height)

# ==============================================================================
# 4. 输出结果（验证是否正确）
# ==============================================================================
print("\n" + "="*80)
print("📌 前5条数据计算结果（论文核心物理量）")
print("="*80)
for i in range(5):
    print(f"样本{i+1}:")
    print(f"  纬度={lat[i]:.1f}° 高度={height[i]:.1f}km  f={f_val[i]:.4f}")
    print(f"  物理公式ZWD_phy = {ZWD_phy[i]:.4f} m")
    print(f"  观测真实ZWD_true = {ZWD_true[i]:.4f} m")
    print("-" * 60)

# ==============================================================================
# 5. 论文损失函数演示（数据损失 + 物理损失）
# ==============================================================================
# 模拟网络预测值
ZWD_pred = ZWD_true + torch.randn_like(ZWD_true) * 0.01

# 1. 数据损失：预测值 ↔ 观测真值
loss_data = torch.mean((ZWD_pred - ZWD_true) ** 2)

# 2. 物理损失：预测值 ↔ 物理公式
loss_phy = torch.mean((ZWD_pred - ZWD_phy) ** 2)

# 总损失（论文公式）
total_loss = loss_data + 0.5 * loss_phy

print("\n" + "="*50)
print("="*50)
print(f"数据损失 (loss_data) = {loss_data:.6f}")
print(f"物理损失 (loss_phy)  = {loss_phy:.6f}")
print(f"总损失   (total_loss)= {total_loss:.6f}")
print("\n✅ 全部公式运行成功！")
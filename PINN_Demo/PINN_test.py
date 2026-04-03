# 代码使用GNSS、ERA5、RS数据，对ZTD进行改正
# 训练集：80%的GNSS站，2022年12月1-4日所有时刻（结合ERA5数据）
# 验证集：剩下20%的GNSS站
# 测试集：RS数据进行改正效果评定
# GNSS、ERA5数据用于训练，RS数据用于验证

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理公式定义 (Saastamoinen 模型)
# ==========================================
def calculate_zhd(p_surface, lat, height):
    """
    根据气压、纬度和高程计算理论静力延迟 ZHD (单位: m)
    """
    # 确保输入是 Tensor
    phi = lat * np.pi / 180.0
    term = 1 - 0.00266 * torch.cos(2 * phi) - 0.00028 * height / 1000
    zhd = 0.0022768 * p_surface / term
    return zhd

# ==========================================
# 2. 定义 PINN 网络结构 (增加时间维度输入)
# ==========================================
class ZTD_PINN(nn.Module):
    def __init__(self):
        super(ZTD_PINN, self).__init__()
        # 输入: [lon, lat, height, time] -> 4个输入
        # 输出: [ZTD_residual] -> 神经网络预测残差
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 数据准备 (模拟 12 月份真实数据结构)
# ==========================================
def prepare_real_style_data():
    # --- 训练集 (GNSS) ---
    n_gnss = 100 
    gnss_lon = np.random.uniform(110, 112, (n_gnss, 1))
    gnss_lat = np.random.uniform(30, 32, (n_gnss, 1))
    gnss_h = np.random.uniform(0, 2000, (n_gnss, 1))
    gnss_t = np.random.uniform(44562, 44592, (n_gnss, 1)) # 模拟 12 月时间序列
    
    # 模拟含残差的 ZTD (物理值 + 随时间波动的细微残差)
    zhd_base = 2.31 * np.exp(-gnss_h / 7000) 
    residual_true = 0.05 * np.sin(gnss_t) # 模拟真实残差
    ztd_obs = zhd_base + residual_true + np.random.normal(0, 0.002, (n_gnss, 1))
    
    # --- 物理约束集 (ERA5 Grid) ---
    n_grid = 500
    grid_lon = np.random.uniform(110, 112, (n_grid, 1))
    grid_lat = np.random.uniform(30, 32, (n_grid, 1))
    grid_h = np.random.uniform(0, 2500, (n_grid, 1))
    grid_t = np.random.uniform(44562, 44592, (n_grid, 1))
    # 模拟 ERA5 提供的气压
    grid_p = 1013.25 * np.exp(-grid_h / 8000)
    
    return (torch.FloatTensor(np.hstack([gnss_lon, gnss_lat, gnss_h, gnss_t])), torch.FloatTensor(ztd_obs)), \
           (torch.FloatTensor(np.hstack([grid_lon, grid_lat, grid_h, grid_t])), torch.FloatTensor(grid_p))

# ==========================================
# 4. 训练流程 (引入残差学习逻辑)
# ==========================================
def train():
    (x_gnss, y_gnss), (x_grid, p_grid) = prepare_real_style_data()
    
    # --- 标准化参数 (需根据 12 月数据范围手动调整) ---
    # 假设 lon:[110-112], lat:[30-32], height:[0-3000], time:[44562-44592]
    mins = torch.tensor([110.0, 30.0, 0.0, 44562.0])
    maxs = torch.tensor([112.0, 32.0, 3000.0, 44592.0])
    
    def normalize(x):
        return (x - mins) / (maxs - mins)

    x_gnss_norm = normalize(x_gnss)
    x_grid_norm = normalize(x_grid)

    model = ZTD_PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("开始训练 12 月份时空残差 PINN...")
    for epoch in range(3001):
        optimizer.zero_grad()
        
        # --- 1. 物理驱动的基础值计算 (以 ERA5 气压作为先验) ---
        # 对于训练集也需要计算 ZHD 作为基准
        # 注意：实际中训练集的 p_surface 也要通过坐标从 ERA5 提取
        p_gnss_sim = 1013.25 * torch.exp(-x_gnss[:, 2:3] / 8000) # 模拟用
        zhd_gnss_base = calculate_zhd(p_gnss_sim, x_gnss[:, 1:2], x_gnss[:, 2:3])
        
        # --- 2. Data Loss (残差拟合) ---
        # 模型输出的是修正项 res， 预测值 = 基础值 + res
        res_pred_gnss = model(x_gnss_norm)
        ztd_pred_gnss = zhd_gnss_base + res_pred_gnss
        loss_data = torch.mean((ztd_pred_gnss - y_gnss)**2)
        
        # --- 3. Physics Loss (硬约束：总延迟必须大于静力延迟) ---
        res_pred_grid = model(x_grid_norm)
        zhd_grid = calculate_zhd(p_grid, x_grid[:, 1:2], x_grid[:, 2:3])
        
        # 物理规则：ZTD(zhd + res) - ZHD 必须 > 0，即残差项不能让总延迟低于静力延迟
        zwd_violation = -res_pred_grid # 因为 ZTD = ZHD + res, 所以 ZWD = res
        loss_phys = torch.mean(torch.relu(zwd_violation)**2)
        
        # --- 4. 空间平滑约束 (可选) ---
        # 可以增加对 res 梯度的惩罚，防止空间突变
        
        total_loss = loss_data + 0.1 * loss_phys 
        
        total_loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch}: Loss = {total_loss.item():.8f}, Data_RMSE = {np.sqrt(loss_data.item())*1000:.2f} mm, Phys_Violation = {loss_phys.item():.8f}")

    return model, x_gnss, y_gnss, normalize

# ==========================================
# 5. 可视化与验证
# ==========================================
def visualize(model, x_gnss, y_gnss, norm_func):
    # 模拟在某一固定时刻的垂直剖面
    t_fixed = 44577.0 # 12月中旬某时刻
    h_test = torch.linspace(0, 4000, 100).view(-1, 1)
    lon_test = torch.full_like(h_test, 111.0)
    lat_test = torch.full_like(h_test, 31.0)
    t_test = torch.full_like(h_test, t_fixed)
    
    x_test = torch.cat([lon_test, lat_test, h_test, t_test], dim=1)
    x_test_norm = norm_func(x_test)
    
    with torch.no_grad():
        # 计算基础值
        p_test = 1013.25 * torch.exp(-h_test / 8000)
        zhd_base = calculate_zhd(p_test, lat_test, h_test)
        # 获取神经网络修正
        res_pred = model(x_test_norm)
        ztd_pred = zhd_base + res_pred

    plt.figure(figsize=(8, 6))
    plt.plot(h_test.numpy(), ztd_pred.numpy(), 'b-', label='PINN (ZHD + Residual)', linewidth=2)
    plt.plot(h_test.numpy(), zhd_base.numpy(), 'g--', label='Pure Saastamoinen (ZHD)')
    plt.scatter(x_gnss[:50, 2], y_gnss[:50], color='red', s=10, label='GNSS Samples', alpha=0.5)
    plt.xlabel('Height (m)')
    plt.ylabel('ZTD (m)')
    plt.title('Spatiotemporal PINN ZTD Inversion (December)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    trained_model, x_obs, y_obs, norm_fn = train()
    visualize(trained_model, x_obs, y_obs, norm_fn)
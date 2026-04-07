# 时间换算: A=year,B=month,C=hour
# 1 标准日期时间D=DATE(A,1,1)+B-1+C/24
# 2 转当年连续小时E=(D-DATE(A,1,1))*24
# 3 年周期F=sin(2*PI()*B/365.25) G=cos(2*PI()*B/365.25)
# 4 日周期H=sin(2*PI()*C/24) I=cos(2*PI()*C/24)

# 代码逻辑
# 使用Saastamoinen模型计算ZHD（但需要借助ERA5数据的气压值），
# 对PINN输入经度、纬度、高程、时间，然后训练他输出每个时刻的ZWD的预测值（残差项），
# 根据这两部分进行求和，得到ZTD。
# 与GNSS测站的ZTD相比较（数据损失），计算误差损失（物理损失：残差大于等于0），
# 后向传播，调整PINN模型权重，降低ZWD部分的损失。
# 物理损失：单独用ERA5格网的时空特征输入PINN，
# 得到格网点的残差（ZWD），检查残差是否大于等于0。
# 相当于数据损失靠GNSS数据算，物理损失靠ERA5数据算。

# 这些不对
# 研究范围:全国
# 数据及时间范围: GNSS、ERA5数据用于训练，RS数据用于验证 2022年12月1-5日
# GNSS数据:经度\纬度\高程\时间\ZTD观测值
# ERA5数据:经度\纬度\高程\时间\气压(高程最好使用高分辨率DEM提取)
# RS数据:经度\纬度\高程\时间\ZTD观测值(真值)
# 代码使用GNSS、ERA5、RS数据，对ZTD进行改正
# 训练集：80%的GNSS站，2022年12月1-4日所有时刻（结合ERA5数据） train_gnss.csv
# 验证集：剩下20%的GNSS站 grid_physics.csv(物理约束集)ERA5数据
# 测试集：RS数据进行改正效果评定 val_res.csv(精度评定)RS数据


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. 物理公式定义 (Saastamoinen 模型)计算干延迟ZHD
# （应该使用ERA5提供的气压、纬度、高程数据）
# ==========================================
def calculate_zhd(p_surface, lat, height):
    """
    根据气压、纬度和高程计算理论静力延迟 ZHD (单位: m)
    """
    # 确保输入是 Tensor
    phi = lat * np.pi / 180.0 # 纬度从角度转为弧度
    term = 1 - 0.00266 * torch.cos(2 * phi) - 0.00028 * height / 1000
    # ZHD和地表气压正相关
    zhd = 0.0022768 * p_surface / term
    return zhd # 返回ERA5计算的ZHD,作为基础值，神经网络学习残差部分

# ==========================================
# 2. 定义 PINN 网络结构 (增加时间维度输入)
# 应该输入经纬度+高程+时间,输出ZTD残差
# ==========================================
class ZTD_PINN(nn.Module):
    def __init__(self):
        super(ZTD_PINN, self).__init__()
        # 神经网络结构:4输入-64-64-64-1输出
        # 输入: [lon, lat, height, time] -> 4个输入
        # 输出: [ZTD_residual] -> 神经网络预测残差
        self.net = nn.Sequential(
            nn.Linear(4, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # 前向传播:输入x-网络-输出残差
        return self.net(x)

# ==========================================
# 3. 数据准备 (模拟 12 月份真实数据结构)
# ==========================================
def prepare_real_style_data():
    # --- 训练集 (GNSS) ---
    n_gnss = 100 # 100个GNSS站点，每个站点多个时间点的观测
    gnss_lon = np.random.uniform(110, 112, (n_gnss, 1)) # 经度范围
    gnss_lat = np.random.uniform(30, 32, (n_gnss, 1)) # 纬度范围
    gnss_h = np.random.uniform(0, 2000, (n_gnss, 1)) # 高程范围0-2000m
    gnss_t = np.random.uniform(44562, 44592, (n_gnss, 1)) # 模拟 12 月时间序列
    
    # 模拟含残差的 ZTD (物理值 + 随时间波动的细微残差)
    zhd_base = 2.31 * np.exp(-gnss_h / 7000)  # 用高度模型ZHD基础值(高度越高,ZHD越小)
    residual_true = 0.05 * np.sin(gnss_t) # 模拟真实残差
    ztd_obs = zhd_base + residual_true + np.random.normal(0, 0.002, (n_gnss, 1))
    
    # --- 物理约束集 (ERA5 Grid) ---
    n_grid = 500 # 500个网格点，覆盖全国范围
    grid_lon = np.random.uniform(110, 112, (n_grid, 1))
    grid_lat = np.random.uniform(30, 32, (n_grid, 1))
    grid_h = np.random.uniform(0, 2500, (n_grid, 1))
    grid_t = np.random.uniform(44562, 44592, (n_grid, 1))
    # 模拟 ERA5 提供的气压:高度越高,气压越低（标准大气压高公式）
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
    
    # 数据标准化
    def normalize(x):
        return (x - mins) / (maxs - mins)

    x_gnss_norm = normalize(x_gnss)
    x_grid_norm = normalize(x_grid)

    # 创建模型和优化器
    model = ZTD_PINN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("开始训练 12 月份时空残差 PINN...")
    for epoch in range(3001):
        optimizer.zero_grad()
        
        # --- 1. 物理驱动的基础值计算 (以 ERA5 气压作为先验) ---
        # 对于训练集也需要计算 ZHD 作为基准
        # 注意：实际中训练集的 p_surface 也要通过坐标从 ERA5 提取
        # 计算GNSS站点的基础ZHD，作为残差学习的基准
        p_gnss_sim = 1013.25 * torch.exp(-x_gnss[:, 2:3] / 8000) # 模拟用
        zhd_gnss_base = calculate_zhd(p_gnss_sim, x_gnss[:, 1:2], x_gnss[:, 2:3])
        
        # --- 2. Data Loss (残差拟合) ---
        # 模型输出的是修正项 res， 预测值 = 基础值 + res
        # 数据损失:让网络输出残差,拟合GNSS观测
        # ZTD预测=ZHD物理值+网络输出的残差
        res_pred_gnss = model(x_gnss_norm)
        ztd_pred_gnss = zhd_gnss_base + res_pred_gnss
        loss_data = torch.mean((ztd_pred_gnss - y_gnss)**2)
        
        # --- 3. Physics Loss (硬约束：总延迟必须大于静力延迟) ---
        # 物理损失约束：ZTD必须大于ZHD，即残差不能让总延迟低于静力延迟
        res_pred_grid = model(x_grid_norm)
        zhd_grid = calculate_zhd(p_grid, x_grid[:, 1:2], x_grid[:, 2:3])
        
        # 物理规则：ZTD(zhd + res) - ZHD 必须 > 0，即残差项不能让总延迟低于静力延迟
        # 物理惩罚：如果违反了物理规则（即残差过大导致总延迟小于静力延迟），则增加损失
        zwd_violation = -res_pred_grid # 因为 ZTD = ZHD + res, 所以 ZWD = res
        loss_phys = torch.mean(torch.relu(zwd_violation)**2)
        
        # --- 4. 空间平滑约束 (可选) ---
        # 可以增加对 res 梯度的惩罚，防止空间突变
        # 设置数据损失和物理损失的不同权重
        total_loss = loss_data + 15 * loss_phys 
        
        # 反向传播，更新参数
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
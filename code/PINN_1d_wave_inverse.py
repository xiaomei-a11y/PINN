# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:29:24 2025

@author: WeiZh
"""

# # **物理信息神经网络（PINNs）求解一维波动方程的逆问题**
# ## 1. 波动方程解析解与PINNs实现

# 本Notebook实现了波动方程的解析解，并通过物理信息神经网络（PINNs）求解一维波动方程的逆问题。以下是波动方程的具体定义：

# ### 初始条件：
# 波动方程的初始条件为：
# $
# u(x, 0) = \sin(\pi x) + \sin(2\pi x),
# $
# 表示初始时刻 $t = 0$ 下，空间 $x$ 的初始状态。

# ### 边界条件：
# 波动方程在边界上的条件为：
# $
# u(0, t) = 0, \quad u(1, t) = 0,
# $
# 表示在 $x = 0$ 和 $x = 1$ 的位置上，波动的值始终为 0。

# ### 解析解：
# 根据 d'Alembert 公式，波动方程的解析解为：
# $
# u(x, t) = \frac{1}{2} \left[f(x + ct) + f(x - ct)\right],
# $
# 其中 $f(x) = \sin(\pi x) + \sin(2\pi x)$，$c$ 为波速。

# ### 目标：
# 通过 PINNs 反演波速 $c$（记为 $c_{\text{hat}}$），利用观测数据和物理约束优化模型，并记录 $c_{\text{hat}}$ 和数据损失 $\text{loss_data}$ 的变化，对比解析解验证模型的精度。

# 引入必要的库
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决某些环境下可能出现的库冲突问题

# 创建结果保存文件夹（如果不存在）
os.makedirs("Results", exist_ok=True)

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
from timeit import default_timer

# 设置计算设备（优先使用 GPU，若无则用 CPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置默认浮点精度为 float32
torch.set_default_dtype(torch.float32)

# 设置随机种子以保证结果可重复
torch.manual_seed(42)
np.random.seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)

# ## **2. 定义 PINNs 模型**
class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=50, num_hidden=3, activation='tanh'):
        """
        初始化 PINN 模型。
        参数:
        input_dim: 输入维度，默认 2（x 和 t）----在GNSS中，x一般有多个量（如纬度、高程、）
        output_dim: 输出维度，默认 1（u）
        hidden_dim: 隐藏层神经元数量，默认 50
        num_hidden: 隐藏层数量，默认 3
        activation: 激活函数类型，默认 'tanh'，可选 'sin'
        """
        super(PINN, self).__init__()
        
        # 可学习参数 c_hat（波速）---初始化逆问题物理参数<-与正问题的不同之处。初始化[0, 1]之间的随机值，允许求导
        self.c_hat = nn.Parameter(torch.tensor([np.random.rand()], requires_grad=True).float())

        # 定义神经网络层
        self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
        for _ in range(num_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        # 设置激活函数
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sin':
            self.activation = torch.sin
        else:
            raise ValueError("不支持的激活函数，仅支持 'tanh' 或 'sin'。")

    def forward(self, x, t):
        """前向传播，输入 x 和 t，输出预测值 u"""
        input_combined = torch.cat([x, t], dim=-1)  # 拼接输入 [batch_size, 2]
        out = input_combined
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)  # 最后一层无激活函数
        return out
    
    def calculate_derivatives(self, x, t):
        """
        计算 PDE 残差所需的二阶导数。

        参数:
        x: 空间坐标
        t: 时间坐标

        返回:
        u_pred: 预测值
        u_xx: 二阶空间导数
        u_tt: 二阶时间导数
        """
        x = x.requires_grad_()
        t = t.requires_grad_()
        u_pred = self.forward(x, t)
        
        u_x = grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        u_t = grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_tt = grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        
        return u_pred, u_xx, u_tt
    
    def losses(self, x, t, u_data):
        """
        计算所有损失项，包括 PDE 残差、初始条件、边界条件和数据损失。

        参数:
        x: 空间坐标（字典格式）
        t: 时间坐标（字典格式）
        u_data: 观测数据（字典格式）

        返回:
        total_loss: 总损失
        loss_pde: PDE 残差损失
        loss_ic1: 初始条件 1 损失
        loss_ic2: 初始条件 2 损失
        loss_bc: 边界条件损失
        loss_data: 数据损失
        """
        # 计算 PDE 残差
        u_pde, u_xx, u_tt = self.calculate_derivatives(x['pde'], t['pde'])
        residual = u_tt - self.c_hat**2 * u_xx
        loss_pde = torch.mean(residual**2)
        
        # 计算初始条件损失
        x_ic = x['ic'].requires_grad_()
        t_ic = t['ic'].requires_grad_()
        u_ic = self.forward(x_ic, t_ic)
        u_t = grad(u_ic, t_ic, grad_outputs=torch.ones_like(u_ic), create_graph=True)[0]
        loss_ic1 = torch.mean((u_ic - torch.sin(np.pi * x['ic']) - torch.sin(2 * np.pi * x['ic']))**2)
        loss_ic2 = torch.mean(u_t**2)

        # 计算边界条件损失
        loss_bc = torch.mean(self.forward(x['bc'], t['bc'])**2)

        # 计算数据损失
        u_pred_data = self.forward(x['data'], t['data'])
        loss_data = torch.mean((u_pred_data - u_data['data'])**2)

        # 总损失
        # total_loss = loss_pde + loss_ic1 + loss_ic2 + loss_bc + 10*loss_data # 这里调整了权重为10
        total_loss = loss_pde + loss_ic1 + loss_ic2 + loss_bc + loss_data # 2
        # 尝试不同的损失函数权重，经验：导数阶数越高的loss项权重越低
        # 求导之后会放大误差

        # 初始化损失历史（如果尚未初始化）
        if not hasattr(self, 'history'):
            self.history = {
                'total_loss': [],
                'loss_pde': [],
                'loss_ic1': [],
                'loss_ic2': [],
                'loss_bc': [],
                'loss_data': [],
                'c_hat': []
            }

        # 记录损失和 c_hat
        self.history['total_loss'].append(total_loss.item())
        self.history['loss_pde'].append(loss_pde.item())
        self.history['loss_ic1'].append(loss_ic1.item())
        self.history['loss_ic2'].append(loss_ic2.item())
        self.history['loss_bc'].append(loss_bc.item())
        self.history['loss_data'].append(loss_data.item())
        self.history['c_hat'].append(self.c_hat.item())

        return total_loss, loss_pde, loss_ic1, loss_ic2, loss_bc, loss_data

# ## **3. 数据生成模块**
def get_data(c, batch_size_pde, batch_size_ic, batch_size_bc, num_data_points=100):
    """
    生成训练和测试数据，用于 PINNs 求解波动方程的逆问题。

    参数:
    c: 真实波速
    batch_size_pde: PDE 点的批量大小
    batch_size_ic: 初始条件点的批量大小
    batch_size_bc: 边界条件点的批量大小
    num_data_points: 观测数据点的数量，默认 100

    返回:
    X, T: 网格化的空间和时间坐标
    x_test, t_test, u_test: 测试数据的坐标和真实解
    x_train, t_train, u_train: 训练数据的坐标和真实解（字典格式）
    U: 解析解的网格数据
    """
    # 定义空间和时间域，均匀划分 101 个点
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)
    
    # 使用 d'Alembert 公式计算解析解
    f = lambda x: np.sin(np.pi * x) + np.sin(2 * np.pi * x)
    U = 0.5 * (f(X + c * T) + f(X - c * T))
    
    # 生成 PDE 点：从内部区域随机采样，排除边界
    x_pde = x[1:-1]  # 排除边界点 x=0 和 x=1
    t_pde = t[1:-1]  # 排除边界点 t=0 和 t=1
    X_pde, T_pde = np.meshgrid(x_pde, t_pde)
    X_pde_flat = X_pde.flatten()[:, None]
    T_pde_flat = T_pde.flatten()[:, None]
    U_pde_flat = U[1:-1, 1:-1].flatten()[:, None]
    
    indices_pde = np.random.choice(len(X_pde_flat), batch_size_pde, replace=False)
    x_pde_train = X_pde_flat[indices_pde]
    t_pde_train = T_pde_flat[indices_pde]
    u_pde_train = U_pde_flat[indices_pde]
    
    # 生成初始条件（IC）点：t=0, x 在 (0,1) 区间内
    x_ic = x[1:-1]  # 排除 x=0 和 x=1
    t_ic = np.zeros_like(x_ic)
    u_ic = U[0, 1:-1]
    
    indices_ic = np.random.choice(len(x_ic), batch_size_ic, replace=False)
    x_ic_train = x_ic[indices_ic][:, None]
    t_ic_train = t_ic[indices_ic][:, None]
    u_ic_train = u_ic[indices_ic][:, None]
    
    # 生成边界条件（BC）点：x=0 和 x=1, t 在 (0,1) 区间内
    t_bc = t[1:]  # 排除 t=0
    x_bc_left = np.zeros_like(t_bc)
    x_bc_right = np.ones_like(t_bc)
    u_bc_left = U[1:, 0]
    u_bc_right = U[1:, -1]
    
    x_bc = np.concatenate((x_bc_left, x_bc_right))[:, None]
    t_bc = np.concatenate((t_bc, t_bc))[:, None]
    u_bc = np.concatenate((u_bc_left, u_bc_right))[:, None]
    
    indices_bc = np.random.choice(len(x_bc), batch_size_bc, replace=False)
    x_bc_train = x_bc[indices_bc]
    t_bc_train = t_bc[indices_bc]
    u_bc_train = u_bc[indices_bc]
    
    # 生成观测数据点
    x_test_flat = X.flatten()[:, None]
    t_test_flat = T.flatten()[:, None]
    u_test_flat = U.flatten()[:, None]
    indices_data = np.random.choice(len(x_test_flat), num_data_points, replace=False)
    x_data_train = x_test_flat[indices_data]
    t_data_train = t_test_flat[indices_data]
    u_data_train = u_test_flat[indices_data]
    
    # 将数据组织成字典格式
    x_train = {'pde': x_pde_train, 'ic': x_ic_train, 'bc': x_bc_train, 'data': x_data_train}
    t_train = {'pde': t_pde_train, 'ic': t_ic_train, 'bc': t_bc_train, 'data': t_data_train}
    u_train = {'pde': u_pde_train, 'ic': u_ic_train, 'bc': u_bc_train, 'data': u_data_train}
    
    # 转换为 PyTorch 张量并移动到指定设备
    for key in x_train.keys():
        x_train[key] = torch.from_numpy(x_train[key]).float().to(device)
        t_train[key] = torch.from_numpy(t_train[key]).float().to(device)
        u_train[key] = torch.from_numpy(u_train[key]).float().to(device)
    
    # 生成测试数据：完整网格
    x_test = x_test_flat
    t_test = t_test_flat
    u_test = u_test_flat
    
    x_test = torch.from_numpy(x_test).float().to(device)
    t_test = torch.from_numpy(t_test).float().to(device)
    u_test = torch.from_numpy(u_test).float().to(device)
    
    return X, T, x_test, t_test, u_test, x_train, t_train, u_train, U

# ## **4. 训练函数**
def train_pinn_adam(pinn, x_train, t_train, u_train, epochs=5000, lr=1e-3):
    """
    使用 Adam 优化器训练 PINN 模型。

    参数:
    pinn: PINN 模型实例
    x_train: 训练数据的空间坐标（字典格式）
    t_train: 训练数据的时间坐标（字典格式）
    u_train: 训练数据的真实解（字典格式）
    epochs: 训练的总轮次，默认 5000
    lr: 学习率，默认 1e-3
    """
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    print("开始 Adam 训练...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss, loss_pde, loss_ic1, loss_ic2, loss_bc, loss_data = pinn.losses(x_train, t_train, u_train)
        total_loss.backward(retain_graph=True)
        optimizer.step()
        if epoch % 500 == 0:
            print(f"轮次 {epoch}: 总损失 = {total_loss.item():.6e}, "
                  f"PDE损失 = {loss_pde.item():.6e}, "
                  f"IC1损失 = {loss_ic1.item():.6e}, "
                  f"IC2损失 = {loss_ic2.item():.6e}, "
                  f"BC损失 = {loss_bc.item():.6e}, "
                  f"数据损失 = {loss_data.item():.6e}, "
                  f"c_hat = {pinn.c_hat.item():.6f}")
    print("Adam 训练完成。")

def train_pinn_lbfgs(pinn, x_train, t_train, u_train, max_iter=500, lr=0.1, tolerance_grad=1e-7, tolerance_change=1e-9):
    """
    使用 L-BFGS 优化器训练 PINN 模型。

    参数:
    pinn: PINN 模型实例
    x_train: 训练数据的空间坐标（字典格式）
    t_train: 训练数据的时间坐标（字典格式）
    u_train: 训练数据的真实解（字典格式）
    max_iter: 最大迭代次数，默认 500
    lr: 学习率，默认 0.1
    tolerance_grad: 梯度容忍度，默认 1e-7
    tolerance_change: 损失变化容忍度，默认 1e-9
    """
    optimizer = torch.optim.LBFGS(
        pinn.parameters(),
        lr=lr,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
        history_size=100
    )

    def closure():
        optimizer.zero_grad()
        total_loss, _, _, _, _, _ = pinn.losses(x_train, t_train, u_train)
        total_loss.backward(retain_graph=True)
        return total_loss

    print("开始 L-BFGS 训练...")
    optimizer.step(closure)
    print("L-BFGS 训练完成。")

def plot_sampling_points(x_train, t_train):
    """
    可视化训练数据的采样点，用不同颜色区分 PDE、IC、BC 和数据点。

    参数:
    x_train: 训练数据的空间坐标（字典格式）
    t_train: 训练数据的时间坐标（字典格式）
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(x_train['pde'].cpu().numpy(), t_train['pde'].cpu().numpy(), s=10, c='b', label='PDE Points')
    plt.scatter(x_train['ic'].cpu().numpy(), t_train['ic'].cpu().numpy(), s=10, c='r', label='Initial Condition Points')
    plt.scatter(x_train['bc'].cpu().numpy(), t_train['bc'].cpu().numpy(), s=10, c='g', label='Boundary Condition Points')
    plt.scatter(x_train['data'].cpu().numpy(), t_train['data'].cpu().numpy(), s=10, c='m', label='Data Points')
    plt.title('Distribution of Sampling Points: PDE, IC, BC, and Data')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Results/sampling_points.png")
    plt.show()

def plot_loss_history(pinn):
    """
    可视化训练过程中的损失和 c_hat 历史。

    参数:
    pinn: 已训练的 PINN 模型实例，损失历史存储在 pinn.history 中
    """
    if not hasattr(pinn, 'history'):
        print("No loss history available.")
        return

    epochs = range(len(pinn.history['total_loss']))

    # 创建双子图：损失和 c_hat
    plt.figure(figsize=(12, 6))

    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.semilogy(epochs, pinn.history['total_loss'], label='Total Loss')
    plt.semilogy(epochs, pinn.history['loss_pde'], label='PDE Residual Loss')
    plt.semilogy(epochs, pinn.history['loss_ic1'], label='Initial Condition 1 Loss')
    plt.semilogy(epochs, pinn.history['loss_ic2'], label='Initial Condition 2 Loss')
    plt.semilogy(epochs, pinn.history['loss_bc'], label='Boundary Condition Loss')
    plt.semilogy(epochs, pinn.history['loss_data'], label='Data Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.grid(True)

    # 绘制 c_hat
    plt.subplot(1, 2, 2)
    plt.plot(epochs, pinn.history['c_hat'], label='Estimated c (c_hat)')
    plt.axhline(y=c_true, color='r', linestyle='--', label='True c')
    plt.title('Estimation of c During Training')
    plt.xlabel('Epochs')
    plt.ylabel('c')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Results/loss_and_c_history.png")
    plt.show()

# ## **5. 训练模型**
# 设置波速和批量大小
c_true = 2  # 真实波速
batch_size_pde = 1000  # PDE 点数量
batch_size_ic = 50     # 初始条件点数量
batch_size_bc = 100    # 边界条件点数量
num_data_points = 100  # 观测数据点数量

# 生成数据
X, T, x_test, t_test, u_test, x_train, t_train, u_train, U = get_data(
    c_true, batch_size_pde, batch_size_ic, batch_size_bc, num_data_points
)

# 可视化采样点
plot_sampling_points(x_train, t_train)

# 初始化模型
model = PINN(
    input_dim=2, 
    output_dim=1, 
    hidden_dim=100, 
    num_hidden=3, 
    activation='tanh'
).to(device)

# 记录训练开始时间
t1 = default_timer()

# 使用 Adam 优化器训练
train_pinn_adam(model, x_train, t_train, u_train, epochs=20000, lr=1e-3)

# 使用 L-BFGS 优化器训练
train_pinn_lbfgs(model, x_train, t_train, u_train, max_iter=10000, lr=0.1)

# 记录训练结束时间
t2 = default_timer()
print(f"训练总耗时：{t2 - t1} 秒")

# 在训练完成后调用此函数
plot_loss_history(model)

# ## **6. 结果评估与可视化**
def relative_l2_error(pred, true):
    """计算相对 L2 误差"""
    return torch.norm(pred - true) / torch.norm(true)

# 计算预测结果
u_pred = model(x_test, t_test).cpu().detach().numpy()
u_error = relative_l2_error(torch.tensor(u_pred), u_test.cpu())
print(f"相对 L2 误差：{u_error.item()}")

# 将预测结果重塑为网格形状
u_pred_reshaped = u_pred.reshape(101, 101)

# 计算绝对误差
abs_error = np.abs(U - u_pred_reshaped)

# 创建可视化图形
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# 绘制真实解
contour_gt = axs[0].contourf(X, T, U, levels=50, cmap="viridis", vmin=np.min(U), vmax=np.max(U))
axs[0].set_title("Ground Truth")
axs[0].set_xlabel("x")
axs[0].set_ylabel("t")
plt.colorbar(contour_gt, ax=axs[0], label="True u(x, t)")

# 绘制预测解
contour_pred = axs[1].contourf(X, T, u_pred_reshaped, levels=50, cmap="viridis", vmin=np.min(U), vmax=np.max(U))
axs[1].set_title("PINN Prediction")
axs[1].set_xlabel("x")
plt.colorbar(contour_pred, ax=axs[1], label="Predicted u(x, t)")

# 绘制绝对误差
contour_error = axs[2].contourf(X, T, abs_error, levels=50, cmap="inferno", vmin=0, vmax=np.max(abs_error))
axs[2].set_title("Absolute Error")
axs[2].set_xlabel("x")
plt.colorbar(contour_error, ax=axs[2], label="Absolute Error")

# 调整布局并保存
plt.tight_layout()
plt.savefig("Results/ground_truth_prediction_error.png")
plt.show()

# 比较训练结果：比较损失曲线、收敛速度、相对L2误差
# 为训练数据添加不同强度的高斯噪声，分析噪声对模型学习的影响（记录学习到的波速和训练损失曲线）

# -*- coding: utf-8 -*-
"""
Modified on Wed Feb 26 23:23:44 2025

@author: WeiZh
"""

# # **物理信息神经网络（PINNs）求解一维波动方程的正问题 - 使用硬约束方法**
# ## 1. 波动方程解析解与PINNs实现

# 引入必要的库
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决某些环境下可能出现的库冲突问题

# 创建结果保存文件夹（如果不存在）
os.makedirs("Results", exist_ok=True)

import torch
import numpy as np
import torch.nn as nn
from torch.autograd import grad # 自动微分
import matplotlib.pyplot as plt
from timeit import default_timer

# 设置计算设备（优先使用GPU，若无则用CPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 设置默认浮点精度为 float32
torch.set_default_dtype(torch.float32)

# 设置随机种子以保证结果可重复
torch.manual_seed(42)
np.random.seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)
    
# ## **2. 定义PINNs模型 - 使用硬约束**
class PINN(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=50, num_hidden=3, activation='tanh'):
        """
        初始化PINN模型。

        参数:
        input_dim: 输入维度，默认2（x和t）
        output_dim: 输出维度，默认1（u）
        hidden_dim: 隐藏层神经元数量，默认50
        num_hidden: 隐藏层数量，默认3
        activation: 激活函数类型，默认'tanh'，可选'sin'---非线性激活函数，一般不用ReLU，因为它不适合处理PDE问题
        """
        super(PINN, self).__init__()

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

    # 前向传播函数，输入x和t，输出预测值u
    def forward(self, x, t):
        """前向传播，输入x和t，输出预测值u"""
        input_combined = torch.cat([x, t], dim=-1)  # 拼接输入 [batch_size, 2]
        out = input_combined
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
        out = self.layers[-1](out)  # 最后一层无激活函数
        
        # 使用硬约束 - 参考您提供的代码中的实现
        # out = torch.sin(np.pi*x) + torch.sin(2*np.pi*x) + torch.sin(np.pi*x)*t**2 * out # 模型的硬约束
        # 模型的硬约束 - 这里使用了一个更复杂的非线性函数来增强模型的表达能力
        out = torch.sin(np.pi*x) + torch.sin(2*np.pi*x) + torch.sin(np.pi*x)*torch.sin(5*np.pi*t**2 * out) 
        
        return out
    
    # 求导数的函数
    def calculate_derivatives(self, x, t):
        """
        计算PDE残差所需的二阶导数。

        参数:
        x: 空间坐标
        t: 时间坐标

        返回:
        u_pred: 预测值
        u_xx: 二阶空间导数
        u_tt: 二阶时间导数
        """
        x = x.requires_grad_() # 使x可求导
        t = t.requires_grad_()
        u_pred = self.forward(x, t) # 得到预测值
        
        # 一阶导数
        u_x = grad(u_pred, x, grad_outputs=torch.ones_like(u_pred),  create_graph=True)[0]
        # 二阶导数
        u_xx = grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        u_t = grad(u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
        u_tt = grad(u_t, t, grad_outputs=torch.ones_like(u_t), create_graph=True)[0]
        
        return u_pred, u_xx, u_tt
    
    # 构建损失函数，计算PDE残差损失
    def losses(self, x, t):
        """
        计算PDE残差损失，虽然使用硬约束满足了边界和初始条件，
        但我们仍然计算所有点的PDE残差
        
        参数:
        x: 训练数据的空间坐标（字典格式）
        t: 训练数据的时间坐标（字典格式）
        
        返回:
        loss_pde: PDE残差损失
        """
        # 将所有类型的点拼接起来计算PDE残差
        x_combined = torch.cat([x['pde'], x['ic'], x['bc']], dim=0)
        t_combined = torch.cat([t['pde'], t['ic'], t['bc']], dim=0)
        
        # 计算PDE残差
        _, u_xx, u_tt = self.calculate_derivatives(x_combined, t_combined) # 调用自动微分的函数
        residual = u_tt - c_true**2 * u_xx  # 波动方程的残差
        loss_pde = torch.mean(residual**2)  # PDE残差的均方误差
        
        # 初始化损失历史（如果尚未初始化）
        if not hasattr(self, 'history'):
            self.history = {
                'loss_pde': []
            }

        # 记录损失
        self.history['loss_pde'].append(loss_pde.item())

        return loss_pde

# ## **3. 数据生成模块**
def get_data(c, batch_size_pde, batch_size_ic, batch_size_bc):
    """
    生成训练和测试数据，包含PDE内部点、初始条件点和边界条件点。
    虽然使用硬约束处理了边界条件和初始条件，但我们仍然包含这些点用于计算残差。

    参数:
    c: 波速
    batch_size_pde: PDE点的批量大小
    batch_size_ic: 初始条件点的批量大小
    batch_size_bc: 边界条件点的批量大小

    返回:
    X, T: 网格化的空间和时间坐标
    x_test, t_test, u_test: 测试数据的坐标和真实解
    x_train, t_train, u_train: 训练数据的坐标和真实解（字典格式）
    U: 解析解的网格数据
    """
    # 定义空间和时间域，均匀划分101个点
    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)
    
    # 使用d'Alembert公式计算解析解
    f = lambda x: np.sin(np.pi * x) + np.sin(2 * np.pi * x)
    U = 0.5 * (f(X + c * T) + f(X - c * T))
    
    # 生成PDE点：从内部区域随机采样
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
    
    # 生成初始条件（IC）点：t=0, x在(0,1)区间内
    x_ic = x[1:-1]  # 排除 x=0 和 x=1
    t_ic = np.zeros_like(x_ic)
    u_ic = U[0, 1:-1]
    
    indices_ic = np.random.choice(len(x_ic), batch_size_ic, replace=False)
    x_ic_train = x_ic[indices_ic][:, None]
    t_ic_train = t_ic[indices_ic][:, None]
    u_ic_train = u_ic[indices_ic][:, None]
    
    # 生成边界条件（BC）点：x=0和x=1, t在(0,1)区间内
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
    
    # 将数据组织成字典格式
    x_train = {'pde': x_pde_train, 'ic': x_ic_train, 'bc': x_bc_train}
    t_train = {'pde': t_pde_train, 'ic': t_ic_train, 'bc': t_bc_train}
    u_train = {'pde': u_pde_train, 'ic': u_ic_train, 'bc': u_bc_train}
    
    # 转换为PyTorch张量并移动到指定设备
    for key in x_train.keys():
        x_train[key] = torch.from_numpy(x_train[key]).float().to(device)
        t_train[key] = torch.from_numpy(t_train[key]).float().to(device)
        u_train[key] = torch.from_numpy(u_train[key]).float().to(device)
    
    # 生成测试数据：完整网格
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    u_test = U.flatten()[:, None]
    
    x_test = torch.from_numpy(x_test).float().to(device)
    t_test = torch.from_numpy(t_test).float().to(device)
    u_test = torch.from_numpy(u_test).float().to(device)
    
    return X, T, x_test, t_test, u_test, x_train, t_train, u_train, U

# ## **4. 训练函数**
def train_pinn_adam(pinn, x_train, t_train, epochs=5000, lr=1e-3):
    """
    使用Adam优化器训练PINN模型。

    参数:
    pinn: PINN模型实例
    x_train: 训练数据的空间坐标（字典格式）
    t_train: 训练数据的时间坐标（字典格式）
    epochs: 训练的总轮次，默认5000
    lr: 学习率，默认1e-3
    """
    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    print("开始Adam训练...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss_pde = pinn.losses(x_train, t_train)
        loss_pde.backward(retain_graph=True)
        optimizer.step()
        if (epoch+1) % 500 == 0:
            print(f"轮次 {epoch+1}: PDE损失 = {loss_pde.item():.6e}")
    print("Adam训练完成。")

def train_pinn_lbfgs(pinn, x_train, t_train, max_iter=500, lr=0.1, tolerance_grad=1e-7, tolerance_change=1e-9):
    """
    使用L-BFGS优化器训练PINN模型。

    参数:
    pinn: PINN模型实例
    x_train: 训练数据的空间坐标（字典格式）
    t_train: 训练数据的时间坐标（字典格式）
    max_iter: 最大迭代次数，默认500
    lr: 学习率，默认0.1
    tolerance_grad: 梯度容忍度，默认1e-7
    tolerance_change: 损失变化容忍度，默认1e-9
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
        loss_pde = pinn.losses(x_train, t_train)
        loss_pde.backward(retain_graph=True)
        return loss_pde

    print("开始L-BFGS训练...")
    optimizer.step(closure)
    print("L-BFGS训练完成。")

def plot_sampling_points(x_train, t_train):
    """
    在一张图上可视化所有采样点，用不同颜色区分PDE、IC和BC点

    参数:
    x_train: 训练数据的空间坐标（字典格式）
    t_train: 训练数据的时间坐标（字典格式）
    """
    plt.figure(figsize=(8, 6))
    
    # 绘制PDE采样点
    plt.scatter(x_train['pde'].cpu().numpy(), t_train['pde'].cpu().numpy(), s=10, c='b', label='PDE Points')
    
    # 绘制初始条件采样点
    plt.scatter(x_train['ic'].cpu().numpy(), t_train['ic'].cpu().numpy(), s=10, c='r', label='Initial Condition Points')
    
    # 绘制边界条件采样点
    plt.scatter(x_train['bc'].cpu().numpy(), t_train['bc'].cpu().numpy(), s=10, c='g', label='Boundary Condition Points')
    
    plt.title('Distribution of Sampling Points: PDE, Initial Condition, and Boundary Condition')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.legend()
    plt.tight_layout()
    plt.savefig("Results/sampling_points.png")
    plt.show()

def plot_loss_history(pinn):
    """
    可视化训练过程中的损失历史。

    参数:
    pinn: 已训练的PINN模型实例
    """
    # 检查是否记录了损失历史
    if not hasattr(pinn, 'history'):
        print("No loss history available.")
        return

    epochs = range(len(pinn.history['loss_pde']))

    plt.figure(figsize=(10, 6))
    plt.semilogy(epochs, pinn.history['loss_pde'], label='PDE Residual Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Log Scale)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Results/loss_history.png")
    plt.show()


# ## **5. 训练模型**
# 设置波速和批量大小
c_true = 10  # 真实波速 - 与参考代码保持一致
batch_size_pde = 1000  # PDE点数量
batch_size_ic = 50     # 初始条件点数量
batch_size_bc = 100    # 边界条件点数量

# 生成数据
X, T, x_test, t_test, u_test, x_train, t_train, u_train, U = get_data(
    c_true, batch_size_pde, batch_size_ic, batch_size_bc
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

# 使用Adam优化器训练
train_pinn_adam(model, x_train, t_train, epochs=20000, lr=1e-3)

# 使用L-BFGS优化器训练
train_pinn_lbfgs(model, x_train, t_train, max_iter=10000, lr=0.1)

# 记录训练结束时间
t2 = default_timer()
print(f"训练总耗时：{t2 - t1} 秒")

# 在训练完成后调用此函数
plot_loss_history(model)

# ## **6. 结果评估与可视化**
def relative_l2_error(pred, true):
    """计算相对L2误差"""
    return torch.norm(pred - true) / torch.norm(true)

# 计算预测结果
u_pred = model(x_test, t_test).cpu().detach().numpy()
u_error = relative_l2_error(torch.tensor(u_pred), u_test.cpu())
print(f"相对L2误差：{u_error.item()}")

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
axs[1].set_title("PINN Prediction (Hard Constraint)")
axs[1].set_xlabel("x")
plt.colorbar(contour_pred, ax=axs[1], label="Predicted u(x, t)")

# 绘制绝对误差
contour_error = axs[2].contourf(X, T, abs_error, levels=50, cmap="inferno", vmin=0, vmax=np.max(abs_error))
axs[2].set_title("Absolute Error")
axs[2].set_xlabel("x")
plt.colorbar(contour_error, ax=axs[2], label="Absolute Error")

# 调整布局并保存
plt.tight_layout()
plt.savefig("Results/hard_constraint_results.png")
plt.show()

# 额外可视化：比较特定时间点的剖面
plt.figure(figsize=(10, 6))
t_idx = 50  # 时间点索引
plt.plot(X[0, :], U[t_idx, :], 'b-', label='True Solution')
plt.plot(X[0, :], u_pred_reshaped[t_idx, :], 'r--', label='PINN Prediction')
plt.title(f'Solution Comparison at t = {T[t_idx, 0]:.2f}')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.legend()
plt.grid(True)
plt.savefig("Results/profile_comparison.png")
plt.show()
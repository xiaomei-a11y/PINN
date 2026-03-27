# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:49:04 2025

@author: WeiZh
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 生成数据
x = torch.linspace(0, 1, 100).reshape(-1, 1)
y = torch.sin(10 * np.pi * x)

# 拆分为训练集、验证集和测试集
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.4, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

# 定义神经网络模型，可以控制层数、每层神经元数以及激活函数
class SimpleNN(nn.Module):
    def __init__(self, num_hidden_layers=1, num_neurons=64, activation_function='relu'):
        super(SimpleNN, self).__init__()
        
        # 创建隐含层
        layers = []
        input_size = 1
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, num_neurons))
            if activation_function == 'relu':
                layers.append(nn.ReLU())
            elif activation_function == 'tanh':
                layers.append(nn.Tanh())
            elif activation_function == 'sigmoid':
                layers.append(nn.Sigmoid())
            input_size = num_neurons
            
        self.hidden = nn.Sequential(*layers)  # 隐藏层
        self.output = nn.Linear(num_neurons, 1)  # 输出层

    def forward(self, x):
        x = self.hidden(x)  # 隐藏层处理
        return self.output(x)

# 初始化模型、损失函数和优化器
model = SimpleNN(num_hidden_layers=2, num_neurons=128, activation_function='tanh')
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练模型
num_epochs = 5000
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # 清除梯度
    output_train = model(x_train)  # 前向传播
    loss_train = criterion(output_train, y_train)  # 计算训练集损失
    loss_train.backward()  # 反向传播
    optimizer.step()  # 更新参数
    
    # 计算验证集损失
    model.eval()
    with torch.no_grad():
        output_val = model(x_val)
        loss_val = criterion(output_val, y_val)
    
    # 保存训练集和验证集的损失
    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())

    # 每500步输出训练和验证集的损失
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss_train.item():.4e}, Val Loss: {loss_val.item():.4e}')

# 绘制训练集和验证集的损失曲线
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Curves')
plt.show()

# Relative L2 error
def relative_l2_error(y_true, y_pred):
    l2_error = torch.norm(y_true - y_pred)  # L2 norm of the error
    l2_true = torch.norm(y_true)  # L2 norm of the true values
    return l2_error / l2_true

# 测试模型
model.eval()
with torch.no_grad():
    predicted_test = model(x_test).detach()

# 计算相对L2误差
relative_l2 = relative_l2_error(y_test, predicted_test)
print(f'Relative L2 Error: {relative_l2.item():.4e}')

# 绘制结果
plt.plot(x.numpy(), y.numpy(), label='True value')
plt.scatter(x_test.numpy(), y_test.numpy(), label='Test value', color='blue', s=20)
plt.scatter(x_test.numpy(), predicted_test.numpy(), label='Predicted value', color='red', s=20)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Test Set Prediction')
plt.show()

# 关于PINN中的硬约束：
# 硬约束在PINNs中用来确保边界条件自动满足。对于一维泊松方程，u(x)=f(x)
# 边界条件为：x=0和x=1处，u(x)=0，其中f(x)=-pai*pai*sin*(pai*x)，方程的解析解为：u(x)=sin(pai*x)
# 硬约束原理：硬约束在PINNs中用来确保边界条件自动满足。构建硬约束，即解：u(x)=x(1-x)N(x)
# 边界条件自动满足：函数形式x(x-1)在边界x=0和x=1处，u(x)=0，这是硬约束
# 神经网络输出：N(x)，没有边界限制，任意预测

# 代码解析：
# 1 PINN类继承自tf.keras.Model的神经网络，包含多个全连接层
#   使用tanh激活函数处理隐藏层，输出层是线性激活
# 2 前向传播：call方法定义了前向传播过程：输入通过各层处理后，输出为N(x)
# 3 自定义训练步骤：使用tf.GradientTape()记录梯度信息，计算神经网络输出N(x)的的二阶导数u_xx预
#   损失函数衡量u(x)和右侧项f(x)的差距，使用优化器更新模型参数
# 4 训练过程：使用Adam优化器迭代更新模型参数，逐步减少损失
#   每100个epoc输出损失值以监控训练过程
# 5 预测与可视化：训练完成后使用模型对测试数据进行预测，绘制模型预测结果与解析解进行对比

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams[
# 'axes.unicode_minus'] = False
# # 定义一个神经网络类，继承自tf.keras.Model
# class PINN(tf.keras.Model):    
# def __init__(self, num_hidden_layers=3, num_neurons_per_layer=20):        
# # 调用父类的初始化函数        
# super(PINN, self).__init__()        
# # 创建一个隐藏层的列表，每个隐藏层是一个全连接层，使用tanh作为激活函数
#         self.hidden_layers = [tf.keras.layers.Dense(num_neurons_per_layer, activation='tanh')                              
# for _ in range(num_hidden_layers)]        
# # 创建输出层，输出为一维，没有激活函数（线性激活）
#         self.output_layer = tf.keras.layers.Dense(1, activation=None)
#     # 定义前向传播的方法    
# def call(self, x):        
# # 初始输入值
#         z = x        
# # 通过每一个隐藏层        
# for layer in self.hidden_layers:
#             z = layer(z)        
# # 输出层得到最终输出        
# return self.output_layer(z)
# # 自定义训练步骤
# @tf.function
# def train_step(model, optimizer, x):    
# # 使用tf.GradientTape()记录梯度信息    
# with tf.GradientTape() as tape:        
# # 预测 u(x) = x(1-x)N(x)，其中N(x)是神经网络的输出
#         u_pred = x * (1 - x) * model(x)                
# # 计算 u'(x) 使用自动微分
#         u_x = tf.gradients(u_pred, x)[0]        
# # 计算 u''(x) 使用自动微分
#         u_xx = tf.gradients(u_x, x)[0]                
# # 定义泊松方程中的右侧项 f(x)
#         f = -np.pi**2 * tf.sin(np.pi * x)        
# # 计算损失函数，衡量 u''(x) 和 f(x) 的差距
#         loss = tf.reduce_mean(tf.square(u_xx - f))        
# # 计算损失函数关于模型参数的梯度
#     gradients = tape.gradient(loss, model.trainable_variables)    
# # 使用优化器更新模型参数
#     optimizer.apply_gradients(zip(gradients, model.trainable_variables))    
# return loss
# # 准备训练数据，生成从0到1的100个点
# x_train = np.linspace(0, 1, 100)[:, None]
# x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
# # 初始化PINN模型和Adam优化器
# model = PINN()
# optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# # 训练模型，迭代1000次
# epochs = 1000
# for epoch in range(epochs):    
# # 每次训练计算损失值
#     loss_value = train_step(model, optimizer, x_train)    
# # 每隔100次输出一次损失值    
# if epoch % 100 == 0:        
# print(f"Epoch {epoch}, Loss: {loss_value.numpy()}")
# # 生成测试数据用于结果可视化
# x_test = np.linspace(0, 1, 100)[:, None]
# x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
# # 使用训练好的模型进行预测
# u_pred = x_test * (1 - x_test) * model(x_test)
# # 绘制预测结果和精确解进行对比
# plt.plot(x_test, u_pred, label='PINN Prediction')
# plt.plot(x_test, np.sin(np.pi * x_test), label='Exact Solution', linestyle='dashed')
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('u(x)')
# plt.title('PINN硬约束解一维泊松方程')
# plt.show()

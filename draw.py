import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    # 使用np.where来处理分段函数
    return np.where(x >= 1, x - 1 + x,
                    np.where(x > 0, 1, 1 - x - x))

# 生成x值
x = np.linspace(-2, 3, 400)  # 从-2到3，生成400个点

# 计算对应的y值
y = f(x)

# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = |x-1| + |x|')
plt.title('Graph of f(x) = |x-1| + |x|')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.show()
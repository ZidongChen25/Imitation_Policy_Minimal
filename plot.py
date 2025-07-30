import numpy as np
import matplotlib.pyplot as plt

# 生成高斯噪声样本
num_samples = 10000
mean = 0.0       # 均值
std_dev = 1.0    # 标准差
noise = np.random.normal(loc=mean, scale=std_dev, size=num_samples)

# 绘制直方图
plt.figure(figsize=(8, 5))
plt.hist(noise, bins=50, alpha=0.7, edgecolor='black')



# 美化
plt.title("Distribution of Gaussian Noise")

plt.ylabel("density")
plt.legend()
plt.tight_layout()

# 显示与保存
plt.show()
# plt.savefig("gaussian_noise_distribution.png")

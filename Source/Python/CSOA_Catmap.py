import numpy as np
import matplotlib.pyplot as plt

# 1. Hàm sinh điểm bằng Cat map
def cat_map(n_points):
    x, y = np.zeros(n_points), np.zeros(n_points)
    x[0], y[0] = np.random.rand(), np.random.rand()  # Khởi tạo ngẫu nhiên điểm đầu
    
    for i in range(1, n_points):
        x[i] = (x[i-1] + y[i-1]) % 1
        y[i] = (x[i-1] + 2 * y[i-1]) % 1
    
    return x, y

# 2. Sinh 50,000 điểm bằng Cat map và ngẫu nhiên
n = 50000
x_cat, y_cat = cat_map(n)
x_rand, y_rand = np.random.rand(n), np.random.rand(n)

# 3. Vẽ biểu đồ phân phối
plt.figure(figsize=(15, 6))

# Biểu đồ Cat map
plt.subplot(1, 2, 1)
plt.scatter(x_cat, y_cat, s=0.1, c='blue', alpha=0.5)
plt.title('Phân phối điểm với Cat map (n=50,000)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# Biểu đồ ngẫu nhiên
plt.subplot(1, 2, 2)
plt.scatter(x_rand, y_rand, s=0.1, c='red', alpha=0.5)
plt.title('Phân phối ngẫu nhiên (n=50,000)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.tight_layout()
plt.show()
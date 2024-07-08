import numpy as np
import matplotlib.pyplot as plt

def lms_identification(d, x, mu, num_coeffs):
    n_samples = len(d)
    w = np.zeros(num_coeffs)  # フィルタ係数の初期化
    y = np.zeros(n_samples)   # 出力信号の初期化
    e = np.zeros(n_samples)   # エラー信号の初期化

    # LMSアルゴリズム
    for n in range(num_coeffs, n_samples):
        x_n = x[n:n-num_coeffs:-1]  # 現在の入力ベクトル
        y[n] = np.dot(w, x_n)       # フィルタの出力
        e[n] = d[n] - y[n]          # エラー計算
        w += 2 * mu * e[n] * x_n    # フィルタ係数の更新

    return y, w, e

# データの生成
np.random.seed(0)
n_samples = 500
num_coeffs = 4

# システムのインパルス応答（例: フィルタ係数）
true_w = np.array([0.5, -0.2, 0.3, -0.4])

# 入力信号（例: 白色ガウスノイズ）
x = np.random.randn(n_samples)

# 出力信号（システムによる生成）
d = np.convolve(x, true_w, mode='full')[:n_samples] + 0.05 * np.random.randn(n_samples)

# LMSアルゴリズムを使用した同定
mu = 0.01  # ステップサイズ
y, w, e = lms_identification(d, x, mu, num_coeffs)

# 結果のプロット
plt.figure(figsize=(12, 8))

# 真のフィルタ係数
plt.subplot(2, 1, 1)
plt.stem(true_w, linefmt='r-', markerfmt='ro', basefmt='r-', label='True Coefficients')
plt.stem(w, linefmt='b-', markerfmt='bo', basefmt='b-', label='Estimated Coefficients')
plt.title('Filter Coefficients')
plt.xlabel('Coefficient Index')
plt.ylabel('Value')
plt.legend()

# 出力信号の比較
plt.subplot(2, 1, 2)
plt.plot(d, label='Desired Signal')
plt.plot(y, label='Output Signal')
plt.title('Output Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

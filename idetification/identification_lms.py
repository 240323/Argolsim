import numpy as np
import matplotlib.pyplot as plt

def lms_identification(d, x, mu, num_coeffs):
    n_samples = len(d)
    h = np.zeros(num_coeffs)  # フィルタ係数の初期化
    y = np.zeros(n_samples)   # 出力信号の初期化
    e = np.zeros(n_samples)   # エラー信号の初期化

    # LMSアルゴリズム
    for n in range(num_coeffs, n_samples):
        x_n = x[n:n-num_coeffs:-1]  # 現在の入力ベクトル
        y[n] = np.dot(h, x_n)       # フィルタの出力
        e[n] = d[n] - y[n]          # エラー計算
        h = h + mu * e[n] * x_n     # フィルタ係数の更新

    return y, h, e

# データの生成
np.random.seed(0)
n_samples = 500
num_coeffs = 8

# システムのインパルス応答（例: フィルタ係数）
true_h = np.array([1, 2, 1, 3, 2, 0, 0, 0])

# 入力信号（例: 白色ガウスノイズ）
x = np.random.randn(n_samples)

# 出力信号（システムによる生成）
d = np.convolve(x, true_h, mode='full')[:n_samples] + 0.05 * np.random.randn(n_samples)

# LMSアルゴリズムを使用した同定
mu = 0.01  # ステップサイズ
y, h, e = lms_identification(d, x, mu, num_coeffs)

# 結果のプロット
plt.figure(figsize=(12, 6))

# 出力信号の比較
plt.plot(d, label='Desired Signal')
plt.plot(y, label='Output Signal')
plt.title('Output Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()




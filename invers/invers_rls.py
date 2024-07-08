import numpy as np
import matplotlib.pyplot as plt

def rls_inverse_modeling(d, x, num_coeffs, delta, lambda_):
    n_samples = len(d)
    w = np.zeros(num_coeffs)  # フィルタ係数の初期化
    P = np.eye(num_coeffs) * delta  # 初期共分散行列
    y = np.zeros(n_samples)   # 出力信号の初期化
    e = np.zeros(n_samples)   # エラー信号の初期化

    # RLSアルゴリズム
    for n in range(num_coeffs, n_samples):
        x_n = x[n:n-num_coeffs:-1]  # 現在の入力ベクトル
        y[n] = np.dot(w, x_n)       # フィルタの出力
        e[n] = d[n] - y[n]          # エラー計算

        # 共分散行列の更新
        k = (P @ x_n) / (lambda_ + x_n.T @ P @ x_n)
        P = (P - np.outer(k, x_n.T @ P)) / lambda_

        # フィルタ係数の更新
        w += k * e[n]

    return y, w, e

# データの生成
np.random.seed(0)
n_samples = 500
num_coeffs = 4

# 元の信号（例: 白色ガウスノイズ）
original_signal = np.random.randn(n_samples)

# システムのインパルス応答（例: フィルタ係数）
system_response = np.array([0.5, -0.2, 0.3, -0.4])

# 観測信号（システムを通過した信号）
observed_signal = np.convolve(original_signal, system_response, mode='full')[:n_samples] + 0.05 * np.random.randn(n_samples)

# RLSアルゴリズムを使用したインバースモデリング
delta = 1.0  # 初期共分散行列の値
lambda_ = 0.99  # 忘却係数
reconstructed_signal, w, e = rls_inverse_modeling(original_signal, observed_signal, num_coeffs, delta, lambda_)

# 結果のプロット
plt.figure(figsize=(12, 8))

# 真のフィルタ係数
plt.subplot(3, 1, 1)
plt.stem(system_response, linefmt='r-', markerfmt='ro', basefmt='r-', label='True System Response')
plt.stem(w, linefmt='b-', markerfmt='bo', basefmt='b-', label='Estimated Inverse Filter Coefficients')
plt.title('Filter Coefficients')
plt.xlabel('Coefficient Index')
plt.ylabel('Value')
plt.legend()

# 観測信号と元の信号の比較
plt.subplot(3, 1, 2)
plt.plot(original_signal, label='Original Signal')
plt.plot(observed_signal, label='Observed Signal')
plt.title('Original vs Observed Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# 再構成された信号と元の信号の比較
plt.subplot(3, 1, 3)
plt.plot(original_signal, label='Original Signal')
plt.plot(reconstructed_signal, label='Reconstructed Signal')
plt.title('Original vs Reconstructed Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

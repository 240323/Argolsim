import numpy as np
import matplotlib.pyplot as plt

def rls_interference_cancelling(d, x, num_coeffs, delta, lambda_):
    n_samples = len(d)
    w = np.zeros(num_coeffs)  # フィルタ係数の初期化
    P = np.eye(num_coeffs) * delta  # 初期共分散行列
    y = np.zeros(n_samples)   # 出力信号の初期化
    e = np.zeros(n_samples)   # エラー信号の初期化

    # RLSアルゴリズムによる干渉キャンセリング
    for n in range(num_coeffs, n_samples):
        x_n = x[n:n-num_coeffs:-1]  # 現在の入力ベクトル
        y[n] = np.dot(w, x_n)       # フィルタの出力（予測干渉信号）
        e[n] = d[n] - y[n]          # エラー信号（干渉キャンセリング後の信号）

        # 共分散行列の更新
        k = (P @ x_n) / (lambda_ + x_n.T @ P @ x_n)
        P = (P - np.outer(k, x_n.T @ P)) / lambda_

        # フィルタ係数の更新
        w += k * e[n]

    return y, e, w

def lms_interference_cancelling(d, x, mu, num_coeffs):
    n_samples = len(d)
    w = np.zeros(num_coeffs)  # フィルタ係数の初期化
    y = np.zeros(n_samples)   # 出力信号の初期化
    e = np.zeros(n_samples)   # エラー信号の初期化

    # LMSアルゴリズムによる干渉キャンセリング
    for n in range(num_coeffs, n_samples):
        x_n = x[n:n-num_coeffs:-1]  # 現在の入力ベクトル
        y[n] = np.dot(w, x_n)       # フィルタの出力（予測干渉信号）
        e[n] = d[n] - y[n]          # エラー信号（干渉キャンセリング後の信号）
        w += 2 * mu * e[n] * x_n    # フィルタ係数の更新

    return y, e, w

# データの生成
np.random.seed(0)
n_samples = 500
num_coeffs = 10

# 望ましい信号（例: 音声信号）
desired_signal = np.sin(0.02 * np.arange(n_samples))

# 干渉信号（例: 高周波ノイズ）
interference_signal = 0.5 * np.sin(0.5 * np.arange(n_samples)) + 0.5 * np.random.randn(n_samples)

# 観測信号（望ましい信号 + 干渉信号）
observed_signal = desired_signal + interference_signal

# 参照信号（干渉信号の一部）
reference_signal = interference_signal + 0.1 * np.random.randn(n_samples)

# LMSアルゴリズムを使用した干渉キャンセリング
mu = 0.01  # ステップサイズ
predicted_interference_lms, cleaned_signal_lms, w_lms = lms_interference_cancelling(observed_signal, reference_signal, mu, num_coeffs)

# RLSアルゴリズムを使用した干渉キャンセリング
delta = 1.0  # 初期共分散行列の値
lambda_ = 0.99  # 忘却係数
predicted_interference_rls, cleaned_signal_rls, w_rls = rls_interference_cancelling(observed_signal, reference_signal, num_coeffs, delta, lambda_)

# 結果のプロット
plt.figure(figsize=(12, 12))

# 望ましい信号
plt.subplot(4, 1, 1)
plt.plot(desired_signal, label='Desired Signal')
plt.title('Desired Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# 観測信号
plt.subplot(4, 1, 2)
plt.plot(observed_signal, label='Observed Signal (with interference)')
plt.title('Observed Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# LMSアルゴリズムによる干渉キャンセリング後の信号
plt.subplot(4, 1, 3)
plt.plot(cleaned_signal_lms, label='LMS Cleaned Signal')
plt.title('LMS Cleaned Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

# RLSアルゴリズムによる干渉キャンセリング後の信号
plt.subplot(4, 1, 4)
plt.plot(cleaned_signal_rls, label='RLS Cleaned Signal')
plt.title('RLS Cleaned Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.legend()

plt.tight_layout()
plt.show()

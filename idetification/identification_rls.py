import numpy as np
import matplotlib.pyplot as plt

def rls_predict(x, d, num_coeffs, delta, lambda_, num_predictions):
    n_samples = len(d)
    w = np.zeros(num_coeffs)  # フィルタ係数の初期化
    P = np.eye(num_coeffs) * delta  # 初期共分散行列
    y = np.zeros(n_samples + num_predictions)  # 出力信号の初期化
    e = np.zeros(n_samples)   # エラー信号の初期化

    # RLSアルゴリズムによるフィルタ係数の更新
    for n in range(num_coeffs, n_samples):
        x_n = x[n:n-num_coeffs:-1]  # 現在の入力ベクトル
        y[n] = np.dot(w, x_n)       # フィルタの出力
        e[n] = d[n] - y[n]          # エラー計算

        # 共分散行列の更新
        k = (P @ x_n) / (lambda_ + x_n.T @ P @ x_n)
        P = (P - np.outer(k, x_n.T @ P)) / lambda_

        # フィルタ係数の更新
        w += k * e[n]

    # 予測ステップ
    for n in range(n_samples, n_samples + num_predictions):
        x_n = y[n-1:n-num_coeffs-1:-1]  # 予測のための入力ベクトル
        y[n] = np.dot(w, x_n)           # 予測値の計算

    return y, w, e

# データの生成
np.random.seed(0)
n_samples = 100
num_coeffs = 3
num_predictions = 10
x = np.random.randn(n_samples + num_predictions)
d = x[:n_samples]  # 目的信号

# RLSアルゴリズムを使用した予測
delta = 1.0  # 初期共分散行列の値
lambda_ = 0.99  # 忘却係数
y, w, e = rls_predict(x, d, num_coeffs, delta, lambda_, num_predictions)

# 結果のプロット
plt.figure(figsize=(12, 6))

# 元の信号
plt.plot(range(len(d)), d, label='Original Signal')
# 予測信号
plt.plot(range(len(y)), y, label='Predicted Signal')

# 予測部分の強調表示
plt.axvline(n_samples, color='r', linestyle='--', label='Prediction Start')
plt.title('RLS Prediction')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.legend()
plt.show()

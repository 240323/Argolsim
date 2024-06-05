#N 以下の正の整数の中で、X の倍数または Y の倍数であるものの個数を算出
#標準入力からN, X, Yを取得
N, X, Y = map(int, input().split())


# Xの倍数の個数を計算
count_x = N // X

# Yの倍数の個数を計算
count_y = N // Y

# XとYの最小公倍数を計算
import math
lcm_xy = X * Y // math.gcd(X, Y)

# XとYの最小公倍数の倍数の個数を計算
count_lcm = N // lcm_xy

# XまたはYの倍数の個数を算出（Xの倍数とYの倍数の和から、XとYの最小公倍数の倍数の個数を引く）
count = count_x + count_y - count_lcm

print(count
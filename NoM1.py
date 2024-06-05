#N 以下の正の整数の中で、X の倍数または Y の倍数であるものの個数を算出
#標準入力からN, X, Yを取得
N, X, Y = map(int, input().split())

count = 0

for i in range(1, N+1):
    if i % X == 0 or i % Y == 0:
        count += 1

print(count)





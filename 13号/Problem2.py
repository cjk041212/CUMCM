import pandas as pd

a1 = 1.1353
a2 = 0.688
a3 = 0.3096
w = int(101.4 * 15)
maxw = int(215.4 * 60 * 4)

price = [a3, a3, a3, a3, a3, a3, a3, a3, a1, a1, a1, a3, a3, a1, a1, a1, a1, a2, a2, a2, a2, a2, a2, a2]
price_repeated = [x for x in price for _ in range(60 * 4)]
df = pd.read_excel('processed_load1.xlsx')
need = df['Interpolated Forecasted Load'].values
for i in range(len(need)):
    need[i] = need[i] * 60 * 4

n = len(price_repeated)
dp = [[-100000.0 for _ in range(maxw + 1)] for _ in range(2)]
dp[0][0] = 0.0
ans = 0
now = 0
for i in range(1, n):
    for j in range(1, maxw):
        w2 = int(min(need[i - 1], w))
        dp[now ^ 1][j] = dp[now][j]
        if j + w2 <= maxw:
            dp[now ^ 1][j] = max(dp[now ^ 1][j], dp[now][j + w2] + price_repeated[i - 1] * w2)
        if j - w >= 0:
            dp[now ^ 1][j] = max(dp[now ^ 1][j], dp[now][j - w] - price_repeated[i - 1] * w)
        ans = max(ans, dp[now ^ 1][j])
    now ^= 1
print(ans)

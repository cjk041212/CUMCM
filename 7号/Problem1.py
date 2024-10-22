import math

C0 = 0.6
C1 = 0.3
t1 = 1.5
C_min = 0.05

k = -math.log(C1 / C0) / t1

t = -math.log(C_min / C0) / k

print(k)
print(t)

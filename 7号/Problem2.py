import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

data = pd.read_excel('xlsx1.xlsx')
# print(data)

data_without_header = data.iloc[1:]
n = data_without_header.iloc[:, 0].to_numpy()
C_0_5h = data_without_header.iloc[:, 1].to_numpy()
C_0_5h = C_0_5h.astype('float')
C_0_5h_transformed = (np.log(C_0_5h / 0.6)) / -0.5
#print(n)
#print(C_0_5h)
print(C_0_5h_transformed)

#def f(X, A, B):
#    return 0.6 * np.exp(-0.5 * (A * X + B))
def f2(X, A, B):
    return A * X + B

popt, pcov = curve_fit(f2, n, C_0_5h_transformed)
#print(popt)
#print(pcov)

a_fit, b_fit = popt
#print(a_fit)
#print(b_fit)

plt.figure(figsize=(8, 6))
plt.scatter(n, C_0_5h_transformed, label='Data')
plt.plot(n, f2(n, popt[0],popt[1]), color='red', label='Fitted model')
plt.xlabel('Number of swimmers')
plt.ylabel('Decomposition rate')
plt.title('Decomposition rate vs Number of swimmers')
plt.legend()
plt.show()

print(a_fit)
print(b_fit)
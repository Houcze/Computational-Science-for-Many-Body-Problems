import numpy as np

s_L = 4
e_L = 16


def get_e1_e2(filepath, threshold=1e-1):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    line_number = next(i for i, line in enumerate(lines) if "Lanczos EigenValue in vec" in line)
    line_above = lines[line_number - 1]
    eigvals = line_above.split('=')[1].split(" ")[2:]
    eigvals = [np.float64(val) for val in eigvals]
    
    e1 = eigvals[0]
    i = 1
    while i < len(eigvals) and abs(e1 - eigvals[i]) < threshold:
        i += 1
    if i == len(eigvals):
        print("Did not find an eigenvalue distinct from e1 within threshold")
        return e1, None
    else:
        e2 = eigvals[i]
        if i != 1:
            print("Degenerated")
        return e1, e2


e1 = []
e2 = []
for L in range(s_L, e_L + 2, 2):
    print('L{}.result checked!'.format(L))
    e1_, e2_ = get_e1_e2("./L{}.result".format(L))
    e1.append(e1_)
    e2.append(e2_)

e1 = np.float64(e1)
e2 = np.float64(e2)

with open("e1.txt", 'w') as f:
    for e1_ in e1:
        f.write(str(e1_))
        f.write('\n')
with open("e2.txt", 'w') as f:
    for e2_ in e2:
        f.write(str(e2_))
        f.write('\n')

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def func1(L, a, b, c):
    return a + b * np.exp(-c * L)

def func2(L, a, b, c):
    return a + b / L + c / (L * L)

x = np.float64(range(s_L, e_L + 2, 2))
popt, pcov = curve_fit(func2, x, e2 - e1)
print(popt)

plt.plot(x, func2(x, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.scatter(x, e2 - e1)

plt.savefig('a2.jpg')

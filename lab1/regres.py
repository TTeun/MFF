import funcs
import matplotlib.pyplot as plt

[x, y] = funcs.x_and_y(100)

[a, b] = funcs.alpha_and_beta(x, y)

r = [b + a * x[0], b + a * x[-1]]

plt.ion()
fig = plt.figure()
plt.plot(x, y, 'o')
plt.plot(r)

plt.grid()

fig.savefig('regression.png')

plt.show()


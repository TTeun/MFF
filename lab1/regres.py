import funcs
import matplotlib.pyplot as plt

[x, y] = funcs.x_and_y(100)

[a, b] = funcs.alpha_and_beta(x, y)

[a2, b2] = funcs.alpha_and_beta_numpy(x, y)

r = [b + a * x[0], b + a * x[-1]]

r2 = [b2 + a2 * x[0], b2 + a2 * x[-1]]

print(a,b,a2,b2)

plt.ion()
fig = plt.figure()
plt.plot(x, y, 'o')
plt.plot([x[0], x[-1]],r)
plt.plot([x[0], x[-1]],r)

plt.grid()

fig.savefig('regression.png')

plt.show()


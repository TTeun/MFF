import numpy as np
import funcs_puls as fp
import matplotlib.pyplot as plt

plt.ion()
plt.figure(1)

p1, = plt.plot([], [])

plt.xlim(0, 1)
plt.ylim(-1, 1)


L = 10
x = list(np.linspace(0, L))

t = 0.05
tmax = 10
while t < tmax:
	plt.ylim([-2., 2.])
	plt.xlim([0., 10.])
	plt.title(t)
	y = fp.periodic(t)
	p1.set_data(x, y)
	plt.plot(x, y, '-o',label='periodic')
	y = fp.transient(t)
	plt.plot(x, y,'-*', label='transient')
	plt.legend()
	plt.draw()
	plt.pause(0.004)
	plt.gcf().clear()
	t += 0.05
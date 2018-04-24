import numpy as np
import funcs_puls as fp
import matplotlib.pyplot as plt

plt.ion()
plt.figure(1)

p_p, = plt.plot([], [], label = 'periodic')
p_t, = plt.plot([], [], label = 'transient')

plt.ylim([-2., 2.])
plt.xlim([0., 10.])


L = 10
x = list(np.linspace(0, L))

times = np.linspace(0, 20, 201)
tmax = 10
for t in times:
	plt.title(t)
	y_p = fp.periodic(t)
	p_p.set_data(x, y_p)
	y_t = fp.transient(t)
	p_t.set_data(x, y_t)
	plt.legend()
	plt.draw()
	plt.pause(0.004)
	t += 0.05
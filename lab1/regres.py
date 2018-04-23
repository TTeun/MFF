'''

This file uses the functions in funcs.py to do the computation for te linear regression assignment.

'''

import funcs
import matplotlib.pyplot as plt
import copy

N = int(1e6) # number of data points

[x, y] = funcs.x_and_y(N) # generate the data

[a, b] = funcs.alpha_and_beta(x, y) # compute alpha and beta without using numpy

[a2, b2] = funcs.alpha_and_beta_numpy(x, y) # compute alpha and beta with using numpy

print(a-a2,b-b2) # print the difference between the two methods

r = [b + a * x[0], b + a * x[-1]] # compute the values of the linear fit without using numpy in x = 0 and x = 1

r2 = [b2 + a2 * x[0], b2 + a2 * x[-1]] # compute the values of the linear fit using numpy in x = 0 and x = 1

plt.ion()
fig = plt.figure()
plt.plot(x, y, 'o') # plot the data
plt.plot([x[0], x[-1]],r) # plot the linear fit without using numpy
plt.plot([x[0], x[-1]],r2) # plot the linear fit with using numpy

plt.grid()

fig.savefig('regression.png')

plt.show()



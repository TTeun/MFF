N=0

fib = [1, 1]



for i in range(N-2):
    fib.append(fib[-1]+fib[-2])

print(fib)

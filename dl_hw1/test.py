import numpy as np
import time


a = list(range(10000))
start_time = time.time()
for _ in range(10000):
    np.random.choice(a, 1)
print(time.time()-start_time)

a = list(range(10000))
start_time = time.time()
np.random.choice(a, 10000, replace=True)
print(time.time()-start_time)
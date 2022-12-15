import numpy as np
import time
import Tempo

N = 10
xy0 = np.ones(N) * 2 

Tempo.Init(xy0, N)
print('PyApp2', Tempo.Foo())
time.sleep(100)

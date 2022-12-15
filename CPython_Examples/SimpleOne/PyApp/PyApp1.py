import numpy as np
import time
import Tempo

class Cl:
    id = 0
    def __init__(self, a):
        self.id = Tempo.init(a)
    def foo(self):
        print(Tempo.Foo(self.id))

cl0 = Cl(2.0)
cl1 = Cl(3.0)
print(Cl.id)

cl0.foo()
cl1.foo()

time.sleep(2)
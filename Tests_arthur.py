from numba import njit
import numpy as np

class MyClass:
    def __init__(self, n):
        self.n = n
        self.data = np.zeros(n, dtype=np.float64)

    @staticmethod
    @njit
    def compute_static(x, n):
        data = np.zeros(n, dtype=np.float64)
        for i in range(n):
            data[i] = x * i
        return np.sum(data)

    def compute(self, x):
        return self.compute_static(x, self.n)

# Example usage
obj = MyClass(10)
print(obj.compute(2.0))


# La fonction d'évaluation d'énergie d'un micro-état va devoir être réécrite
# en vue d'être roulée par Numba. 
# La fonction SciPy de convolution 2D pose actuellement problème.
# Nous allons la réécrire avec du plain Numpy en ayant recours à la FFT.
# On sait depuis PM2 que la convolution peut être réécrite 
# comme une multplication dans l'espace de Fourier conjugué.

"""

from numpy.fft import fft2, ifft2
import numpy as np

def fft_convolve2d(x,y):
    # 2D convolution, using FFT
    fr = fft2(x)
    fr2 = fft2(np.flipud(np.fliplr(y)))
    m,n = fr.shape
    cc = np.real(ifft2(fr*fr2))
    cc = np.roll(cc, -m/2+1,axis=0)
    cc = np.roll(cc, -n/2+1,axis=1)
    return cc
"""

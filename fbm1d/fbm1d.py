import numpy as np

# Fractal motion using Fourier filtering method from 
# The Science of Fractal Images book - Barnsley 1988
# # ALGORITHM SpectralSynthesisFMlD


def FBM1d(L, H):
    '''
    Arguments: 
    L = lenght of array 
    H = 0 < H < 1 determines fractal dimension 
    beta = exponent in the spectral density function (1 < beta < 3)
    rad, phase = polar coordinates of Fourier coefficient
    A, B = real and imaginary parts of Fourier coefficients
    '''
    # Create a vector with zeros
    A = np.zeros(L, dtype = np.complex128)
    
    beta = 2 * H + 1
    
    for i in range(L // 2 - 1):
        rad = np.random.normal() * (i + 1) ** (- beta / 2)
        phase = 2 * np.pi * np.random.random()
        A[i] = rad * np.cos(phase) + rad * np.sin(phase) * 1j
        
    # calculate the fast inverse Fourier transform in 1 dimension for real and imaginary parts of Fourier coefficients
    return np.real(np.fft.ifft(A, n = L // 2))

import numpy as np

def FBM1d(L, H):
    """
    Compute the one-dimensional fractal motion using Fourier filtering method.
   
    This function generate a fractal Brownian motion by spectral representation
    of random functions, this technique is known as spectral synthesis or 
    Fourier filtering method.
    
    The fractal Brownian motion is obtained from real part of fast inverse 
    Fourier transform in one-dimensional of Fourier coefficient.
            
    Parameters
    ----------
    L : int
        Lenght of array.
    H : float
        Hurst exponent which determines fractal dimension (0 < H < 1).
       
    Returns
    -------
    output : float ndarray
        Real part of Fast inverse Fourier transform in 1 dimension of Fourier coefficient
        
    Notes
    -----
    Pseudo code in The Science of Fractal Images book - Barnsley 1988.
    ALGORITHM SpectralSynthesisFM1D
    """
    
    A = np.zeros(L, dtype = np.complex128)
    
    beta = 2 * H + 1 # Exponent in the spectral density function (1 < beta < 3).
    
    for i in range(L // 2 - 1):
        rad = np.random.normal() * (i + 1) ** (- beta / 2) # Polar coordinate of Fourier coefficient.
        phase = 2 * np.pi * np.random.random() # Polar coordinate of Fourier coefficient.
        A[i] = rad * np.cos(phase) + rad * np.sin(phase) * 1j
        
    return np.real(np.fft.ifft(A, n = L // 2)) # Real part of Fourier coefficient.

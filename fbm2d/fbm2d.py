import numpy as np

def FBM2d(L, H):
    """
    Compute the two-dimensional fractal motion using Fourier filtering method.
   
    This function generate a fractal Brownian motion by spectral representation
    of random functions, this technique is known as spectral synthesis or 
    Fourier filtering method.
    
    The fractal Brownian motion is obtained from real part of fast inverse 
    Fourier transform in two-dimensional of Fourier coefficient.
            
    Parameters
    ----------
    L : int
        Lenght of array along one dimension.
    H : float
        Hurst exponent which determines fractal dimension (0 < H < 1).
       
    Returns
    -------
    output : float ndarray
        Real part of Fast inverse Fourier transform in 2 dimensions of Fourier coefficient
        
    Notes
    -----
    Pseudo code SpectralSynthesisFM2D in
    BARNSLEY, Michael F. et al. The science of fractal images. New York: Springer, 1988.
    """
      
    A = np.zeros((L, L), dtype = np.complex128)
    beta = 2 * H + 2 # Exponent in the spectral density function (2 < beta < 4)
    for i in range(L // 2 + 1):
        for j in range(L // 2 + 1):
            phase = 2 * np.pi * np.random.random() # Polar coordinates of Fourier coefficient

            if i != 0 or j != 0:
                rad = np.random.normal() * (i ** 2 + j ** 2) ** (- (beta / 2) / 2) # Polar coordinates of Fourier coefficient
            else:
                rad = 0
            
            A[j, i] = rad * np.cos(phase) + rad * np.sin(phase) * 1j

            if i == 0:
                i0 = 0
            else:
                i0 = L - i
            
            if j == 0:
                j0 = 0
            else:
                j0 = L - j
            
            A[j0, i0] = rad * np.cos(phase) - rad * np.sin(phase) * 1j
    
    A[0, L // 2] = np.real(A[0, L // 2]) + 0j
    A[L // 2, 0] = np.real(A[L // 2, 0]) + 0j
    A[L // 2, L // 2] = np.real(A[L // 2, L // 2]) + 0j
    for i in range(1, L // 2):
        for j in range(1, L // 2):
            phase = 2 * np.pi * np.random.random()
            rad = np.random.normal() * (i ** 2 + j ** 2) ** (- (beta / 2) / 2)

            A[L - j, i] = rad * np.cos(phase) + rad * np.sin(phase) * 1j
            A[j, L - i] = rad * np.cos(phase) - rad * np.sin(phase) * 1j
    
    return np.real(np.fft.ifft2(A)) # Real part of Fourier coefficient
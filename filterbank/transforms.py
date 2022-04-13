import numpy as np
from skimage.util import view_as_windows

from torch import from_numpy, zeros, einsum, max, abs, \
    reshape, squeeze, mul, DoubleTensor, max

from torch.nn.functional import pad, fold

def analysisFB(x, R, window, H):
    '''
    Given data is subject to an analysis filter bank composed by a certain number
    (nBands) channels. If necessary, the filter bank output is critically sampled.

    Note: Within the current implementation, analysis filters are used which exhibit
    brick-wall characteristic

    Source:
    _______
    https://ccrma.stanford.edu/~jos/sasp/Two_Channel_Critically_Sampled_Filter.html
    '''

    N = window.shape[0]

    # Zero padding so that combination of window and stride (decimation factor)
    # is aligned with the input signal

    nWindows = (len(x)-N)/R+1
    L = int((np.ceil(nWindows)-1)*R+N)

    x = pad(x, (0, L - len(x)), 'constant', 0)
    x = x.unfold(0, N, R) * window
    
    Zxx = einsum('kn,wn->wk', H, x)
            
    return Zxx

def synthesisFB(Zxx, I, R, window, H, reduction=True):
    '''
    Given data is subject to synthesis filter bank (Part of perfect
    reconstruction filter bank). If necessary, the filter bank output is
    upsampled.

    Note: Synthesis filter bank must correspond to analysis filter
    bank

    '''
    # Reduce filter length by decimation factor
    window = window[::R // I]
    H = H[:, ::R // I]*window

    # Signal buffers
    C, N = H.shape              # Number of channels, window length
    nWindows, _ = Zxx.shape     # Number of window bins

    #x_rec = zeros((C, (nWindows - 1) * I + N), dtype=float)
    Lout = (nWindows - 1) * I + N
    w = zeros(((nWindows - 1) * I + N))
        
    # Assemble normalization term
    for n in np.arange(nWindows):
        w[n * I:n * I + N] += window**2

    x = einsum('kn,wk->knw', H, Zxx)
    x = reshape(x,(1,-1,x.shape[-1]))
    x_rec = fold(x,output_size=(1,Lout), kernel_size=(1,N), stride=(1,I))
    x_rec = squeeze(x_rec)
    
    if reduction:
        return np.einsum('kn->n', x_rec)/w

    return x_rec/w


# def analysisFB(x, R, window, H):
#     '''
#     Given data is subject to an analysis filter bank composed by a certain number
#     (nBands) channels. If necessary, the filter bank output is critically sampled.

#     Note: Within the current implementation, analysis filters are used which exhibit
#     brick-wall characteristic

#     Source:
#     _______
#     https://ccrma.stanford.edu/~jos/sasp/Two_Channel_Critically_Sampled_Filter.html
#     '''

#     N = window.shape[0]

#     # Zero padding so that combination of window and stride (decimation factor)
#     # is aligned with the input signal

#     nWindows = (len(x)-N)/R+1
#     L = int((np.ceil(nWindows)-1)*R+N)
#     x = np.pad(x,(0,L-len(x)),constant_values=(0,0)) 

#     x = view_as_windows(x,N,R)*window
    
#     # For enabling inversion of the transform, NOLA must be checked:
#     #assert check_COLA(window,N,N-R), '''OverLap add constraint is not met!'''
        
#     Zxx = np.einsum('kn,wn->wk', H,x)
        
#     return Zxx


# def synthesisFB(Zxx, I, R, window, H, reduction=True):
#     '''
#     Given data is subject to synthesis filter bank (Part of perfect
#     reconstruction filter bank). If necessary, the filter bank output is
#     upsampled.

#     Note: Synthesis filter bank must correspond to analysis filter
#     bank

#     '''
#     # Reduce filter length by decimation factor, represent in 
#     # resulting filter in polyphase representation and filter 
#     # signal
#     H = H[:,::R//I]
#     x = np.einsum('kn,wk->wkn',H,Zxx)
#     window = window[::R//I]
    
#     # Signal buffers
#     C, N = H.shape                      # Number of channels, window length
#     nWindows, _ = Zxx.shape             # Number of window bins in spectrogram

#     x_rec = np.zeros((C,(nWindows-1)*I+N),dtype=complex)
#     w = np.zeros(((nWindows-1)*I+N))

#     # Overlap add
#     for n,x_window in enumerate(x):
#         x_rec[:,n*I:n*I+N] += x_window*window
#         w[n*I:n*I+N] += window**2
    
#     if reduction:
#         return np.einsum('kn->n', x_rec)/w

#     return x_rec/w


def dft(N):
    return np.fft.ifft(np.eye(N),norm='ortho')

def mdct(N, M=None, alt=False):
    """ Create MDCT matrix

    Parameters
    ---------
    N : int
        Framelength, number of output coefficients.
    N : int
        Windowlength, number of input samples. Defaults to :code:`2 * N`
    alt : boolean
        Switch to alternative defintion, using shift = + N/2

    Returns
    -------
    kernel : array_like
        MDCT matrix

        To be used in analysis using

        .. code:: python

            spectrum = numpy.einsum('kn,fn->fk', mdctmat, signal)

        and in synthesis using

        .. code:: python

            signal = numpy.einsum('kn,fn->fk', mdctmat.T, spectrum)

    Examples
    -------

    .. plot::

        import matplotlib.pyplot as plt
        from subband import matrix
        mat = matrix.mdct_matrix(64)
        plt.imshow(mat)
        plt.show()

    """
    if M is None:
        M = N * 2
    shift = N // 2
    if not alt:
        shift *= -1
    n, k = np.meshgrid(
        np.arange(M, dtype=float), np.arange(N, dtype=float)
    )
    return np.cos(
        np.pi / N * (n + shift + 1 / 2) * (k + 1 / 2)
    ) / np.sqrt(N / 2)
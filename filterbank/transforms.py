import numpy as np
from torch import zeros, einsum, max, abs, \
    reshape, squeeze, max, is_complex

from torch.nn.functional import pad, fold
import matplotlib.pyplot as plt

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
        
    x = x.reshape((1,-1))
    x = x.unfold(1, N, R) * window
        
    # Transform, where k denotes the frequency bin, n denotes the time sample,
    # w denotes the window bin and b denotes the batch    
    Zxx = einsum('kn,wn->kw', H, x[0,...])
            
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
    C, N = H.shape           # Number of channels, window length
    _, nWindows = Zxx.shape  # Number of window bins
        
    Lout = (nWindows - 1) * I + N
        
    w = zeros((Lout))
        
    # Assemble normalization term
    for n in np.arange(nWindows):
        w[n * I:n * I + N] += window**2
        
    # Transfrom, where k denotes the frequency bin, n denotes time
    # sample, b denotes the batch dimension and w denotes the window  
    x = einsum('kn,kw->knw', H, Zxx)
    x = reshape(x,(1,-1,x.shape[-1]))
    
    x_rec = fold(x,output_size=(1,Lout), kernel_size=(1,N), stride=(1,I))
    x_rec = squeeze(x_rec)
            
    if is_complex(x_rec):
        assert max(abs(x_rec.imag) < 0.00001), '''
        Real world signals are purely real!'''

        x_rec = x_rec.real
    
    if reduction:
        return np.einsum('kn->n', x_rec)/w

    return x_rec/w

def zeropad(x: np.ndarray, C: tuple):
    return pad(x, (C), 'constant', 0)

def crop(x: np.ndarray, C: tuple):
    return x[..., C[0]:-C[1]]


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
from __future__ import division
import numpy as np
from collections import namedtuple
from scipy.special import gamma

SIGMA0 = 2/np.sqrt(3)
EPSILON=1e-6
FBANKTRUNC_T = namedtuple('FBANK_T','psifilters psifilterindices psifilterstarts psitype phifilter phifilterstart phitype')
FBANK_T = namedtuple('FBANK_T','Q B J P phidirac psifilters psitype xipsi sigmapsi phifilter phitype sigmaphi')

#
# relies heavily on scatnet
#
# copyright Mark Stoehr (see LICENSE)

def TtoJ(T,Q=1,B=None,phibwratio=None):
    """
    Compute the maximal wavelet scale J such that for a filter bank
    the largest wavelet is of bandwidth approximately T.

    Parameters:
    -----------
    T: int
       Time bandwidth for window
    Q: int
       Number of wavelets per octave
    B: int
       The reciprocal per-octave bandwidth of the wavelets
    phibwratio: float
       ratio between the lowpass filter phi and the lowest-frequency wavelet. Default is 2 if Q is 1 and otherwise 1.

    Returns
    --------
    J: int
       Number of logarithmically spaced wavelets

    """
    if B is None: B = Q
    if phibwratio is None:
        if type(Q) == np.ndarray:
            phibwratio=1.+(Q==1).astype(int)
        else:
            phibwratio=1+int(Q==1)
    
    if type(Q) == np.ndarray:
        return 1+ (np.log2(T/(4*B/phibwratio))*Q+.5).astype(int)
    else:
        return 1+ int(np.log2(T/(4*B/phibwratio))*Q+.5)

def morletfbank(T,Q=1,
                B=None,
                J=None,
                P=None,
                xipsi=None,
                sigmapsi=None,
                phibwratio=None, 
                sigmaphi=None,
                phidirac=False,
                do_morlet=False,
                threshold=EPSILON,
                ):
    """
    No truncation just the filter matrices where they are arranged in frequency

    Parameters:
    ============
    T: int
        Length of the signal
    Q: int
    B: int
    P: int
    J: int
    
    """
    
    # we fill in the default options if none were given
    if B is None:
        B = Q
    if J is None:
        J = TtoJ(T)
    if xipsi is None:
        xipsi=np.pi/2*(2**(-1/Q)+1)
    if sigmapsi is None:
        sigmapsi = .5*SIGMA0/(1-2**(-1/B))
    if phibwratio is None:
        if type(Q) == np.ndarray:
            phibwratio = 1+(Q==1).astype(int)
        else:
            phibwratio = 1+int(Q==1)
    if sigmaphi is None:
        sigmaphi=sigmapsi/phibwratio
    if J is None:
        J=TtoJ(T,Q=Q,B=B,phibwratio=phibwratio)
    if P is None:
        if type(Q) == np.ndarray:
            P=(.5 + (2**(-1/Q) -1/4*SIGMA0/sigmaphi)/(1-2**(-1/Q))).astype(int)
        else:
            P=int(.5 + (2**(-1/Q) -1/4*SIGMA0/sigmaphi)/(1-2**(-1/Q)))




    # we reflect the signal so that way we can deal with boundary
    # effects gracefully and we also want to make sure that we
    # are using a power of 2 for the length
    N = 2**(int(np.ceil(np.log2(2*T))))
    psicenter, bwpsi, bwphi = morletfreq(xipsi,sigmapsi,sigmaphi,
                                     J,Q,P,phidirac)


    psisigma = SIGMA0*np.pi/2/bwpsi
    phisigma = SIGMA0*np.pi/2/bwphi


    # compute the filter normalization
    # need the sum of squares to be less than one
    S = np.zeros(N)
    psifilters = np.zeros((J+P,N),dtype=float)
    for j1 in xrange(J+P):
        temp = gabor(N,psicenter[j1],psisigma[j1])
        if do_morlet:
            temp = morletify(temp,psisigma[j1])

        
        # only normalize over the first J filters
        if j1 < J:
            S += np.abs(temp)**2
        psifilters[j1] = temp

    # normalization for the psi filter
    psifilters *= np.sqrt(2/np.max(S))


    # compute the low-pass filter
    if not phidirac:
        phifilter =  gabor(N,0,phisigma)
        phitype='gabor'
    else:
        phifilter = np.ones(N)
        phitype='ones'

    
    return FBANK_T(Q=Q,
                    B=B,
                    J=J,
                    P=P,
                    phidirac=phidirac,
                    psifilters=psifilters,
                    psitype='morlet' if do_morlet else 'gabor',
                    xipsi=xipsi,
                    sigmapsi=sigmapsi,
                    phifilter=phifilter,
                    phitype=phitype,
                    sigmaphi=sigmaphi)





def morletfbank_truncate(T,Q=1,
                B=None,
                J=None,
                P=None,
                xipsi=None,
                sigmapsi=None,
                phibwratio=None, 
                sigmaphi=None,
                phidirac=False,
                do_morlet=False,
                threshold=EPSILON,
                ):
    """
    Parameters:
    ============
    T: int
        Length of the signal
    Q: int
    B: int
    P: int
    J: int
    
    """
    
    # we fill in the default options if none were given
    if B is None:
        B = Q
    if J is None:
        J = TtoJ(T)
    if xipsi is None:
        xipsi=np.pi/2*(2**(-1/Q)+1)
    if sigmapsi is None:
        sigmapsi = .5*SIGMA0/(1-2**(-1/B))
    if phibwratio is None:
        phibwratio = 1+int(Q==1)
    if sigmaphi is None:
        sigmaphi=sigmapsi/phibwratio
    if J is None:
        J=TtoJ(T,Q=Q,B=B,phibwratio=phibwratio)
    if P is None:
        P=int(.5 + (2**(-1/Q) -1/4*SIGMA0/sigmaphi)/(1-2**(-1/Q)))

    # we reflect the signal so that way we can deal with boundary
    # effects gracefully and we also want to make sure that we
    # are using a power of 2 for the length
    N = 2**(int(np.ceil(np.log2(2*T))))
    psixi, bwpsi, bwphi = morletfreq(xipsi,sigmapsi,sigmaphi,
                                     J,Q,P,phidirac)

    psisigma = SIGMA0*np.pi/2/bwpsi
    phisigma = SIGMA0*np.pi/2/bwphi

    # compute the filter normalization
    # need the sum of squares to be less than one
    S = np.zeros((N,1))
    psisupports = np.zeros(J,dtype=int)
    psimids = np.zeros(J,dtype=int)
    for j1 in xrange(J):
        temp = gabor(N,psicenter[j1],psisigma[j1])
        if do_morlet:
            temp = morletify(temp,psisigma[j1])

        temp = np.abs(temp)
        S += temp**2
        psisupports[j1],psimids[j1] = filter_support_mid(temp,threshold)

    # normalization for the psi filter
    psiampl= np.sqrt(2/np.max(S))
    psifilters = np.zeros(psisupports.sum())
    psifilterindices= psisupports.cumsum()
    psifilterstarts = np.zeros(J)
    
    for j1 in xrange(J):
        temp = gabor(N,psicenter[j1],psisigma[j1])
        if do_morlet:
            temp = morletify(temp,psisigma[j1])
            psitype='morlet'
        else:
            psitype='gabor'

        psifilters[
            psifilterindices[j1]:
            psifilterindices[j1]
            +temp.shape[0]],psifilterstarts[j1] = truncate_filter(
                psiampl*temp,supportlen=psisupports[j1],mid=psimids[j1])

    # compute the low-pass filter
    if phidirac:
        phifilter =  gabor(N,0,phisigma)
        phitype='gabor'
    else:
        phifilter = np.ones(N)
        phitype='ones'

    phifilter,phifilterstart = truncate_filter(phifilter,threshold=threshold)
    
    return FBANK_T(psifilters=psifilters, psifilterindices=psifilterindices, psifilterstarts=psifilterstarts, phifilter=phifilter, phifilterstart=phifilterstart)

        


def truncate_filter(filter_f,threshold=None,supportlen=None,mid=None):
    """
    Parameters:
    ------------
    filter_f: np.ndarray[ndim=1]
        Fourier transform of the filter
    
    threshold: float
        threshold to truncate the filter by
    
    Out:
    -----
    filter: np.ndarray
        The truncated filter
    filterstart: int
        Index where the filter starts
    """
    n=len(filter_f)
    absfilter=np.abs(filter_f)
    maxval = np.max(absfilter)
    maxidx = np.argmax(absfilter)
    # center the maximum point of the filter
    recentershift = n/2-maxidx-1
    filter_f = np.roll(filter_f,recentershift)
    
    if mid is None or supportlen is None:
        absfilter = np.roll(absfilter,recentershift)
        supportlen,mid = filter_support_mid(absfilter,threshold)
    else:
        mid += recentershift

    # will be an integer since supportlen is a multiple of 2
    # since n is a multiple of 2
    startidx = max(int(mid - supportlen/2+.5),0)
    endidx=startidx+supportlen
    return filter_f[startidx:endidx],(startidx-recentershift) % n
    
    
def filter_support_mid(absf,threshold):
    """
    get the support of the filter as a power of 2
    assume that the input filter is the magnitude of values
    not the raw filter coefficients


    Output:
    -------
    supportlen: int
        length of the support
    mid: int
        .5-midpoint of the support (we assume an even length filter so that the mid point is between two values)
    """
    n = absf.shape[0]
    abovethreshidx = np.where(absf > threshold*absf.max())[0]
    supportlen = abovethreshidx[-1] - abovethreshidx[0] + 1
    return n/2**(int(np.log2(n/supportlen))), int((abovethreshidx[0] + abovethreshidx[-1])/2+.5)

def gabor(N, xi, sigma, radius=1):
    """
    Parameters:
    ----------
    invsigma: float
        1/sigma
    """
    return np.exp( - ((np.lib.stride_tricks.as_strided(
        np.arange(N),
        shape=(N,2*radius+2),
        strides=(8,0)) - N*np.lib.stride_tricks.as_strided(
            np.arange(2*radius+2)-radius,
            shape=(N,2*radius+2),
            strides=(0,8)))
            /N*2*np.pi-xi)**2*sigma**2/2).sum(1)
    
def morletify(f,sigma):
    """

    Returns
    -------
    
    """
    return f-f[0]*gabor(len(f),0,sigma)

def filter_freq(fbank):
    """
    If fbank is None then filtertype should be a string
    fbank should have type FBANK_T (at top of file)
    this is a wrapper function to take in the various kinds of
    filters and output their center frequencies and bandwidths

    Returns
    -------
    xipsi, bwpsi, bwphi
    """
    if fbank.psitype in ['morlet','gabor']:
        return morletfreq(fbank.xipsi,fbank.sigmapsi,fbank.sigmaphi,fbank.J,
                   fbank.Q,fbank.P,fbank.phidirac)

def morletfreq(xipsiconst,sigmapsiconst,sigmaphi,J,Q,P,phidirac):
    """
    Gives the frequencies and bandwidths for the filters

    Parameters:
    ===========
    xipsiconst: float
    sigmapsiconst: float
    sigmaphi: float
    J: int
    Q: int
    P: int

    Returns:
    ========
    xipsi:
    bwpsi:
    bwphi:

    """
    # logarithmically spaced band-pass filters
    xipsi = np.zeros(J+P)
    sigmapsi = np.zeros(J+P)
    xipsi[:J] = xipsiconst * 2**(np.arange(0,-J,-1)/Q)
    sigmapsi[:J] = sigmapsiconst * 2**(np.arange(J)/Q)
    
    # linearly-spaced band-pass filters for the rest of the spectrum
    if P  > 0:
        step = np.pi * 2**(-J/Q) * (1 - .25*SIGMA0/sigmaphi*2**(
            1/Q))/P
        xipsi[J:] = xipsiconst * 2**((1-J)/Q) - step*(1+np.arange(P))
        sigmapsi[J:] = sigmapsiconst *2**((J-1)/Q)


    # band-pass filter computation
    sigmaphi = sigmaphi * 2**((J-1)/Q)

    bwpsi =np.pi/2 * SIGMA0/sigmapsi
    if not phidirac:
        bwphi = np.pi/2 * SIGMA0/sigmaphi;
    else:
        bwphi = 2 * np.pi;
    
    return xipsi, bwpsi, bwphi
    

def hermite_window(taper_length,
                   order,
                   half_time_support,pad_windows = 0):
    """
    Compute the hermite tapers up to second derivatives of the hermite
    window
    """
    dt = (2.*half_time_support)/(taper_length-1)
    tt = np.linspace(-half_time_support,
                     half_time_support,
                     taper_length)
    g = np.exp(-tt**2/2)
    
    P = np.ones((order+2,taper_length))
    HTemp = np.ones((order+2,taper_length))
    DH = np.ones((order+1,taper_length))
    DDH = np.ones((order,taper_length))
    P[1] = 2 *tt
    
    for k in xrange(2,order+2):
        P[k] = 2*tt*P[k-1] - 2*(k-1)*P[k-2]
    
    for k in xrange(order+2):
        HTemp[k] = P[k] * g/np.sqrt(np.sqrt(np.pi)*2**k*gamma(k+1))*np.sqrt(dt)

    for k in xrange(order+1):
        DH[k] = (tt*HTemp[k] - np.sqrt(2.*(k+1)) * HTemp[k+1])*dt

    # second derivative in order to do Hessian computations
    # for the reassignment subspace method
    for k in xrange(order):
        DDH[k] = (HTemp[k]*dt + tt*DH[k] - np.sqrt(2.*(k+1)) * DH[k+1])*dt

    if pad_windows > 0:
        HTemp = pad_multitaper_filters(HTemp)
        DH = pad_multitaper_filters(DH)
        DDH = pad_multitaper_filters(DDH)
        tt = pad_multitaper_filters(tt)

    return HTemp[:order], DH,DDH,tt

def pad_multitaper_filters(multitaper_filters,pad_length=1):
    """
    Do a symmetric padding
    """
    if multitaper_filters.ndim == 2:
        padded_multitaper_filters = np.zeros((
            multitaper_filters.shape[0],
            multitaper_filters.shape[1] + pad_length))
        padded_multitaper_filters[:,:-pad_length] = multitaper_filters
        padded_multitaper_filters[:,-pad_length:] = multitaper_filters[:,:-pad_length-1:-1]
    else:
        padded_multitaper_filters = np.ones(multitaper_filters.shape[0] + pad_length)
        padded_multitaper_filters[-pad_length:] = multitaper_filters[-pad_length:][::-1]
        padded_multitaper_filters[:-pad_length] = multitaper_filters
    return padded_multitaper_filters

def get_gauss_filter(x,y,sigma):
    return np.exp(-((np.mgrid[:x,:y]-x/2)**2).sum(0)/(2*sigma))

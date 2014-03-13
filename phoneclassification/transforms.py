from __future__ import division
import numpy as np
import filterbank as fb
from scipy.ndimage.filters import maximum_filter, convolve


def preemphasis(x,preemph=.95):
    return np.append( (x[1:] - .95*x[:-1])[::-1],x[0])[::-1]

def process_wav(x):
    return x.astype(float)/2**15

def spec_freq_avg(x,fbank1,fbank2,oversampling,return_midpoints=False):
    """
    wavelet-averaged spectrogram--an approximation
    to MFCCs using wavelets rather than Mel-scale filters
    """
    # we just want the bandwidth of the lowpass filter
    N = x.size
    supp_mult=4
    bwphi2 = fb.filter_freq(fbank2)[2]
    Nfilt = fbank1.psifilters.shape[1]
    N1 = 2**int(.5+(np.log2(2*np.pi/bwphi2)))

    fs = np.abs(fbank1.psifilters[:,::Nfilt/(N1*supp_mult)])
    window = np.fft.ifft(fbank2.phifilter)
    window = np.hstack((window[Nfilt-N1*supp_mult/2:],
                    window[:N1*supp_mult/2]))

    # number of output frames
    nframes = int(.5 + N/N1*2**oversampling)

    # get the indices for each sample
    indices = (np.arange(N1*supp_mult,dtype=int)-int((N1*supp_mult)/2))[np.newaxis, : ] + int(N1/2**oversampling)*np.arange(nframes,dtype=int)[:,np.newaxis]
    # symmetrize the front
    indices *= (2*(indices > 0)-1)
    # symmetrize the tail
    tail_indices = indices > N-1
    indices[tail_indices] = N-1 - (indices[tail_indices] - N+1)

    frames = np.abs(np.fft.fft(x[indices] * window))
    if return_midpoints:
        return np.dot(fs,frames.T), indices[:,int(indices.shape[1]/2+.5)]

    return np.dot(fs,frames.T)

def spectrogram(x,sample_rate,freq_cutoff,winsize,nfft,oversampling,
                                    h,return_midpoints=False):
    """
    Compute a simple spectrogram using a given window
    """
    N = len(x)
    nframes = int(.5 + N/winsize*2**oversampling)

    greater_than_winlength = winsize*np.ones((nframes,nfft)) > np.arange(nfft)

    indices = (np.arange(nfft,dtype=int)-int(nfft/2))[np.newaxis, : ] + int(nfft/2**oversampling)*np.arange(nframes,dtype=int)[:,np.newaxis]
    
    indices *= (2*(indices > 0)-1)
    # symmetrize the tail
    tail_indices = indices > N-1
    indices[tail_indices] = N-1 - (indices[tail_indices] - N+1)

    """
    """
    abs_avg = np.zeros(indices.shape)

    for i in xrange(5):
        f = np.fft.fft((x[indices]*greater_than_winlength) * zero_pad_window(h[i],nfft-winsize),nfft)
        f_mv = f * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)
        abs_avg += np.abs(f_mv)


    abs_avg/=5




    cutoff_idx = int(freq_cutoff/sample_rate * winsize)

    if return_midpoints:
        return abs_avg[:,:cutoff_idx], indices[:,int(indices.shape[1]/2+.5)]

    return abs_avg[:,:cutoff_idx]



def zero_pad_window(w,n_pad):
    w0 = np.zeros(w.shape[0] + n_pad)
    w0[:w.shape[0]] = w
    return w0

def spectrogram_magnitude_gradients(x,sample_rate,freq_cutoff,winsize,nfft,oversampling,
                                    h,dh,tt,return_midpoints=False):
    """
    Returns the spectrogram as well as magnitude gradients
    all of these are multitaper
    """
    N = len(x)
    nframes = int(.5 + N/winsize*2**oversampling)

    greater_than_winlength = winsize*np.ones((nframes,nfft)) > np.arange(nfft)

    indices = (np.arange(nfft,dtype=int)-int(nfft/2))[np.newaxis, : ] + int(nfft/2**oversampling)*np.arange(nframes,dtype=int)[:,np.newaxis]
    
    indices *= (2*(indices > 0)-1)
    # symmetrize the tail
    tail_indices = indices > N-1
    indices[tail_indices] = N-1 - (indices[tail_indices] - N+1)

    """
    """
    abs_avg = np.zeros(indices.shape)
    avg_dM_dt = np.zeros(indices.shape)
    avg_dM_dw = np.zeros(indices.shape)

    for i in xrange(5):
        f = np.fft.fft((x[indices]*greater_than_winlength) * zero_pad_window(h[i],nfft-winsize),nfft)
        f_mv = f * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)
        abs_avg += np.abs(f_mv)
        df = np.fft.fft((x[indices]*greater_than_winlength) * zero_pad_window(dh[i],nfft-winsize),nfft)
        df_mv = df * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)
        tf = np.fft.fft((x[indices]*greater_than_winlength)*zero_pad_window(tt*h[i],nfft-winsize) ,nfft)
        tf_mv = tf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)

        abs_f_mv = np.abs(f_mv)**2
        dM_dt = np.real((df_mv * f_mv.conj())/abs_f_mv)
        dM_dw = - np.imag((tf_mv * f_mv.conj())/abs_f_mv)
        avg_dM_dt += dM_dt
        avg_dM_dw += dM_dw

    abs_avg/=5
    avg_dM_dt /= 5
    avg_dM_dw /= 5




    cutoff_idx = int(freq_cutoff/sample_rate * winsize)

    if return_midpoints:
        return abs_avg[:,:cutoff_idx], avg_dM_dt[:,:cutoff_idx], avg_dM_dw[:,:cutoff_idx], indices[:,int(indices.shape[1]/2+.5)]

    return abs_avg[:,:cutoff_idx], avg_dM_dt[:,:cutoff_idx], avg_dM_dw[:,:cutoff_idx]


def spectrogram_reassignment(x,sample_rate,freq_cutoff,winsize,nfft,oversampling,
                                    h,dh,tt,return_midpoints=False):
    """
    Returns the spectrogram as well as magnitude gradients
    all of these are multitaper
    """
    N = len(x)
    nframes = int(.5 + N/winsize*2**oversampling)

    greater_than_winlength = winsize*np.ones((nframes,nfft)) > np.arange(nfft)

    indices = (np.arange(nfft,dtype=int)-int(nfft/2))[np.newaxis, : ] + int(nfft/2**oversampling)*np.arange(nframes,dtype=int)[:,np.newaxis]
    
    indices *= (2*(indices > 0)-1)
    # symmetrize the tail
    tail_indices = indices > N-1
    indices[tail_indices] = N-1 - (indices[tail_indices] - N+1)

    """
    """
    abs_avg = np.zeros(indices.shape)
    avg_dM_dt = np.zeros(indices.shape)
    avg_dM_dw = np.zeros(indices.shape)

    for i in xrange(5):
        f = np.fft.fft((x[indices]*greater_than_winlength) * zero_pad_window(h[i],nfft-winsize),nfft)
        f_mv = f * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)
        abs_avg += np.abs(f_mv)
        df = np.fft.fft((x[indices]*greater_than_winlength) * zero_pad_window(dh[i],nfft-winsize),nfft)
        df_mv = df * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)
        tf = np.fft.fft((x[indices]*greater_than_winlength)*zero_pad_window(tt*h[i],nfft-winsize) ,nfft)
        tf_mv = tf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)

        abs_f_mv = np.abs(f_mv)**2
        t_hat = np.real((df_mv * f_mv.conj())/abs_f_mv)
        dM_dw = - np.imag((tf_mv * f_mv.conj())/abs_f_mv)
        avg_dM_dt += dM_dt
        avg_dM_dw += dM_dw

    abs_avg/=5
    avg_dM_dt /= 5
    avg_dM_dw /= 5




    cutoff_idx = int(freq_cutoff/sample_rate * winsize)

    if return_midpoints:
        return abs_avg[:,:cutoff_idx], avg_dM_dt[:,:cutoff_idx], avg_dM_dw[:,:cutoff_idx], indices[:,int(indices.shape[1]/2+.5)]

    return abs_avg[:,:cutoff_idx], avg_dM_dt[:,:cutoff_idx], avg_dM_dw[:,:cutoff_idx]


def binary_phase_features(x,sample_rate,freq_cutoff,winsize,nfft,oversampling,h,dh,tt,gfilter,gsigma,fthresh,othresh,tthresh=4.125,spread_length=3,return_midpoints=True,return_frequencies=False):
    """
    We assume x has already been preemphasized, this just recovers the frequency components of the signal.
    """
    N = len(x)
    nframes = int(.5 + N/winsize*2**oversampling)
    greater_than_winlength = winsize*np.ones((nframes,nfft)) > np.arange(nfft)

    indices = (np.arange(nfft,dtype=int)-int(nfft/2))[np.newaxis, : ] + int(nfft/2**oversampling)*np.arange(nframes,dtype=int)[:,np.newaxis]

    indices *= (2*(indices > 0)-1)
    # symmetrize the tail
    tail_indices = indices > N-1
    indices[tail_indices] = N-1 - (indices[tail_indices] - N+1)


    gt,gf = gfilter.shape
    gdwfilter = -(np.mgrid[:gt,:gf]-3.5)[1]/gsigma * gfilter
    gdtfilter = -(np.mgrid[:gt,:gf]-3.5)[0]/gsigma * gfilter

    """
    """
    abs_avg = np.zeros(indices.shape)
    avg_dphi_dt = np.zeros(indices.shape)
    avg_dphi_dw = np.zeros(indices.shape)

    for i in xrange(5):
        f = np.fft.fft((x[indices]*greater_than_winlength) * h[i])
        f_mv = f * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)
        abs_avg += np.abs(f_mv)
        df = np.fft.fft((x[indices]*greater_than_winlength) * dh[i])
        df_mv = df * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)
        tf = np.fft.fft((x[indices]*greater_than_winlength) * (tt*h[i]))
        tf_mv = tf * np.exp(2j*np.pi*np.outer(indices[:,0],np.arange(indices.shape[1]))/nfft)

        abs_f_mv = np.abs(f_mv)**2
        dphi_dt = np.imag((df_mv * f_mv.conj())/abs_f_mv)
        dphi_dw = - np.real((tf_mv * f_mv.conj())/abs_f_mv)
        avg_dphi_dt += dphi_dt
        avg_dphi_dw += dphi_dw

    abs_avg/=5
    avg_dphi_dt /= 5
    avg_dphi_dw /= 5

    filter_d2phi_dwdt = convolve(avg_dphi_dt,gdwfilter)
    filter_d2phi_dt2 = convolve(avg_dphi_dt,gdtfilter)
    filter_d2phi_dtdw = convolve(avg_dphi_dw,gdtfilter)


    F = np.zeros(filter_d2phi_dwdt.shape + (5,),dtype=np.uint8)
    S = np.sign(filter_d2phi_dt2)*( np.abs(filter_d2phi_dt2) > othresh)
    F[:,:,0] = filter_d2phi_dwdt > fthresh
    F[:,:,1] = F[:,:,0] * maximum_filter(S == 1,footprint=np.eye(spread_length),mode='constant')
    F[:,:,2] = F[:,:,0] * maximum_filter(S == 0,footprint=np.ones((1,spread_length)),mode='constant')
    F[:,:,3] = F[:,:,0] * maximum_filter(S == -1,footprint=np.eye(spread_length)[::-1],mode='constant')
    F[:,:,4] = filter_d2phi_dtdw > tthresh

    cutoff_idx = int(freq_cutoff/sample_rate * winsize)
    frequencies = np.arange(cutoff_idx) * sample_rate/winsize

    if return_frequencies:
        return abs_avg[:,:cutoff_idx], F[:,:cutoff_idx], indices[:,int(indices.shape[1]/2+.5)], frequencies
    if return_midpoints:
        return abs_avg[:,:cutoff_idx], F[:,:cutoff_idx], indices[:,int(indices.shape[1]/2+.5)]

    return abs_avg[:,:cutoff_idx], F[:,:cutoff_idx]

    
def wavelet_scat(x,fbank,
            oversampling=1,
            psi_mask = None,
            x_resolution = 0):
    """
    """

    if psi_mask is None:
        psi_mask = np.ones(filterbank.psifilters.shape[0],dtype=bool)
    N = x.shape[0]

    assert x.ndim == 1
    bwpsi, bwphi = fb.filter_freq(fbank)[1:]
    N_padded = fbank.psifilters.shape[1]
    n_psi = fbank.psifilters.shape[0]
    x = pad_signal(x,N_padded)
    xf = np.fft.fft(x)
    # compute the downsampling factor as a power of 2
    downsample = max(int(.5 + np.log2(2*np.pi/bwphi)) - j0  - oversampling,0)

    x_phi = unpad_signal(np.real(conv_sub(xf, fbank.phifilter, downsample)),
                         downsample,N)
    x_psi = np.zeros((n_psi,N))
    for i,psifilter in enumerate(fbank.psifilters):
        downsample = max(int(.5 + np.log2(np.pi/bwsi[i])) - j0  - max(1,oversampling),0)
        x_psi[i] = unpad_signal(conv_sub(xf,psifilter,downsample),downsample, N)
    
    return x_phi,x_psi
        
    

    
    
def pad_signal(x, Npad):
    """
    The input signal x is padded to be of length Npad
    using a symmetric boundary so that any discontinuities are as
    far from the signal as possible
    the signal is to the far left of the output signal
    it allows for convolutions to be calculated with lower
    error

    No handling for complex input yet

    We assume that Npad is no greater than twice the signal length
    """
    
    Norig = x.shape[0]
    y = np.zeros(Norig*2)
    y[:Norig] = x
    y[Norig:] = x[::-1]
    midpoint = Norig/2
    rm_nsamples = y.shape[0] - Npad
    # we want to remove a chunk with the property that
    # start:end is removed and Npad - start == y.shape[0] - end
    # deriving further we get
    # start == Npad - y.shape[0] + end
    start = Norig + int(midpoint - rm_nsamples/2)
    end = start + rm_nsamples
    # 
    y[start:Npad] = y[end:]
    return y[:Npad]
    
def unpad_signal(x, resolution, Norig):
    """
    The signal x is assumed to be a padded version at resolution
    2**resolution of a signal of length Norig

    We unpad it in this function
    """

    return x[:int(Norig/2**resolution)]

def conv_sub(xf, filt, ds):
    """
    TODO: make work for a truncated filter to make processing
    much faster
    Parameters
    ----------
    xf: array-like
       Fourier transform of the signal assume that it has even
       length (this can be produced via padding from an earlier step)
    filt: array-like
       Filter to convolve the signal with
    ds: int
       downsampling factor
    """
    N = xf.shape[0]
    # modified filter has the correct length and corrects for the
    # middle point
    mod_filter = np.zeros(N)
    mod_filter[:int(N/2)] = filt[:int(N/2)]
    mod_filter[int(N/2)] = (filt[int(N/2)] + filt[-int(N/2)])/2
    mod_filter[int(N/2)+1:] = filt[1-int(N/2):]
    yf = mod_filter * xf

    # compute the downsampling factor
    downsampj = ds + np.log2(yf.shape[0]/N)

    if downsampj > 0:
        yf_ds = yf.reshape(int(N/2**downsampj),2**downsampj).sum(1)
    elif downsampj < 0:
        yf_ds = np.zeros(2**(-downsampj)*yf.shape[0] )
        yf_ds[:yf.shape[0]] = yf
    else:
        yf_ds = yf

    return np.fft.ifft(yf_ds)/2**(int(ds/2))

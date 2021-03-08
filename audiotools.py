import numpy as np
import matplotlib.pyplot as plt
import warnings
import librosa
    
def save_audio(wavfilename, x, sr):
    """
    Save audio to a file

    Parameters
    ----------
    wavfilename: string
        Filename to which to save file
    x: ndarray(N, 2)
        Stereo audio to save
    sr: int
        Sample rate of audio to save
    """
    from scipy.io import wavfile
    wavfile.write(wavfilename, sr, x)

def get_mfcc_mod(x, sr, hop_length, n_mfcc=120, drop=20, n_fft = 2048):
    """
    Compute the mfcc_mod features, as described in Gadermaier 2019

    Parameters
    ----------
    x: ndarray(N)
        Audio samples
    sr: int
        Sample rate
    hop_length: int
        Hop size between windows
    n_mfcc: int
        Number of mfcc coefficients to compute
    drop: int
        Index under which to ignore coefficients
    n_fft: int
        Number of fft points to use in each window
    
    Returns
    -------
    X: ndarray(n_win, n_mfcc-drop)
        The mfcc-mod features
    """
    X = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length, n_mfcc = n_mfcc, n_fft=n_fft, htk=True)
    X = X[drop::, :].T
    return X

def timemap_stretch(x, sr, path, hop_length=32, n_fft = 4096):
    """
    Stretch audio x so that it aligns with another
    audio clip, according to a warping path

    Parameters
    ----------
    x: ndarray(N)
        An array of audio samples
    sr: int
        Sample rate
    path: ndarray(K, 2)
        Warping path.  Indices of x are in first row
    hop_length: int
        Hop length to use in the phase vocoder
    n_fft: int
        Number of fft samples to use in the phase vocoder
    """
    # Break down into regions of constant slope
    xdiff = path[1::, 0] - path[0:-1, 0]
    ydiff = path[1::, 1] - path[0:-1, 1]
    xdiff = xdiff[1::] - xdiff[0:-1]
    ydiff = ydiff[1::] - ydiff[0:-1]
    diff = xdiff + ydiff
    ret = np.array([])
    i1 = 0
    while i1 < len(diff):
        i2 = i1+1
        while i2 < len(diff) and diff[i2] == 0:
            i2 += 1
        while i2 < len(diff) and path[i2, 0] - path[i1, 0] < n_fft:
            i2 += 1
        if i2 >= len(diff):
            break
        fac = (path[i2, 1]-path[i1, 1])/(path[i2, 0]-path[i1, 0])
        if fac > 0:
            fac = 1/fac
            xi = x[path[i1, 0]:path[i2, 0]+1]
            D = librosa.stft(xi, n_fft = n_fft, hop_length=hop_length)
            DNew = librosa.phase_vocoder(D, fac, hop_length=hop_length)
            xifast = librosa.istft(DNew, hop_length=hop_length)
            ret = np.concatenate((ret, xifast))
        i1 = i2
    return ret


def stretch_audio(x1, x2, sr, path, hop_length):
    """
    Wrap around pyrubberband to warp one audio stream
    to another, according to some warping path

    Parameters
    ----------
    x1: ndarray(M)
        First audio stream
    x2: ndarray(N)
        Second audio stream
    sr: int
        Sample rate
    path: ndarray(P, 2)
        Warping path, in units of windows
    hop_length: int
        The hop length between windows
    
    Returns
    -------
    x3: ndarray(N, 2)
        The synchronized audio.  x2 is in the right channel,
        and x1 stretched to x2 is in the left channel
    """
    from alignmenttools import refine_warping_path
    print("Stretching...")
    path_final = [(row[0], row[1]) for row in path if row[0] < x1.size and row[1] < x2.size]
    path_final.append((x1.size, x2.size))
    path_final = hop_length*np.array(path_final, dtype=int)
    x3 = np.zeros((x2.size, 2))
    x3[:, 1] = x2
    x1_stretch = timemap_stretch(x1, sr, path_final)
    print("x1.shape = ", x1.shape)
    print("x2.shape = ", x2.shape)
    print("x1_stretch.shape = ", x1_stretch.shape)
    x1_stretch = x1_stretch[0:min(x1_stretch.size, x3.shape[0])]
    x3 = x3[0:min(x3.shape[0], x1_stretch.size), :]
    x3[:, 0] = x1_stretch
    return x3
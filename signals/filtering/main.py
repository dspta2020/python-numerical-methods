import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import ShortTimeFFT

from FilterGen import FilterGen
from WaveformGen import WaveformGen


def PlotSpectrogram(ax, window_size, waveform, fs, nFFT):

    window_size = window_size
    stft = ShortTimeFFT(np.hamming(window_size), hop=window_size//2, fs=fs, fft_mode='centered', mfft=nFFT)

    # get the plotted components
    mag = 20*np.log10(stft.spectrogram(waveform.flatten()))
    freqs = np.arange(-stft.f_pts/2, stft.f_pts/2) * (fs/stft.f_pts)
    time_vec = np.linspace(0, len(waveform)/fs, int(mag.shape[1]))

    ax.pcolormesh(time_vec, freqs*1e-3, mag, shading='auto')
    ax.set_ylabel('Frequency [MHz]')
    ax.set_xlabel('Time [msec]')


if __name__ == "__main__":

    # some code to mess with makeing filters
    fs = 10e3 # samples per sec
    dur = 5 # sec
    t = np.arange(0, fs*dur) / fs

    wf_gen = WaveformGen(fs, dur)

    noise = wf_gen.make_white_gaussian_noise(5)
    tone1 = wf_gen.make_tone(3e3,7)
    pulse1 = wf_gen.make_windowed_tone(-2.5e3, 2, 4,7)
    bits1 = wf_gen.make_random_ofdm_bits(128, 5, 1e3, 1)
    lfm1 = wf_gen.make_windowed_lfm_start_stop(-1e3, -4e3, 1, 4,7)

    data_complex = noise + bits1 + tone1 + pulse1 + lfm1

    fig, ax = plt.subplots()
    ax.plot(t, abs(data_complex))

    fig2, ax2 = plt.subplots()
    PlotSpectrogram(ax2, 128, data_complex, fs, 8192)

    plt.show()



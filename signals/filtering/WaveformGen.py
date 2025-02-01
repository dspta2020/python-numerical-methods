import numpy as np
import matplotlib.pyplot as plt

class WaveformGen:

    def __init__(self, fs, dur):
        
        self.fs = fs 
        self.dur = dur 
        self.num_samples = int(self.dur * self.fs)

    def make_lfm_by_startstop(self, f0, f1, A=1):

        bw = f1 - f0
        slope = bw / self.dur
        t = np.arange(self.num_samples) / self.fs

        return A * np.exp(1j * 2*np.pi * (1/2*slope*t**2 + f0*t))


    def make_lfm_by_slope(self, f0, slope, A=1):
        
        t = np.arange(self.num_samples) / self.fs

        return A * np.exp(1j * 2*np.pi * (1/2*slope*t**2 + f0*t))
    

    def make_windowed_lfm_start_stop(self, f0, f1, t0, t1, A=1):

        bw = f1 - f0
        slope = bw / (t1 - t0)
        t0_samples = round(t0 * self.fs)
        t1_samples = round(t1 * self.fs)

        
        t_samples = np.arange(t1_samples - t0_samples) / self.fs

        waveform = A * np.exp(1j * 2*np.pi * (1/2*slope*t_samples**2 + f0*t_samples))

        full_vector = np.zeros(self.num_samples, dtype=complex)
        full_vector[t0_samples:t1_samples] = waveform

        return full_vector


    def make_tone(self, f0, A=1):
        
        t = np.arange(self.num_samples) / self.fs
        
        return A * np.exp(1j * 2 * np.pi * f0 * t)


    def make_windowed_tone(self, f0, t0, t1, A=1):
        # pass t0, t1 in seconds and then will round to the nearest sample 
        # if t does not fall into a clean time bin 

        t0_samples = int(np.round(t0 * self.fs))
        t1_samples = int(np.round(t1 * self.fs))
        t_samples_windowed = np.arange(t0_samples, t1_samples)

        full_vector = np.zeros(self.num_samples, dtype=complex)

        waveform = A * np.exp(1j * 2 * np.pi * f0 * t_samples_windowed/self.fs)
        full_vector[t0_samples:t1_samples] = waveform
        
        return full_vector

    
    def make_white_gaussian_noise(self, A=1):

        return A * ((np.random.randn(self.num_samples) + np.random.randn(self.num_samples) * 1j) / np.sqrt(2))
    

    def make_random_ofdm_bits(self, num_subcarriers, carrier_spacing, f0, t0, A=1):

        max_nIFFT = self.fs * self.dur
        nIFFT = self.fs / carrier_spacing

        if nIFFT > max_nIFFT:
            print('Incompatible nIFFT. (nIFFT > max_nIFFT)')
        elif nIFFT < num_subcarriers:
            print('Incompatible nIFFT. (nIFFT < num_subcarriers)')

        # setup the ifft 
        bits = np.random.randint(0, 2, num_subcarriers) * 2 - 1
        zeros_to_pad = nIFFT - num_subcarriers
        half_zeros_to_pad = int(zeros_to_pad // 2)
        bits_padded = np.pad(bits, pad_width=(half_zeros_to_pad, half_zeros_to_pad), mode='constant', constant_values=0)

        # make waveform
        t_windowed = np.arange(len(bits_padded)) / self.fs
        waveform = np.fft.ifft(np.fft.ifftshift(bits_padded, 0), axis=0) * np.exp(1j * 2 * np.pi * f0 * t_windowed) * A * (nIFFT) * np.hamming(len(t_windowed))

        full_vector = np.zeros(self.num_samples, dtype=complex) 
        t0_samples = round(t0 * self.fs)
        full_vector[t0_samples:t0_samples+len(waveform)] = waveform

        return full_vector  
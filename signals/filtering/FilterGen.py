import numpy as np 

class FilterGen:

    def __init__(self, cutoff, sampling_rate, order):
        
        self.fc = cutoff
        self.fs = sampling_rate
        self.N = order

        self.fc_normalized = self.fc / self.fs
        self.midpoint = (self.N - 1) / 2

    def make_low_pass_filter(self):
        
        # filter response
        h = np.zeros((self.N, 1))
        
        for n in np.arange(self.N):
            if n != self.midpoint:
                # sinc function filter formula 
                h[n] = np.sin(2 * np.pi * self.fc_normalized * (n - self.midpoint)) / (np.pi * (n - self.midpoint))
            else:
                h[n] = 2 * self.fc_normalized
        
        return h
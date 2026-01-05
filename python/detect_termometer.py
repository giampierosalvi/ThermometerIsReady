from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
import numpy as np

# frequency of termometer signal
freq = 6080

nfft = 2**16

samplerate, data = wavfile.read('Termometro.wav')

fft_bin = int(float(freq)*float(nfft)/float(samplerate))

print(f"fft bin = {fft_bin}")

print(f"number of channels = {data.shape[1]}")
length = data.shape[0] / samplerate
print(f"length = {length}s")
print('using channel 0')

spgrm = spectrogram(data[:,0], fs=samplerate, nfft=nfft, nperseg=nfft, noverlap=nfft//2, scaling='spectrum')

print(f"shape of spectrogram = {spgrm[2].shape}")

spgrm_neg = spgrm[2]
spgrm_neg[(fft_bin-2):(fft_bin+2),:] = 0
detect = np.zeros((2, spgrm_neg.shape[1]))
detect[0, :] = spgrm_neg.sum(axis=0)
detect[1, :] = spgrm[2][(fft_bin-2):(fft_bin+2)].sum(axis=0)
threshold = 0.1
scaling = detect.sum(axis=0)
scaling[scaling<threshold] = threshold
detect = detect/scaling

plt.plot(detect[1,:])

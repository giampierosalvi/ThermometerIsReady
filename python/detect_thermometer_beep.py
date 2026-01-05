# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Detect thermometer beep

# %%
from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
import numpy as np
from python_speech_features import sigproc

# %%
# frequency of termometer signal
freq = 6080
nfft = 2**12

samplerate, data = wavfile.read('../data/thermometer00.wav')
fft_bin = int(float(freq)*float(nfft)/float(samplerate))

# use one of the two channels
frames = sigproc.framesig(data[:, 0], nfft, nfft//2, winfunc=np.hamming)
nframes = frames.shape[0]
frame_len = frames.shape[1]

# %%
autocorr = np.zeros((nframes, frame_len*2-1))
for t in range(nframes):
    autocorr[t, :] = np.correlate(frames[t], frames[t], mode='full')

# %%
plt.imshow(np.log(autocorr))

# %%
tref = np.linspace(0, frame_len/samplerate, frame_len)
xref = np.sin(tref*freq*2*np.pi)
crosscorr = np.zeros((nframes, frame_len*2-1))
for t in range(nframes):
    crosscorr[t, :] = np.correlate(frames[t], xref, mode='full')

# %%
plt.figure(figsize=(16,6))
plt.plot(np.linspace(0, nframes*(nfft//2)/samplerate, nframes), crosscorr.max(axis=1)/autocorr.max(axis=1))
plt.xlabel('time (sec)')
plt.axis([100, 130, 0, 0.06])

# %%
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


# %%

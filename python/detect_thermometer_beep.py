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
# templates of thermometer signal: defines tone or combination of tones emitted by the thermometer
templates = [
    {'frequencies': [6080], 'amplitudes': [1.0], 'model': 'Domotherm'},
    {'frequencies': [3937, 7875, 11813, 15735, 19673], 'amplitudes': [0.2, 0.2, 0.2, 0.2, 0.2], 'model': 'Braun PRT1000'}
]
# choose size of analysis window
frame_len = 2**12
samplerate = 48000 # we should verify that this is the same for sound
# generate template data
tref = np.linspace(0, frame_len/samplerate, frame_len)
for template in templates:
    template['data'] = np.zeros(frame_len)
    for c in range(len(template['frequencies'])):
        template['data'] += template['amplitudes'][c] * np.sin(2*np.pi*tref*template['frequencies'][c])


# %%
def detect(data, frame_len, templates):
    ntemplates = len(templates)
    # divide recording into frames using one of the two channels
    frames = sigproc.framesig(data[:, 0], frame_len, frame_len//2, winfunc=np.hamming)
    nframes = frames.shape[0]
    # energy of the signal as reference
    energy = (frames**2).sum(axis=1)
    # compute cross correlation with template data
    crosscorr = np.zeros((nframes, frame_len*2-1))
    detection = np.zeros((nframes, ntemplates))
    for idx, template in enumerate(templates):
        for t in range(nframes):
            crosscorr[t, :] = np.correlate(frames[t], template['data'], mode='full')
        detection[:, idx] = (crosscorr**2).sum(axis=1)/energy
    return detection


# %%
# labels:
labels = [t['model'] for t in templates]
time_step = (frame_len//2)/samplerate
# read wave data
fig, axs = plt.subplots(3, 1, figsize=(16, 12))
samplerate, data = wavfile.read('../data/thermometer00.wav')
detection = detect(data, frame_len, templates)
axs[0].plot(np.linspace(0, detection.shape[0]*time_step, detection.shape[0]), detection)
axs[0].legend(labels)
samplerate, data = wavfile.read('../data/thermometer01.wav')
detection = detect(data, frame_len, templates)
axs[1].plot(np.linspace(0, detection.shape[0]*time_step, detection.shape[0]), detection)
axs[1].legend(labels)
samplerate, data = wavfile.read('../data/thermometer02.wav')
detection = detect(data, frame_len, templates)
axs[2].plot(np.linspace(0, detection.shape[0]*time_step, detection.shape[0]), detection)
axs[2].legend(labels)


# %%
# obsolete, kept for reference
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
plt.plot(templates[1]['frequencies'])

# %%
freqs = templates[1]['frequencies']
freqs[0]*5-freqs[4]

# %%

# %%

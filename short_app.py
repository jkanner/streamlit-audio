import streamlit as st
import pandas as pd
import numpy as np
from numpy import random

import matplotlib.pyplot as plt

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwosc.api import fetch_event_json

from copy import deepcopy

import io
from scipy import signal
from scipy.io import wavfile

# Title the app
st.title('Audio signal processing demo')

st.markdown("""
 * Use the menu at left to select data and set plot parameters
 * Learn more at https://gw-openscience.org
""")


t0 = 0



def make_audio(bp_data, t0):
    # -- window data for gentle on/off
    window = signal.windows.tukey(len(bp_data), alpha=1.0/10)
    win_data = bp_data*window

    # -- Normalize for 16 bit audio
    win_data = np.int16(win_data/np.max(np.abs(win_data)) * 32767 * 0.9)

    return win_data.value


def make_audio_file(bp_data, t0=None):
    # -- window data for gentle on/off
    window = signal.windows.tukey(len(bp_data), alpha=1.0/10)
    win_data = bp_data*window

    # -- Normalize for 16 bit audio
    win_data = np.int16(win_data/np.max(np.abs(win_data)) * 32767 * 0.9)


    fs=1/win_data.dt.value
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, int(fs), win_data)
    
    return virtualfile


    



# -- Try making some random noise
@st.cache
def makewhitenoise(fs, dt):
    noise = TimeSeries(random.normal(scale=.1, size=fs*noisedt), sample_rate=fs)
    return noise

fs = 32000
noisedt = 8
noise = deepcopy(makewhitenoise(fs, noisedt))
    
st.markdown("## Filtering Tutorial")

st.markdown("""
In this demo, we will try to find a **secret sound** hidden in noisy
data.  To do this, we will practice with a few signal processing concepts:

 * Plotting data in the time domain
 * Plotting data in the frequency domain
 * Band-pass filtering
 * Whitening
""")

st.markdown("## White noise")

st.markdown("""
To get started, we'll take a look at some **white noise**.  Any 
signal can be represented based on its frequency content.  When we 
say that noise is *white*, we mean that signal has about the same 
signal power at all frequencies.  

Below, we'll represent the **same signal three different ways**:

 * A time-domain signal
 * A frequency domain signal
 * An audio file
""")

st.markdown("### Time-domain")

st.markdown("""
In the time-domain, we see the siganl as a function of time.  The 
x-axis represents time, and the y-axis represents the **amplitude** of
the signal.  For an audio signal, the amplitude corresponds to the 
amount of pressure felt on your ear-drum at any moment.  For a 
gravitatonal wave signal, the amplitude represents the strain - 
or fractional change in length - of the observatory's arms.
""")

st.pyplot(noise.plot())

st.markdown("### Frequency domain")

st.markdown("""
In the **frequency domain**, the x-axis represents a frequency 
value, and the y-axis shows the amount of signal power at that
frequency.  Since white noise has about the same power at each 
frequency, this plot is mostly flat as you move from left to right.
""")

figwn = noise.asd(fftlength=1).plot()
plt.ylim(1e-10, 1)
st.pyplot(figwn)
#st.pyplot(noise.asd(fftlength=1).plot())

st.markdown("### Audio player")
st.markdown("""
You can use the audio player to listen the signal.  You should hear
 a hiss of white noise (works best in Chrome browser).
""")

st.audio(make_audio_file(noise), format='audio/wav')


#-- Try to color the noise

noisefreq = noise.fft()
color = 1.0 / (noisefreq.frequencies)**2
indx = np.where(noisefreq.frequencies.value < 30)
color[indx] = 0  #-- Avoid weirdness at frequency 0 and try low freq cut-off

# -- Try making red noise

weightedfreq = noisefreq * color.value

# -- Try returning to time domain
colorednoise = weightedfreq.ifft()

###
# -- Inject the signal
###
secret = TimeSeries.read('LOZ_Secret.wav')

# -- Normalize and convert to float
#secret = np.int16(secret)
secret -= secret.value[0]
#secret = np.float(secret)
secret = np.float64(secret)
secret = secret/np.max(np.abs(secret)) * 1*1e-8
secret.t0 = 4
#secret = secret/np.max(np.abs(secret)) * 1e-7



#-- Window the noise
#window = signal.windows.tukey(len(colorednoise), alpha=1.0/10)
#colorednoise = colorednoise*window






#st.markdown("Next two plots show secret")
#st.pyplot(secret.plot())
#st.pyplot(secret.asd().plot())

#audiodata = make_audio(secret, t0)
#fs=1/secret.dt.value
#virtualfile = io.BytesIO()
#wavfile.write(virtualfile, int(fs), audiodata)
#st.audio(virtualfile, format='audio/wav')

st.markdown("## Red Noise")

st.markdown("""
Next, we'll look at some **red noise**.  Red noise 
has more power at low frequencies than high high frequencies.

Imagining random noise at different frequencies can be a hard thing
to understand.  A silly way to picture this is as a sports stadium
full of animals cheering. Some animals (like birds and kittens)
cheer with higher pitches, and other animals (like bull frogs and 
lions) will cheer with lower pitches.  If the stadium has animals of 
all kinds in equal numbers, you might get white noise cheering.  
If the stadium is full of low pitch creatures (say, full of bull frogs), 
you might get red noise cheering.  Can you imagine the difference?

A similar idea can be seen in noise in the LIGO and Virgo instruments.
Low frequency noise sources contribute noise at low frequencies.  These 
are big, slowly vibrating things, especially motion from the constant 
shaking of the ground, called seismic motion.  At higher frequencies, 
there are lots of noise souces from vibrating instrument parts, like 
shaking mirrors and tables.  
""")

###
# -- Show red noise with signal
###

maze = colorednoise.inject(secret)

st.markdown("### Time-domain (red noise)")
st.markdown("In the time-domain, you can see the signal look random")
st.pyplot(maze.plot())
figrn = maze.asd(fftlength=1).plot()
plt.ylim(1e-11, 1e-4)
plt.xlim(30, fs/2)
st.pyplot(figrn)
st.audio(make_audio_file(maze), format='audio/wav')

#audio = make_audio(maze, t0)
#outfile = io.BytesIO()
#wavfile.write(outfile, int(fs), audio)
#st.audio(outfile, format='audio/wav')


# ----
# Try to recover the signal
# ----

st.markdown("## Find the Secret Sound")

st.markdown("""
The red noise above isn't just noise - there's a secret sound 
inside.  Did you hear it?  Probably not!  All of that low-frequency
noise is making the secret sound very hard to hear.  But ... if the
secret sound is not at low frequencies, maybe we could still hear it.

What we need is a way to get rid of some of the low frequency noise, 
while keeping the high frequency part of the signal.  In signal processing,
this is known as a **high pass filter** - a filter that all removes
low frequency sounds, and keeps (or allows to pass) the high frequency 
sounds.  The frequency above which signals are passed is called the 
**cutoff frequency**.

See if you can use a high pass filter to find the secret sound.  Adjust the 
cutoff frequency using the slider below, and see if you can remove 
some noise to find the secret sound.

""")

lowfreq = st.slider("High pass filter cutoff frequency", 0, 3000, 0, step=100)

if lowfreq == 0: lowfreq=1

highpass = maze.highpass(lowfreq)
st.pyplot(highpass.plot())
fighp = highpass.asd(fftlength=1).plot()
plt.ylim(1e-11, 1e-4)
plt.xlim(30, fs/2)
st.pyplot(fighp)
st.audio(make_audio_file(highpass), format='audio/wav')




# -- Try to recover the signals
whitemaze = maze.whiten()
st.pyplot(whitemaze.plot())


#-- Try making an audio file from real LIGO data
#audiodata = make_audio(bp_data, t0)
#fs = 16384
audio = make_audio(whitemaze, t0)
outfile = io.BytesIO()
wavfile.write(outfile, int(fs), audio)
st.audio(outfile, format='audio/wav')


#-- bandpass


highfreq = st.slider("High frequency cut-off", 1000, 16000, 2000)
bpmaze = maze.bandpass(lowfreq, highfreq)

#bpmaze = whitemaze.highpass(800)
st.markdown("## Better audio mehod test")
st.pyplot(bpmaze.plot())
st.audio(make_audio_file(bpmaze), format='audio/wav')


#-- Q transform
#dt = 2  #-- Set width of q-transform plot, in seconds
#hq = maze.q_transform(outseg=(4-dt, 4+dt), qrange=(100,300))
#fig4 = hq.plot()
#ax = fig4.gca()
#fig4.colorbar(label="Normalised energy", vmax=40)
#ax.grid(False)
#ax.set_yscale('log')
#plt.ylim(30, 16000)
#st.pyplot(fig4)


page = st.number_input('Section #', min_value=1, max_value=4, value=1)

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

from freqdomain import showfreqdomain


# Title the app
st.title('Signal Processing Tutorial')

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

@st.cache   #-- Magic command to cache data
def load_gw(t0, detector):
    strain = TimeSeries.fetch_open_data(detector, t0-14, t0+14, cache=False)
    return strain


# -- Method to make and cache random noise
@st.cache
def makewhitenoise(fs, dt):
    noise = TimeSeries(random.normal(scale=.1, size=fs*noisedt), sample_rate=fs)
    return noise

fs = 32000
noisedt = 8
noise = deepcopy(makewhitenoise(fs, noisedt))

#-- Try to color the noise
noisefreq = noise.fft()
color = 1.0 / (noisefreq.frequencies)**2
indx = np.where(noisefreq.frequencies.value < 30)
color[indx] = 0  #-- Apply low frequency cut-off at 30 Hz

#-- Red noise in frequency domain
weightedfreq = noisefreq * color.value

# -- Try returning to time domain
colorednoise = weightedfreq.ifft()

###
# -- Inject the signal
###
secret = TimeSeries.read('LOZ_Secret.wav')

# -- Normalize and convert to float
secret -= secret.value[0]  #-- Remove constant offset
secret = np.float64(secret)
secret = secret/np.max(np.abs(secret)) * 1*1e-8   #-- Set amplitude
secret.t0 = 4
maze = colorednoise.inject(secret)

# -- Might be useful to make easier to hear option
mazeloud = colorednoise.inject(10*secret)


# -------
# Begin Display Here
# -------


st.markdown("## Introduction")

st.markdown("""
In this demo, we will try to find a **secret sound** hidden in noisy
data.  To do this, we will practice with a few signal processing concepts:

 * Plotting in the time domain and frequency domain
 * Highpass and bandpass filtering
 * Whitening

*Note: This app works best in the **Chrome Browser** *
""")

sectionnames = [
                'Introduction to the frequency domain',
                'White Noise',
                'Red Noise',
                'Find the Secret Sound',
                'Whitening',
                'Gravitational Wave Data',
]

def headerlabel(number):
    return "{0}: {1}".format(number, sectionnames[number-1])
    
page = st.radio('Select Section:', [1,2,3,4,5,6], format_func=headerlabel)

st.markdown("## {}".format(headerlabel(page)))

if page==1:
    
    showfreqdomain()
    
if page==2:

    # White Noise
    
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

    st.markdown("### Time domain")

    st.markdown("""
    In the time domain, we see a siganl as a function of time.  The 
    x-axis represents time, and the y-axis represents the **amplitude** of
    the signal.  For an audio signal, the amplitude corresponds to the 
    amount of pressure felt on your eardrum at any moment.  For a 
    gravitatonal-wave signal, the amplitude represents the strain - 
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


    st.markdown("### Audio player")
    st.markdown("""
    You can use the audio player to listen the signal.  You should hear
    a hiss of white noise (works best in Chrome browser).
    """)
    
    st.audio(make_audio_file(noise), format='audio/wav')

    st.markdown("")
    st.markdown("""
    When ready, go to the next section using the controls at the 
    top.
    """)
    
if page == 3:

    # st.markdown("## 3: Red Noise")
    
    st.markdown("""
    Next, we'll look at some **red noise**.  Red noise 
    has more power at low frequencies than high frequencies.
    
    Imagining random noise at different frequencies can be a hard thing
    to understand.  A silly way to picture this is as a sports stadium
    full of animals cheering. Some animals (like birds and kittens)
    cheer with higher pitches, and other animals (like bullfrogs and 
    lions) will cheer with lower pitches.  If the stadium has animals of 
    all kinds in equal numbers, you might get white noise cheering.  If 
    the stadium is full of low pitch creatures (say, lots of bullfrogs), 
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

    st.markdown("In the time-domain, you can see the signal looks random.")
    st.pyplot(maze.plot())

    st.markdown("In the frequency-domain, the red noise has lots of power at low frequencies.")
    figrn = maze.asd(fftlength=1).plot()
    plt.ylim(1e-11, 1e-4)
    plt.xlim(30, fs/2)
    st.pyplot(figrn)
    st.audio(make_audio_file(maze), format='audio/wav')
    st.markdown("""
    Can you hear the bullfrogs cheering?

    How does this compare with the white noise sound?
    """)

if page == 4:

    # ----
    # Try to recover the signal
    # ----
    # st.markdown("## 4: Find the Secret Sound")
    
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
    #st.pyplot(highpass.plot())

    fighp = highpass.asd(fftlength=1).plot()
    plt.ylim(1e-12, 1e-5)
    plt.xlim(30, fs/2)
    #plt.vlines(lowfreq, 1e-11, 1e-4, colors='red')
    ax = plt.gca()
    ax.axvspan(1, lowfreq, color='red', alpha=0.3, label='Removed by filter')
    plt.legend()
    st.pyplot(fighp)
    st.audio(make_audio_file(highpass), format='audio/wav')

    st.markdown("Can you hear the sound now?  What value of the cutoff frequency makes it easiest to hear?")

    st.markdown("")
    needhint = st.checkbox("Need a hint?", value=False)

    if needhint:

        st.markdown("""Here is the secret sound.  Can you find it hidden in the
        red noise above?
        """)

        st.audio(make_audio_file(secret), format='audio/wav')
        
if page == 5:
    # st.markdown("## 5: Whitening")

    st.markdown("""
    **Whitening** is a process that re-weights a signal, so that all
    frequency bins have a nearly equal amount of noise.  In our example,
    it is hard to hear the signal, because all of the low-frequency 
    noise covers it up.  By whitening the data,
    we can prevent the low-frequency noise from dominating what we hear. 
    
    """)
    
    whiten = st.checkbox("Whiten the data?", value=False)

    if whiten:
        whitemaze = maze.whiten()
    else:
        whitemaze = maze

    st.markdown("""
    After whitening, you can see the secret sound in the time domain.
    """)
    
    st.pyplot(whitemaze.plot())

    figwh = whitemaze.asd(fftlength=1).plot()
    plt.ylim(1e-12, 1)
    plt.xlim(30, fs/2)
    st.pyplot(figwh)
    
    st.audio(make_audio_file(whitemaze), format='audio/wav')

    st.markdown("""Try using the checkbox to whiten the data.  Is it 
    easier to hear the secret sound with or without whitening?
    """)


if page == 6:

    # st.markdown("## 6: Gravitational Wave Data")

    st.markdown("""
    Finally, we'll try what we've learned on some real 
    gravitational-wave data from LIGO, around the binary black 
    hole signal GW150914.  We'll add one more element: 
    a **bandpass filter**.  A bandpass filter uses both a low frequency
    cutoff and a high frequency cutoff, and only passes signals in the 
    frequency band between these values. 

    Try using a whitening filter and a band-pass filter to reveal the
    gravitational wave signal in the data below.  
    """)

    detector = 'H1'
    t0 = 1126259462.4   #-- GW150914

    st.text("Detector: {0}".format(detector))
    st.text("Time: {0} (GW150914)".format(t0))
    strain = load_gw(t0, detector)
    center = int(t0)
    strain = strain.crop(center-14, center+14)

    # -- Try whitened and band-passed plot
    # -- Whiten and bandpass data
    st.subheader('Whitened and Bandbassed Data')

    lowfreqreal, highfreqreal = st.slider("Band-pass filter",
                                          1, 2000, value=(1,2000) )

    makewhite = st.checkbox("Apply whitening", value=False)

    if makewhite:
        white_data = strain.whiten()
    else:
        white_data = strain

    bp_data = white_data.bandpass(lowfreqreal, highfreqreal)

    st.markdown("""
    With the right filtering, you might be able to see the signal in the time domain plot.
    """)
    
    fig3 = bp_data.plot()
    plt.xlim(t0-0.1, t0+0.1)
    st.pyplot(fig3)



    # -- PSD of whitened data
    # -- Plot psd
    plt.figure()
    psdfig = bp_data.asd(fftlength=4).plot()
    plt.xlim(10, 1800)
    ax = plt.gca()
    ax.axvspan(1, lowfreqreal, color='red', alpha=0.3, label='Removed by filter')
    ax.axvspan(highfreqreal, 1800, color='red', alpha=0.3, label='Removed by filter')   
    st.pyplot(psdfig)

    # -- Audio
    st.audio(make_audio_file(bp_data.crop(t0-1, t0+1)), format='audio/wav')

    # -- Close all open figures
    plt.close('all')

    st.markdown("""With the right filtering, you might be able to hear
    the black hole signal.  It doesn't sound like much - just a quick thump.  
 """)

    st.markdown("")
    hint = st.checkbox('Need a hint?')

    if hint:

        st.markdown("""
        Hint: Try using a band pass from 30 to 400 Hz, with whitening on.
        This is similar to what was used for Figure 1 of the 
        [GW150914 discovery paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.116.061102), also shown below:
        """)
        
        st.image('https://journals.aps.org/prl/article/10.1103/PhysRevLett.116.061102/figures/1/large')


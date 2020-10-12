import streamlit as st
import gwpy
from gwpy.timeseries import TimeSeries
from scipy import signal
from numpy import random
import numpy as np
import io
from scipy.io import wavfile
import matplotlib.pyplot as plt

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
    noise = TimeSeries(random.normal(scale=.1, size=fs*dt), sample_rate=fs)
    return noise


def makesine(freq, amp, makeplot=True, cropstart=1.0, cropend=1.05):
    fs = 4096
    
    time = np.arange(0,3, 1.0/fs)
    y1 = amp*np.sin( 2*np.pi*freq*time )
    if amp>0:
        sig1 = TimeSeries(y1, dt=1.0/fs).taper() 
    else:
        sig1 = TimeSeries(y1, dt=1.0/fs)
    if makeplot:
        fig_sig1 = sig1.crop(cropstart, cropend).plot(
            xlim=(cropstart, cropend),
            ylim=(-5,5),
            title='Frequency {0} Hz - Amplitude {1}'.format(freq,amp),
            ylabel='Pressure',
            xlabel='Time (Seconds)',
        )
        st.pyplot(fig_sig1, clear_figure=True)
    return(sig1)



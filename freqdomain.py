import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets

cropstart = 1.0
cropend   = 1.3

def makesine(freq, amp):
    fs = 4096
    time = np.arange(0,3, 1.0/fs)
    y1 = amp*np.sin( 2*np.pi*freq*time )
    sig1 = TimeSeries(y1, dt=1.0/fs).taper() # ALS: Effect visible in plot: need to address or hide.
    plt.figure()
    fig_sig1 = sig1.crop(cropstart, cropend).plot()
    plt.xlim(cropstart, cropend)
    plt.ylim(-5,5)
    plt.title('Frequency {0} Hz - Amplitude {1}'.format(freq,amp))
    st.pyplot(fig_sig1, clear_figure=True)
    return(sig1)


def showfreqdomain():

    st.markdown("""

INTRODUCTION

An important step in many signal processing algorithms is to transform time series data (data points sequential in time) into a new representation in the frequency domain.  We will begin to explain what that means and why it is useful in this tutorial.  

FOURIER SERIES

Any periodic signal, $s_N(x)$, can be constructed by adding together sine waves of different amplitudes, frequencies, and phases, as:

    """)
    
    st.latex("""
    s_N(x) = \\frac{a_0}{2} + \\sum_{n=1}^N \\left(a_n \\cos\\left(\\tfrac{2\\pi nx}{P}\\right) + b_n \\sin\\left(\\tfrac{2\\pi nx}{P} \\right) \\right).
    """)

    st.markdown("""
    Where P is the period and the coefficients $a_n$ and $b_n$ represent the amplitude of each 
    sine wave.  We can refer to $a_n$ and $b_n$ as the **frequency domain amplitudes**, or 
    Fourier series.  Notice there is one amplitude for each sine wave used to re-construct
    the original signal.  By calculating the frequency domain amplitudes for a signal, we can 
    learn what frequencies were important in constructing a time-domain signal.  The 
    process of converting a time-domain signal to its frequency domain amplitudes is called a 
    Fourier Transform.
    """)

    st.markdown("""
    As an example, consider this animation (available on Wikipedia):
    """)
    
    st.image('https://upload.wikimedia.org/wikipedia/commons/b/bc/Fourier_series_for_square_wave.gif')

    st.markdown("""
    In the image, the blue square wave is the time domain signal.  The red curve represents a sum 
    of sine waves.  At each step of the animation, an additional sine wave is added, so that by the 
    time a large number of sine waves are included, the red curve is a very good match to the blue 
    curve.  The amplitudes and phases associated with each added sine wave describes
    the frequency content of the signal.  
    """)

    st.markdown("""
    THE FREQUENCY DOMAIN

    Data that are normally recorded in experiments are typically either sequential in time or space.  
    Let’s assume that the horizontal axis on the square wave shown above has units of time.  
    Then we would say that this is a representation of the data in the time domain because 
    it is a function of time.  Using the Fourier Transform, we can convert that into a function 
    of frequency yielding an equivalent representation in the frequency domain.
    """)
    
    st.markdown("### Make a signal with 3 sine waves")

    st.markdown("""
    Let’s create a time-domain signal that is made up of 3 sine waves where we 
    determine the frequency and amplitude.  Drag each 
    slider to the desired frequency and amplitude of each wave.  The plot of each wave 
    will then be updated in the main window along with the sum of the three.
    """)
    
    st.markdown("#### Sine Wave 1")
    freq1 = st.slider("Frequency", 20, 200, 20)
    amp1 = st.slider("Amplitude", 1.0, 5.0, 5.0)
    sig1 = makesine(freq1, amp1)
    
    st.markdown("#### Sine Wave 2")
    freq2 = st.slider("Frequency", 20, 200, 103)
    amp2 = st.slider("Amplitude", 1.0, 5.0, 2.0)
    sig2 = makesine(freq2, amp2)
    
    st.markdown("#### Sine Wave 3")
    freq3 = st.slider("Frequency", 20, 200, 195)
    amp3 = st.slider("Amplitude", 1.0, 5.0, 4.0)
    sig3 = makesine(freq3, amp3)

    st.markdown("### Adding the 3 sine waves together:")
    plt.figure()
    signal = sig1 + sig2 + sig3
    figsum = signal.crop(cropstart, cropend).plot(color='orange')
    plt.xlim(cropstart, cropend)
    plt.title("Total signal in time domain")
    st.pyplot(figsum, clear_figure=True)

    st.markdown("""
    Imagine that we've recorded the signal above in the lab, and we would like to figure out
    what frequencies went into producing it.  To do this, we can apply a Fourier Transform
    to the data.  The result is shown below:
    """)

    freqdomain = signal.fft()
    # sigfig = np.abs(freqdomain).plot()
    freqplot = plt.figure()
    plt.plot(freqdomain.frequencies, np.abs(freqdomain), color='orange')
    plt.title("Total signal in frequency domain")
    plt.ylim(0,5)
    plt.xlim(0,250)
    plt.ylabel('Signal amplitude')
    plt.xlabel('Frequency (Hz)')
    st.pyplot(freqplot, clear_figure=True)
    #st.pyplot(sigfig)

    st.markdown("""
    How does the result of the Fourier Transform compare to the signal you generated?

    One way to think about this is that each of the 3 sine waves is one 
    note played by an instrument.  Our ears hear all 3 notes together.
    Converting the signal to the frequency domain lets us see
    which 3 notes were being played.
    """)
    
    # -- Close all open figures
    plt.close('all')

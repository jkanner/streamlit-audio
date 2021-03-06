import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

import requests, os
from gwpy.timeseries import TimeSeries
from gwosc.locate import get_urls
from gwosc import datasets
from gwpy.plot import Plot
from scipy import signal

from helper import makesine, make_audio_file, plot_signal

# -- Need to lock plots to be more thread-safe
from matplotlib.backends.backend_agg import RendererAgg
lock = RendererAgg.lock

cropstart = 1.0
cropend   = 1.05


def showfreqdomain():

    st.markdown("""

INTRODUCTION

An important step in many signal processing algorithms is to transform time series data 
(data points sequential in time) into a new representation in the frequency domain.  
We will begin to explain what that means and why it is useful in this tutorial, by 
recreating a target signal from components.

THREE NOTES

The target signal below is composed of 3 different pitches, or 
**[frequencies](https://youtu.be/Axx8WfxQDkk)**.  Imagine we record this signal from our 
favorite song, and we want to figure out the three frequencies used to make it.  How could we do this?  A similar 
problem comes up in many experiments, when we record some data, and then wish to know what frequencies were used
to generate the signal.

""")

    st.markdown("#### Target signal in time domain:")

    sig1 = makesine(200, 4, False)
    sig2 = makesine(250, 3, False)
    sig3 = makesine(300, 2, False)
    
    totalsignal = sig1+sig2+sig3
    plot_signal(totalsignal, color_num=1)

    st.audio(make_audio_file(totalsignal), format='audio/wav')

    st.markdown("""
    The above plot shows the target signal in the **time domain**.  In a time-domain plot, the x-axis 
    always represents time.  The y-axis represents the quantity measured at each time sample.  
    For sound, this is the air pressure striking your ear or microphone at any moment.

    Can you tell which 3 **frequencies**, or pitches, were used to create this signal?  Probably not!
    While the time domain is how we often record data, it is not a good way to see the componennt frequencies.
    Instead, we can use a process known as a 
    [Fourier Transform](https://www.youtube.com/watch?v=1JnayXHhjlg) to convert the signal to the 
    **frequency domain**.  

    :point_right: **Click the check box below to convert the target signal to the frequency domain**.

    """)

    showfreq = st.checkbox('Convert target signal to the frequency domain', value=False)

    if showfreq:
        freqdomain = totalsignal.fft()

        source = pd.DataFrame({
            'Frequency (Hz)': freqdomain.frequencies,
            'Amplitude': np.abs(freqdomain.value),
            'color':['#1f77b4', '#ff7f0e'][1]
        })

        chart = alt.Chart(source).mark_line().encode(
            alt.X('Frequency (Hz)',
                  scale=alt.Scale(
                      domain=(0, 400),
                      clamp=True)),
            alt.Y('Amplitude:Q',
                  scale=alt.Scale(
                      domain=(-0, 5),
                      clamp=True)),
            color=alt.Color('color', scale=None)
        ).properties(title='Target Signal in Frequency Domain')

        st.altair_chart(chart, use_container_width=True)
            
        st.markdown("""
        Converting to the **frequency domain** shows us the individual components that contributed to the total.
        In the **frequency domain**, the frequency (or pitch) of each component signal is shown on the x-axis.
        The **[amplitude](https://www.youtube.com/watch?v=TsQL-sXZOLc)** 
        (or loudness) of each component signal is shown on the y-axis.

        Using the frequency domain plot above:
        * What are the 3 frequencies used to make the total signal?  
        * What is the amplitude of each frequency?
        """)


    st.markdown("""
    :point_right: **Try to recreate the above signal, using three components, or notes.  You can adjust the sliders to create each component**.
    """)

    st.markdown("#### Component 1")
    freq1 = st.slider("Frequency (Hz)", 100, 400, 100, step=10)
    amp1 = st.number_input("Amplitude", 0, 5, 0, key='amp1slider')

    with lock:
        guess1 = makesine(freq1, amp1)
    
    st.markdown("#### Component 2")
    freq2 = st.slider("Frequency (Hz)", 100, 400, 150, step=10)
    amp2 = st.number_input("Amplitude", 0, 5, 0, key='amp2slider')

    with lock:
        guess2 = makesine(freq2, amp2)
    
    st.markdown("#### Component 3")
    freq3 = st.slider("Frequency (Hz)", 100, 400, 200, step=10)
    amp3 = st.number_input("Amplitude", 0, 5, 0, key='amp3slider')

    with lock:
        guess3 = makesine(freq3, amp3)

    st.markdown("### Adding the 3 components together:")
    
    guess  = guess1 + guess2 + guess3

    chart1 = plot_signal(guess, color_num=0, display=False)
    chart2 = plot_signal(totalsignal, color_num=1, display=False)
    chart = (chart2 + chart1).properties(title='Target Signal (orange) & Guess (blue)')
    st.altair_chart(chart, use_container_width=True)
        
    mismatch = (totalsignal.crop(cropstart, cropend) - guess.crop(cropstart, cropend)).value.max()
    # st.write(mismatch)

    if mismatch < 0.1:
        st.markdown("### A perfect match!  Great job!!  :trophy:")
        st.balloons()
    elif mismatch < 3:
        st.markdown("### That's really close!")    
    
    st.markdown("#### Audio for target signal")
    st.audio(make_audio_file(totalsignal), format='audio/wav')

    st.markdown("#### Audio for guess")
    st.audio(make_audio_file(guess), format='audio/wav')
    
    st.markdown("""
    See if you can recreate the target signal, by adjusting the 3 components.  

    *Hint: Look for the component frequencies and amplitudes in the frequency-domain plot.*
    """)

    st.markdown("""
    When ready, go to the next section using the controls at the 
    top.
    """)
    
    
    # -- Close all open figures
    plt.close('all')

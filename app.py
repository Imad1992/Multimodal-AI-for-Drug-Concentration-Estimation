"""Streamlit demo. No patient records or real medical inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sensor_ai.simulation import simulate,load_dataset
from sensor_ai.pipeline import predict
from sensor_ai.optimisation import deconvolve
st.set_page_config(page_title='Multimodal Sensor AI',layout='wide')
st.title('Multimodal Sensor AI Laboratory')
st.warning('Synthetic research demonstration only. Not for diagnosis, treatment, dosing, or clinical decision-making.')
tabs=st.tabs(['Overview & signals','AI predictions','Evolutionary deconvolution','Model evaluation'])
path=Path('data/synthetic_sensors.npz');ready=Path('artifacts/fusion_model.pt').exists()
if not path.exists():
    st.error('Dataset missing. Run: python -m scripts.bootstrap');st.stop()
s,e,y=load_dataset(path)
with tabs[0]:
    st.write('Paired synthetic SERS-like spectra (128 points) and electrochemical-like traces (64 points), with three normalised analyte concentrations.')
    ix=st.slider('Sample index',0,len(y)-1,0)
    c1,c2=st.columns(2)
    with c1: st.subheader('SERS-like signal');st.line_chart(pd.DataFrame({'Intensity':s[ix]}))
    with c2: st.subheader('Electrochemical-like signal');st.line_chart(pd.DataFrame({'Current (arbitrary units)':e[ix]}))
    st.write('Simulated ground truth (unitless):',dict(zip(['Analyte A','Analyte B','Analyte C'],np.round(y[ix],3))))
with tabs[1]:
    if not ready: st.info('Train the model with python -m scripts.bootstrap')
    else:
        idx=st.number_input('Measurement index',0,len(y)-1,0,key='pred')
        if st.button('Predict concentrations'):
            estimate=predict(s[idx:idx+1],e[idx:idx+1])[0]
            st.dataframe(pd.DataFrame({'Analyte':['A','B','C'],'Prediction':estimate,'Synthetic truth':y[idx]}),hide_index=True)
            st.caption('Unitless, synthetic concentrations. No calibrated drug concentration units.')
with tabs[2]:
    idx=st.number_input('Sample for optimisation',0,len(y)-1,0,key='opt')
    if st.button('Run differential evolution'):
        with st.spinner('Optimising...'):
            result=deconvolve(s[idx],e[idx],maxiter=60)
        st.json(result)
        st.write('Synthetic ground truth:',y[idx].tolist())
with tabs[3]:
    p=Path('artifacts/metrics.json')
    if p.exists():
        st.json(json.loads(p.read_text()))
        h=Path('artifacts/history.json')
        if h.exists():st.line_chart(pd.DataFrame(json.loads(h.read_text())).set_index('epoch'))
    else:st.info('Train first to see holdout evaluation metrics.')

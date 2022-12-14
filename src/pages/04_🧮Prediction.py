import streamlit as st
import pandas as pd
from lib.prediction.supervised import classification, regression

st.sidebar.markdown(
    """
    
    Created By: [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/hazirafidi)
    
    [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
    
    """)

st.title("Prediction")
options = st.selectbox("Select Model", ("Regression", "Classification"))
if options=="Regression":
    regression.main()
if options=="Classification":
    classification.main()
import streamlit as st
from lib.modelling.supervised import regression
from lib.modelling.supervised import classification
from lib.modelling.unsupervised import clustering

st.sidebar.markdown(
    """
    
    Created By: [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/hazirafidi)
    
    [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
    
    """)

st.title("AutoML with Pycaret")
module = st.radio("Select module", ("Supervised ML", "Unsupervised ML"), horizontal=True)

if module == "Supervised ML":
    with st.expander("Collapse", expanded=True):
        option = st.selectbox("Select your model", ("Regression", "Classification"))

        if option == "Regression":
            regression.main()

        if option == "Classification":
            classification.main()

if module == "Unsupervised ML":
    with st.expander("Collapse", expanded=True):
        option = st.selectbox("Select your model", ("Clustering", "Anomaly Detection", "NLP", "Data Mining"))

        if option == "Clustering":
            st.info("Page under maintenance")

        if option == "Anomaly Detection":
            st.info("Page under maintenance")

        if option == "NLP":
            st.info("Page under maintenance")

        if option == "Data Mining":
            st.info("Page under maintenance")

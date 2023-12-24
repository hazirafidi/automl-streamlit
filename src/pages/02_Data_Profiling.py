import streamlit as st
import pandas as pd
import os
# from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


st.sidebar.markdown(
    """
    
    Created By: [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/hazirafidi)
    
    [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
    
    """)

st.title("Data Profiling")
options = st.selectbox("Select your data", ("Model Data", "Unseen Data"))

if options=="Model Data":
    with st.container():
        path = "./data/df_model.csv"
        if path:
            try:
                df = pd.read_csv(path, index_col=None)
                st.dataframe(df, use_container_width=True)
                df_info = pd.DataFrame(df.dtypes, columns=['Data Type'])
                st.subheader("Data Type")
                st.dataframe(df_info, use_container_width=True)
                st.info(f"Sample Size: {len(df)} rows")
                if st.button("Profiling"):
                    pr = ProfileReport(df, title="Profiling Report")
                    # pr = df.profile_report()
                    st_profile_report(pr)
                    report = pr.to_html()
                    st.download_button(label='Download Report', data=report, file_name="Analysis.html")
            except FileNotFoundError:
                st.error("Data Not found!")

if options=="Unseen Data":
    with st.container():
        path = "./data/df_unseen.csv"
        if path:
            try:
                df = pd.read_csv(path, index_col=None)
                df_info = pd.DataFrame(df.dtypes, columns=['Data Type'])
                st.dataframe(df_info, use_container_width=True)
                st.info(f"Sample Size: {len(df)} rows")
                if st.button("Profiling"):
                    pr = ProfileReport(df, title="Streamlit AutoML Dataset")
                    st_profile_report(pr)
                    report = pr.to_html()
                    st.download_button(label='Download Report', data=report, file_name="Analysis.html")
            except FileNotFoundError:
                st.error("Data Not found!")

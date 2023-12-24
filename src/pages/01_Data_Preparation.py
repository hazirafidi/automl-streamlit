import streamlit as st
import os
import pandas as pd

st.sidebar.markdown(
    """
    
    Created By: [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/hazirafidi)
    
    [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
    
    """)

st.title("Data Preparation")
with st. expander("Collapse", expanded=True):
    st.markdown(
        """
        **Instructions:**
        
        1. Before model training, data should be splitted to model data(training/validation data) and unseen data(test data)
        2. Upload your dataset here use split button to separate between model data and unseen data
        3. Model data will be used in Modelling section for the model training and unseen data will be used in the prediction section

        ---
        
        """)

with st.expander("Collapse", expanded=True):
    file = st.file_uploader("Upload your file", type=["csv"])
    if file:
        with open(os.path.join("data", file.name), "wb") as f:
            f.write(file.getbuffer())
        df = pd.read_csv(file, index_col=None)
        df_info = pd.DataFrame(df.describe())
        st.dataframe(df, use_container_width=True)
        st.subheader("Data Statistic")
        st.write("""
        Describe the statistic values of your data
        """)
        st.dataframe(df_info, use_container_width=True)
        if st.button("Split dataset"):
            partition = int(0.8 * len(df))
            df_model = pd.DataFrame(df.iloc[:partition, :]) \
                        .to_csv('./data/df_model.csv', index=False)
            df_unseen = pd.DataFrame(df.iloc[partition:, :].reset_index(drop=True)) \
                        .to_csv('./data/df_unseen.csv', index=False)
            st.success(f"Succesfully split dataset.")    
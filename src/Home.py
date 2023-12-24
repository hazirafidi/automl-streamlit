import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="AutoML", 
    page_icon='🏚️', 
    layout="centered", 
    initial_sidebar_state="expanded",
    menu_items={
        "About": "[![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/hazirafidi)",
    })

st.sidebar.markdown(
    """
    
    Created By: [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/hazirafidi)
    
    [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
    
    """)

st.title("Simple AutoML WebApp")
# img = Image.open('./resources/image.png')
# st.image(img)
st.markdown(
    """
    This Webapp is developed using pycaret and streamlit python framework.

    Instructions.

    1. Navigate to Data Preparation Section to split data for Modelling and Prediction.
    2. Get insights of your data in Data Profiling Section.
    3. Train Machine Learning Algorithms in Modelling Section.
    4. Predict using unseen data at Prediction Section.

    Visit [Pycaret](https://pycaret.gitbook.io/docs/) Official Website for more info and tutorials.

    Created by [![Repo](https://badgen.net/badge/icon/GitHub?icon=github&label)](https://github.com/hazirafidi)

    Made with [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)

    """
    )
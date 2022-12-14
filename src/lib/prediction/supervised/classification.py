from pycaret.regression import predict_model, create_model, load_model
import streamlit as st
import time
import os
import pandas as pd


def main():
    choice = st.radio("Choose your mode", ["Default", "Upload own model"], horizontal=True)

    if choice=="Default":
        with st.expander("Collapse", expanded=True):
            st.subheader("Unseen Data")
            path = "./data/unseen_df.csv"
            if path:
                try:
                    df_unseen = pd.read_csv("./data/df_unseen.csv")
                    st.dataframe(df_unseen, use_container_width=True)
                except FileNotFoundError:
                    st.error("File doesn't exists!")
            model = st.text_input("Enter model name",)
            if st.button("Predict"):
                best_model = create_model(str(model))
                with st.spinner("Wait for it..."):
                    time.sleep(3)
                    predictions = predict_model(best_model, data=df_unseen)
                    st.subheader("Results")
                    st.dataframe(predictions, use_container_width=True)
                    st.success("Done!")
    
    if choice=="Upload own model":
        with st.expander("Collapse", expanded=True):
            st.subheader("Dataset")
            file = st.file_uploader("Upload your dataset here!")
            if file:
                with open(os.path.join("data", file.name), "wb") as f:
                    f.write(file.getbuffer())
                
                df = pd.read_csv(file, index_col=None)
                st.dataframe(df, use_container_width=True)

            st.subheader("Upload Model")
            model = st.file_uploader("Upload your model here!")
            if model:
                with open(os.path.join("data", model.name), "wb") as f:
                    f.write(model.getbuffer())

            if st.button("Predict"):
                with st.spinner("Wait for it..."):
                    saved_mdl = load_model("./data/best_model")
                    time.sleep(3)
                    predictions = predict_model(saved_mdl, data=df)
                    st.subheader("Results")
                    st.dataframe(predictions, use_container_width=True)
                    st.success("Done!")
        
if __name__ == "__main__":
    main()
# from lib.test.setup_data import mainSetup
from pycaret.regression import setup, pull, compare_models, create_model, save_model, plot_model, dashboard
import streamlit as st
import pandas as pd
import os

@st.cache
def save_csv(df):
    csv = df.to_csv("./training/model_summary.csv")
    return csv

@st.cache
def convert_df(df):
    csv = df.to_csv().encode('utf-8')
    return csv

def main():
    path = "./data/df_model.csv"
    if path:
        try:
            df = pd.read_csv(path, index_col=None)
            chosen_target = st.selectbox('Choose the Target Column', df.columns)
            col = [col for col in df.select_dtypes(include=['object'])]
            cat_features = st.multiselect('Select categorical features', df.columns, default=col)
            process_data = st.selectbox("Preprocess Data", (False, True), help='Select False if data input already preprocessed.')
            norm = st.selectbox("Normalize Data", (False, True))
            norm_method = st.selectbox("Normalization Method", ('zscore', 'minmax', 'maxabs', 'robust'))
            cpu_num = st.number_input("Enter Number of CPU for the training", value=-1, max_value=8)
            gpu = st.selectbox("Use gpu", (False, True))
            fold_num = st.number_input("Enter number of fold", value=2, min_value=2)
        except FileNotFoundError:
            st.error("File doesn't exist!")

    if st.button('Start AutoML'):
        st.write("Progress Bar")
        progress_bar = st.progress(0)
        status_text = st.empty()
        # clf = mainSetup(df, chosen_target, norm, norm_method, process_data, cat_features, cpu_num,gpu, fold_num)
        setup(
            data=df, target=chosen_target, html= False, preprocess=bool(process_data),
            normalize=bool(norm), normalize_method=norm_method, n_jobs=cpu_num, 
            categorical_features=cat_features, fold=fold_num,
            log_experiment=False, profile=True, use_gpu=bool(gpu)
            )
        reg_df = pull()
        st.subheader("Setup Summary")
        st.dataframe(reg_df,use_container_width=True)
        for i in range(1, 101):
            status_text.text("%i%% Complete" % i)
            compare_models()
            progress_bar.progress(i)
        compare_df = pull()
        st.subheader("Model Summary")
        st.dataframe(compare_df, use_container_width=True)
        save_csv(compare_df)
        csv = convert_df(compare_df)
        st.download_button(label="Download", data=csv, file_name="model_score.csv", mime="text/csv")
        best_model = create_model(str(compare_df.index[0]))
        plot_model(best_model, plot="error", verbose=True, display_format="streamlit")
        st.info("Done!") 
        save_model(best_model, "./model/best_model")
    
    path = "./model/best_model.pkl"
    if os.path.exists(path):
        with open("./model/best_model.pkl", "rb") as f:
            st.download_button('Download Model', f, file_name="best_model.pkl",)


if __name__ == "__main__":
    main()
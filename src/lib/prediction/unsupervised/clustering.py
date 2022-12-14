from pycaret.clustering import predict_model, create_model, pull
import streamlit as st

def predict_cluster(df_unseen):
    compare_df = pull()
    if compare_df is not None:
        try:
            st.dataframe(compare_df, use_container_width=True)
        except NameError:
            st.error("Error")
    else:
        st.info("Error")
    if st.button("Predict"):
        best_model = create_model(str(compare_df.index[0]))
        predictions = predict_model(best_model, data=df_unseen)
        st.subheader("Results")
        st.dataframe(predictions, use_container_width=True)
        st.success("Done!")
 

if __name__ == "__main__":
    predict_cluster()
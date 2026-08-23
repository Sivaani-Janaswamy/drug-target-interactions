import streamlit as st

st.title("DTI-ML Predictor")
st.caption("Initial app skeleton for kinase drug-target affinity prediction.")

st.text_input("SMILES", placeholder="e.g. CCOC(=O)N1CCN(C(=O)...)" )
st.text_input("Protein sequence", placeholder="e.g. MKT..." )

if st.button("Predict"):
    st.info("Model wiring will be added in the next setup phase.")

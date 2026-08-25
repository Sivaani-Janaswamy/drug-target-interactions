import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Ensure root directory is accessible in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.helpers import (
    render_molecule_svg,
    interpret_affinity_score,
    load_model_checkpoint,
    get_preset_examples,
    get_benchmark_results_data,
)
from evaluation.shap_utils import compute_shap_attributions, get_top_feature_attributions

# Page Configuration
st.set_page_config(
    page_title="DTI-ML | Kinase Affinity Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for polished styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35em 0.8em;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 6px;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title Header
st.markdown('<div class="main-header">🧪 DTI-ML Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Kinase Drug-Target Binding Affinity Prediction & SHAP Feature Attribution (KIBA Score Benchmark)</div>',
    unsafe_allow_html=True,
)

# Navigation Tabs
tab_predict, tab_benchmarks, tab_gallery = st.tabs(
    ["🎯 Affinity Predictor & SHAP", "📊 Model Benchmarks (Cold Splits)", "🖼️ Showcase Gallery"]
)

presets = get_preset_examples()

# ==========================================
# TAB 1: PREDICTOR & SHAP EXPLAINER
# ==========================================
with tab_predict:
    st.markdown("### Input Drug & Target Parameters")
    
    # Preset selection
    selected_preset = st.selectbox(
        "⚡ Load Preset Showcase Example (Optional)",
        options=["-- Custom Input --"] + list(presets.keys()),
        help="Select a pre-configured drug-kinase pair to auto-fill inputs."
    )
    
    preset_smiles = ""
    preset_sequence = ""
    if selected_preset != "-- Custom Input --":
        preset_smiles = presets[selected_preset]["smiles"]
        preset_sequence = presets[selected_preset]["sequence"]
        st.info(f"Loaded preset: {presets[selected_preset]['description']}")

    col_drug, col_protein = st.columns(2)

    with col_drug:
        st.markdown("#### 💊 Small-Molecule Drug (SMILES)")
        smiles_input = st.text_area(
            "SMILES String",
            value=preset_smiles if preset_smiles else "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            height=120,
            placeholder="Enter SMILES sequence e.g. CC1=C...",
        )
        
        # 2D RDKit Structure Preview
        if smiles_input.strip():
            svg_text = render_molecule_svg(smiles_input.strip())
            if svg_text:
                st.markdown("##### 2D Structure Preview")
                st.image(svg_text, width=320)
            else:
                st.warning("⚠️ Invalid SMILES string. Unable to render 2D molecular structure.")

    with col_protein:
        st.markdown("#### 🧬 Protein Target Sequence")
        sequence_input = st.text_area(
            "Amino Acid Sequence",
            value=preset_sequence if preset_sequence else "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTIYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCMPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPAEGLKKPVLPATPKPQSAKEPSGTPISPTPVPSTLAAPAPAPLPPDSKPSMPPQLQPEREETEPASPSPPPPALPEAKPPRPEPPAPQPEPT",
            height=200,
            placeholder="Enter protein amino acid sequence e.g. MKT...",
        )
        if sequence_input.strip():
            st.caption(f"Sequence length: {len(sequence_input.strip())} amino acids")

    st.markdown("---")
    st.markdown("#### 🤖 Model Selection & Execution")
    
    col_model, col_btn = st.columns([3, 1])
    with col_model:
        selected_model = st.selectbox(
            "Select Machine Learning Model",
            options=[
                "Random Forest",
                "XGBoost",
                "Support Vector Regression (SVR)",
                "Gaussian Process Regression (GPR)",
            ],
            help="Choose between tree ensembles (RF/XGBoost) or kernel-based regressors (SVR/GPR).",
        )
        
    with col_btn:
        st.write("")
        st.write("")
        predict_button = st.button("🚀 Run Prediction", type="primary", use_container_width=True)

    if predict_button:
        if not smiles_input.strip() or not sequence_input.strip():
            st.error("Please provide both a valid SMILES string and a protein sequence.")
        else:
            with st.spinner("Computing features and running ML inference..."):
                model, is_mock = load_model_checkpoint(selected_model)
                
                from features.featurizer import featurize_single_pair, get_drug_feature_names, get_protein_feature_names
                
                # Featurize SMILES + sequence pair (1071 dims)
                sample_features = featurize_single_pair(smiles_input.strip(), sequence_input.strip())
                
                std_dev = None
                if is_mock:
                    st.toast("ℹ️ Using fallback baseline model.", icon="ℹ️")
                    affinity_score = model.predict(smiles_input.strip(), sequence_input.strip())
                    if hasattr(model, "predict_uncertainty"):
                        std_dev = model.predict_uncertainty(smiles_input.strip(), sequence_input.strip())
                else:
                    if "GPR" in selected_model:
                        try:
                            pred, std_arr = model.predict(sample_features.reshape(1, -1), return_std=True)
                            affinity_score = float(np.asarray(pred).flatten()[0])
                            std_dev = float(np.asarray(std_arr).flatten()[0])
                        except Exception:
                            pred = model.predict(sample_features.reshape(1, -1))
                            affinity_score = float(np.asarray(pred).flatten()[0])
                    else:
                        pred = model.predict(sample_features.reshape(1, -1))
                        affinity_score = float(np.asarray(pred).flatten()[0])

                label, color, explanation = interpret_affinity_score(affinity_score)
                
            st.markdown("---")
            st.markdown("### 📈 Prediction Results")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>Predicted KIBA Affinity Score</h4>
                        <h1 style="color: {color}; margin: 0;">{affinity_score:.3f}</h1>
                        <br>
                        <span class="badge" style="background-color: {color};">{label}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                if std_dev is not None:
                    st.info(f"🛡️ GPR Predictive Uncertainty (Std Dev): **±{std_dev:.3f}**")
                    
                st.caption(f"**Interpretation:** {explanation}")
                
            with res_col2:
                st.markdown("#### 🔍 SHAP Feature Attribution Breakdown")
                st.caption("Top molecular & protein descriptors driving this affinity prediction.")
                
                all_feature_names = get_drug_feature_names() + get_protein_feature_names()
                shap_vals = compute_shap_attributions(model, sample_features)
                df_shap = get_top_feature_attributions(all_feature_names, shap_vals[0], top_k=8)
                
                st.dataframe(
                    df_shap[["Feature", "SHAP Value", "Impact Type"]],
                    use_container_width=True,
                    hide_index=True,
                )
                
                # Bar chart visualization
                st.bar_chart(
                    data=df_shap.set_index("Feature")["SHAP Value"],
                    use_container_width=True,
                )

# ==========================================
# TAB 2: MODEL BENCHMARKS & COLD SPLITS
# ==========================================
with tab_benchmarks:
    st.markdown("### 📊 Classical ML Benchmark Comparison (12-Cell Matrix)")
    st.markdown(
        """
        Evaluation metrics across **Random**, **Cold-Drug**, and **Cold-Protein** split protocols.
        * **Random Split:** Standard baseline (vulnerable to similarity leakage).
        * **Cold-Drug Split:** Evaluates generalization on *unseen drug molecules*.
        * **Cold-Protein Split:** Evaluates generalization on *unseen kinase targets*.
        """
    )
    
    df_results = get_benchmark_results_data()
    
    # Filter by Split
    splits = list(df_results["Split"].unique())
    selected_splits = st.multiselect("Filter by Split Protocol", options=splits, default=splits)
    
    filtered_df = df_results[df_results["Split"].isin(selected_splits)]
    
    st.dataframe(
        filtered_df.style.highlight_max(axis=0, subset=["CI", "Pearson r"], color="#D1FAE5")
                         .highlight_min(axis=0, subset=["MSE", "RMSE"], color="#D1FAE5"),
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown("#### 💡 Key Insights")
    st.info(
        """
        1. **Generalization Gap:** Model metrics (CI and Pearson r) decrease significantly under Cold-Drug and Cold-Protein splits compared to Random splits, exposing similarity leakage in standard evaluations.
        2. **Algorithm Performance:** XGBoost and Random Forest consistently achieve higher Concordance Index (CI > 0.81 on random split) compared to SVR and GPR.
        """
    )

# ==========================================
# TAB 3: SHOWCASE GALLERY
# ==========================================
with tab_gallery:
    st.markdown("### 🖼️ Benchmark Kinase-Inhibitor Pair Gallery")
    st.caption("Showcase examples of FDA-approved targeted kinase therapies.")
    
    for title, details in presets.items():
        with st.expander(f"📌 {title}", expanded=True):
            col_img, col_info = st.columns([1, 2])
            with col_img:
                svg = render_molecule_svg(details["smiles"])
                if svg:
                    st.image(svg, width=280)
            with col_info:
                st.markdown(f"**Description:** {details['description']}")
                st.markdown(f"**SMILES:** `{details['smiles'][:60]}...`")
                st.markdown(f"**Sequence Length:** {len(details['sequence'])} amino acids")

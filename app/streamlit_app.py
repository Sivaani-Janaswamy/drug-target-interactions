import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Ensure root directory is accessible in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from app.helpers import (
    render_molecule_svg,
    interpret_affinity_score,
    load_model_checkpoint,
    get_preset_examples,
    get_benchmark_results_data,
)
from evaluation.shap_utils import compute_shap_attributions, get_top_feature_attributions
from app.chatbot import render_chat_widget

# Page Configuration
st.set_page_config(
    page_title="DTI-ML | Drug-Target Binding Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# DESIGN TOKENS (from approved mockup)
# ==========================================
# st.markdown() routes its body through a Markdown parser before rendering
# HTML (unsafe_allow_html only controls whether tags survive that pass, it
# doesn't skip Markdown parsing). Indented lines are Markdown's code-block
# syntax, so this indented CSS blob was rendered as literal text instead of
# a real <style> tag. st.html() does no Markdown parsing at all, so it's the
# correct API for pure CSS/HTML injection like this.
st.html(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
    :root{
        --ink:#12201F; --paper:#F4F7F5; --paper-alt:#EAEFEA; --panel:#FFFFFF;
        --line:#D7DED9; --teal:#1F6F63; --teal-deep:#0F4A41; --amber:#DE9A34;
        --amber-deep:#B87A1F; --graphite:#5B655F; --danger:#B84B3A;
    }
    html, body, [class*="css"]  { font-family: 'Inter', sans-serif; color: var(--ink); }
    .stApp { background: var(--paper); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing:-0.01em; color: var(--ink); }
    .mono, code { font-family: 'IBM Plex Mono', monospace !important; }

    .eyebrow{
        font-family:'IBM Plex Mono'; font-size:11.5px; letter-spacing:0.08em; text-transform:uppercase;
        color:var(--teal-deep); font-weight:500; margin-bottom:6px; display:flex; align-items:center; gap:8px;
    }
    .eyebrow::before{content:""; width:16px; height:1px; background:var(--amber-deep);}
    .lede{font-size:16px; color:var(--graphite); max-width:680px; margin-top:6px;}

    .card{
        background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:22px;
        margin-bottom:14px;
    }
    .card h3{font-size:16px; margin:0 0 8px 0;}
    .card p{font-size:13.5px; color:var(--graphite); margin:0;}
    .analogy{
        margin-top:10px; font-size:12.5px; padding:9px 11px; background:var(--paper-alt);
        border-left:2px solid var(--amber-deep); border-radius:0 6px 6px 0; color:var(--ink);
    }
    .analogy b{color:var(--teal-deep);}

    .pstep{
        background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px 18px;
    }
    .pstep .num{font-family:'IBM Plex Mono'; font-size:11px; color:var(--amber-deep); font-weight:500;}
    .pstep h3{font-size:14.5px; margin-top:6px;}
    .pstep p{font-size:12.5px; color:var(--graphite); margin-top:4px;}

    .gauge-wrap{background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:26px; margin-top:10px;}
    .gauge-score{font-family:'Space Grotesk'; font-size:36px; font-weight:700;}
    .gauge-label{
        font-size:13px; font-weight:600; padding:4px 12px; border-radius:20px; background:#FCEFDD; color:var(--amber-deep);
        display:inline-block;
    }
    .gauge-track{position:relative; height:14px; background:linear-gradient(90deg,#8FBBB0,#1F6F63 40%,#DE9A34 75%,#B84B3A); border-radius:20px; margin:20px 0 8px;}
    .gauge-marker{position:absolute; top:-8px; width:3px; height:30px; background:var(--ink); border-radius:2px; transition:left 0.9s cubic-bezier(.2,.85,.3,1);}
    .gauge-scale{display:flex; justify-content:space-between; font-size:11px; color:var(--graphite); font-family:'IBM Plex Mono';}

    .reason{
        display:flex; gap:12px; padding:12px 14px; border:1px solid var(--line); border-radius:10px;
        align-items:flex-start; margin-bottom:10px; background:var(--panel);
    }
    .reason .bar{width:6px; align-self:stretch; border-radius:4px; flex-shrink:0;}
    .reason .bar.pos{background:var(--teal);}
    .reason .bar.neg{background:var(--danger);}
    .reason .rtext{font-size:13px;}
    .reason .rtext b{font-family:'IBM Plex Mono'; font-size:12.5px;}
    .reason .rtext span{display:block; color:var(--graphite); margin-top:2px;}

    .mini-note{
        display:flex; gap:10px; align-items:flex-start; padding:12px 14px; background:var(--paper-alt);
        border-radius:8px; font-size:12.5px; color:var(--graphite); margin-top:14px;
    }

    .member{background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:16px; text-align:left;}
    .avatar{width:34px; height:34px; border-radius:50%; background:var(--teal-deep); color:#fff; display:flex; align-items:center; justify-content:center; font-family:'Space Grotesk'; font-weight:600; margin-bottom:8px;}
    .member h3{font-size:13.5px; margin:0;}
    .member p{font-size:12px; color:var(--graphite); margin-top:4px;}

    .split-tag{font-family:'IBM Plex Mono'; font-size:10.5px; padding:2px 7px; border-radius:10px; background:var(--paper-alt); color:var(--graphite);}

    /* Pipeline step connector arrow, matching the mockup's .pstep::after */
    .pstep{ position:relative; }
    div[data-testid="column"]:not(:last-child) .pstep::after{
        content:"→"; position:absolute; right:-19px; top:50%; transform:translateY(-50%);
        color:var(--amber-deep); font-size:15px; z-index:2;
    }

    /* ---- Native Streamlit widget re-skin (CSS only, same widgets/behavior) ---- */
    section[data-testid="stSidebar"] { background: var(--paper-alt); }

    /* Sidebar nav radio -> pill-style links, mirroring the mockup's .navlink */
    section[data-testid="stSidebar"] div[role="radiogroup"] { gap:4px; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label{
        border-radius:20px; padding:8px 13px; font-size:13.5px; font-weight:500;
        color:var(--graphite); border:1px solid transparent; transition:all .15s ease;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
        color:var(--ink); background:var(--panel);
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
    section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
        color:var(--teal-deep); background:#E4EFEA; border-color:#CFE2D9;
    }
    section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{ display:none; }

    /* Buttons -> mockup's .btn-primary / .btn-ghost look */
    .stButton > button{
        border-radius:8px; font-size:14.5px; font-weight:600; padding:12px 22px;
        border:1px solid var(--line); transition:transform .12s ease, box-shadow .12s ease;
    }
    .stButton > button:active{ transform:scale(0.98); }
    .stButton > button[kind="primary"]{
        background:var(--teal-deep); color:#fff; border-color:var(--teal-deep);
    }
    .stButton > button[kind="primary"]:hover{ background:var(--teal); border-color:var(--teal); }
    .stButton > button[kind="secondary"]{ background:transparent; color:var(--ink); }
    .stButton > button[kind="secondary"]:hover{ border-color:var(--graphite); }

    /* Text inputs / textareas / selects -> mockup's mono field style */
    .stTextArea textarea, .stTextInput input{
        border-radius:8px; border:1px solid var(--line); background:var(--paper);
        font-family:'IBM Plex Mono', monospace; color:var(--ink);
    }
    div[data-testid="stSelectbox"] > div{ border-radius:8px; }

    /* Dataframe header row, matching .card/table look */
    div[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }
    </style>
    """
)

presets = get_preset_examples()

# ==========================================
# NAVIGATION (mockup's 6 sections)
# ==========================================
PAGES = ["Home", "How it works", "Try the predictor", "Sample result", "Model & science", "About"]

if "page" not in st.session_state:
    st.session_state.page = "Home"
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.markdown(
        "<div style='font-family:Space Grotesk; font-weight:700; font-size:18px;'>DTI‑ML</div>"
        "<div style='font-family:IBM Plex Mono; font-size:11px; color:var(--graphite);'>binding affinity predictor</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    choice = st.radio("Navigate", PAGES, index=PAGES.index(st.session_state.page), label_visibility="collapsed")
    st.session_state.page = choice

page = st.session_state.page

_chat_context = None
if st.session_state.last_result:
    _r = st.session_state.last_result
    _chat_context = {
        "score": _r["score"],
        "label": _r["label"],
        "top_features": [
            {"name": f["Feature"], "direction": "pos" if f["SHAP Value"] > 0 else "neg", "weight": f["SHAP Value"]}
            for f in _r["top_features"]
        ],
    }
render_chat_widget(_chat_context)

# ==========================================
# HOME
# ==========================================
if page == "Home":
    hero_col1, hero_col2 = st.columns([1.1, 0.9])
    with hero_col1:
        st.markdown("## Predicting whether a drug *fits* its target — before it's ever tested in a lab.")
        st.markdown(
            '<p class="lede">Feed it a drug and a protein. It predicts how tightly they\'d bind — tested only on pairs it\'s never seen before.</p>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([1, 1])
        with c1:
            if st.button("Try a prediction →", type="primary"):
                st.session_state.page = "Try the predictor"
                st.rerun()
        with c2:
            if st.button("What does this actually mean?"):
                st.session_state.page = "How it works"
                st.rerun()
    with hero_col2:
        # Decorative lock-and-key visual from the approved mockup — purely
        # illustrative, no interactivity or logic.
        st.html(
            """
            <div style="background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:20px;">
            <svg viewBox="0 0 380 220" style="width:100%; height:auto; display:block;">
              <style>
                @keyframes dtiFloatHero{0%,100%{transform:translateY(0);}50%{transform:translateY(-6px);}}
                @keyframes dtiWiggleHero{0%,100%{transform:rotate(0deg);}25%{transform:rotate(-3deg);}75%{transform:rotate(3deg);}}
                @keyframes dtiDashHero{to{stroke-dashoffset:-24;}}
                .dti-hero-key{animation:dtiFloatHero 3.2s ease-in-out infinite;}
                .dti-hero-kinase{animation:dtiWiggleHero 4.5s ease-in-out infinite; transform-origin:290px 110px;}
                .dti-hero-path{stroke-dasharray:6 6; animation:dtiDashHero 1.4s linear infinite;}
              </style>
              <g class="dti-hero-key">
                <circle cx="95" cy="110" r="46" fill="#E4EFEA" stroke="#1F6F63" stroke-width="2"/>
                <path d="M75 96 l14 14 -14 14" stroke="#1F6F63" stroke-width="4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
                <circle cx="95" cy="110" r="9" fill="#1F6F63"/>
              </g>
              <text x="95" y="172" text-anchor="middle" font-family="IBM Plex Mono" font-size="11" fill="#5B655F">drug (SMILES)</text>
              <path class="dti-hero-path" d="M155 110 h60" stroke="#DE9A34" stroke-width="2"/>
              <text x="185" y="98" text-anchor="middle" font-family="Space Grotesk" font-size="11" fill="#B87A1F">binds?</text>
              <g class="dti-hero-kinase">
                <rect x="245" y="70" width="90" height="80" rx="12" fill="#FCEFDD" stroke="#DE9A34" stroke-width="2"/>
                <path d="M270 110 q20 -26 40 0 q-20 26 -40 0 z" fill="none" stroke="#B87A1F" stroke-width="2.5"/>
              </g>
              <text x="290" y="172" text-anchor="middle" font-family="IBM Plex Mono" font-size="11" fill="#5B655F">kinase (protein)</text>
              <text x="190" y="30" text-anchor="middle" font-family="Space Grotesk" font-weight="600" font-size="13" fill="#12201F">Affinity score = how tightly it fits</text>
            </svg>
            </div>
            """
        )

    # Educational drug-kinase binding illustration — pure HTML/CSS (no <svg>,
    # which Streamlit's st.html() sanitizer strips entirely with no way to
    # allow it) — purely decorative, no interactivity or logic.
    st.html(
        """
        <div style="background:var(--panel); border:1px solid var(--line); border-radius:16px;
                    padding:32px 24px; margin-top:24px; text-align:center;">
          <h3 style="font-family:'Space Grotesk',sans-serif; font-size:17px; font-weight:600;
                     color:var(--ink); margin:0 0 28px 0;">Affinity score = how tightly it fits</h3>
          <div style="display:flex; align-items:center; justify-content:center; gap:18px; flex-wrap:wrap;">
            <div style="display:flex; flex-direction:column; align-items:center; gap:12px;">
              <div style="width:110px; height:110px; border-radius:50%; background:#E4EFEA;
                          border:2px solid #1F6F63; display:flex; align-items:center; justify-content:center;">
                <div style="width:22px; height:22px; border-radius:50%; background:#1F6F63; position:relative;">
                  <div style="position:absolute; width:26px; height:3px; background:#1F6F63; top:9.5px; left:-24px; transform:rotate(35deg); border-radius:2px;"></div>
                </div>
              </div>
              <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--graphite);">drug (SMILES)</span>
            </div>
            <div style="display:flex; flex-direction:column; align-items:center; gap:6px; padding-bottom:28px;">
              <span style="font-family:'Space Grotesk',sans-serif; font-size:13px; color:var(--amber-deep); font-weight:600;">binds?</span>
              <div style="width:70px; height:0; border-top:2px dashed var(--amber-deep);"></div>
            </div>
            <div style="display:flex; flex-direction:column; align-items:center; gap:12px;">
              <div style="width:110px; height:110px; border-radius:22px; background:#FCEFDD;
                          border:2px solid #DE9A34; display:flex; align-items:center; justify-content:center;">
                <div style="width:34px; height:20px; border:2px solid #B87A1F; border-radius:50% 50% 50% 50% / 90% 90% 10% 10%;"></div>
              </div>
              <span style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:var(--graphite);">kinase (protein)</span>
            </div>
          </div>
          <p style="font-size:13.5px; color:var(--graphite); max-width:520px; margin:24px auto 0;">
            The model estimates how strongly a drug molecule can bind to a kinase protein.
            Higher affinity generally indicates a stronger interaction.
          </p>
        </div>
        """
    )

    st.markdown("### The pipeline, in four steps")
    cols = st.columns(4)
    steps = [
        ("01", "Read the drug & protein", "The drug is written as a text code (SMILES); the protein as its amino-acid sequence. No 3D lab data needed."),
        ("02", "Turn them into numbers", "Chemistry software converts each into structured features — shape, composition, physical properties."),
        ("03", "Ask the model", "Four different ML algorithms independently estimate how strong the interaction is."),
        ("04", "Explain the answer", "The tool shows not just a score, but which features pushed it up or down."),
    ]
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f'<div class="pstep"><div class="num">{num}</div><h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Why this matters")
    cols = st.columns(3)
    cards = [
        ("Faster than lab testing", "Screening a candidate drug computationally takes seconds. Testing it physically can take weeks and real money."),
        ("Honestly evaluated", "We test on drugs and proteins the model has never seen before — not just shuffled copies of its training data."),
        ("Not a black box", "Every prediction comes with a plain-language reason, so the \"why\" is never hidden behind the score."),
    ]
    for col, (title, desc) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="card"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

# ==========================================
# HOW IT WORKS
# ==========================================
elif page == "How it works":
    st.markdown('<div class="eyebrow">Plain-language explainer</div>', unsafe_allow_html=True)
    st.markdown("## Five terms this project relies on — explained without the jargon")
    st.markdown(
        '<p class="lede">You don\'t need a biology background to follow this. Here\'s every technical term on this site, explained once, properly.</p>',
        unsafe_allow_html=True,
    )

    terms = [
        ("What's a kinase?",
         "A kinase is a protein that acts like a molecular <b>switch</b> — it turns other proteins on or off inside a cell. When kinases misfire, cells can grow uncontrollably, which is why they're one of the most important drug targets in cancer treatment.",
         "a light switch. The wrong drug flips it the wrong way; the right drug holds it exactly where you want it."),
        ("What's \"binding affinity\"?",
         "It's a number describing how strongly a drug molecule sticks to its target protein. Higher affinity generally means the drug is more effective at that target — and less likely to need a huge dose.",
         "how snugly a key fits a lock. A loose fit barely turns it; a snug fit works reliably."),
        ("Why \"cold-split\" evaluation?",
         "Most simple tests let a model see similar drugs or proteins during both training and testing — which makes it look smarter than it is. A cold split removes that overlap entirely, so the reported accuracy reflects real-world performance on drugs it has genuinely never encountered.",
         "testing a student on questions from a totally different textbook, not the one they studied from."),
        ("What is SHAP, and why show it?",
         "SHAP is a method that scores how much each input feature (a molecule shape, a protein property) pushed the final prediction up or down. Instead of trusting the model blindly, you can see exactly what it \"noticed.\"",
         "a teacher showing their working, not just the final answer on the exam."),
    ]
    cols = st.columns(2)
    for i, (title, body, analogy) in enumerate(terms):
        with cols[i % 2]:
            st.markdown(
                f'<div class="card"><h3>{title}</h3><p>{body}</p>'
                f'<div class="analogy"><b>Think of it as:</b> {analogy}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### The four models we compare")
    st.markdown(
        '<p class="lede">Rather than picking one algorithm and hoping it\'s right, this project trains four different classical ML models on the exact same data, so their strengths and weaknesses can be compared directly.</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    models_info = [
        ("Random Forest", "Many decision trees voting together. Robust, hard to fool, good default baseline."),
        ("XGBoost", "Trees built one after another, each correcting the last one's mistakes. Usually the most accurate."),
        ("Support Vector Regression", "Finds the smoothest boundary that still respects the data. Good with smaller, cleaner feature sets."),
        ("Gaussian Process", "Slower, but tells you how confident it is in each prediction — not just the number itself."),
    ]
    for col, (title, desc) in zip(cols, models_info):
        with col:
            st.markdown(f'<div class="card"><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

# ==========================================
# TRY THE PREDICTOR
# ==========================================
elif page == "Try the predictor":
    st.markdown('<div class="eyebrow">Interactive demo</div>', unsafe_allow_html=True)
    st.markdown("## Try a prediction")
    st.markdown(
        '<p class="lede">Pick a real example, or paste your own. No chemistry knowledge needed — the dropdowns give you known drugs and kinases to start with.</p>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        selected_preset = st.selectbox(
            "⚡ Load preset example (optional)",
            options=["-- Custom Input --"] + list(presets.keys()),
            help="Select a pre-configured drug-kinase pair from our example dataset to auto-fill inputs.",
        )

        preset_smiles, preset_sequence = "", ""
        if selected_preset != "-- Custom Input --":
            preset_smiles = presets[selected_preset]["smiles"]
            preset_sequence = presets[selected_preset]["sequence"]
            st.info(f"Loaded: {presets[selected_preset]['description']}")

        smiles_input = st.text_area(
            "Drug molecule — SMILES string",
            value=preset_smiles if preset_smiles else "CC(=O)OC1=CC=CC=C1C(=O)O",
            height=100,
        )
        if smiles_input.strip():
            svg_text = render_molecule_svg(smiles_input.strip())
            if svg_text:
                st.image(svg_text, width=280)
            else:
                st.warning("⚠️ Invalid SMILES string.")

        sequence_input = st.text_area(
            "Target protein — kinase sequence",
            value=preset_sequence if preset_sequence else "",
            height=140,
            placeholder="e.g. MKTAYIAKQR…",
        )
        if sequence_input.strip():
            st.caption(f"Sequence length: {len(sequence_input.strip())} amino acids")

        selected_model = st.selectbox(
            "Model",
            options=["Random Forest", "XGBoost", "Support Vector Regression (SVR)", "Gaussian Process Regression (GPR)"],
        )

        run = st.button("Predict binding affinity →", type="primary", use_container_width=True)

        st.markdown(
            '<div class="mini-note">ℹ️ This model was trained on the KIBA benchmark dataset (~118K measured drug–kinase pairs) '
            'and tested on kinases and drugs it never saw during training.</div>',
            unsafe_allow_html=True,
        )

        if run:
            if not smiles_input.strip() or not sequence_input.strip():
                st.error("Please provide both a valid SMILES string and a protein sequence.")
            else:
                with st.spinner("Computing features and running ML inference..."):
                    model, is_mock = load_model_checkpoint(selected_model)
                    from features.featurizer import featurize_single_pair, get_drug_feature_names, get_protein_feature_names

                    sample_features = featurize_single_pair(smiles_input.strip(), sequence_input.strip())

                    std_dev = None
                    if is_mock:
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

                    all_feature_names = get_drug_feature_names() + get_protein_feature_names()
                    shap_vals = compute_shap_attributions(model, sample_features)
                    df_shap = get_top_feature_attributions(all_feature_names, shap_vals[0], top_k=4)

                    # Map score to gauge percent: 8.5 -> weak(0%), 14.5 -> strong(100%), clamp
                    gauge_pct = float(np.clip((affinity_score - 8.5) / (14.5 - 8.5) * 100, 0, 100))

                    st.session_state.last_result = {
                        "score": affinity_score,
                        "label": label,
                        "color": color,
                        "explanation": explanation,
                        "std_dev": std_dev,
                        "gauge_pct": gauge_pct,
                        "top_features": df_shap.to_dict("records"),
                        "drug_name": selected_preset if selected_preset != "-- Custom Input --" else "custom drug",
                        "target_name": selected_preset if selected_preset != "-- Custom Input --" else "custom target",
                    }
                st.success("Prediction complete — see the Sample result page.")
                st.session_state.page = "Sample result"
                st.rerun()

    with col_right:
        st.markdown("#### What happens after you click predict")
        steps = [
            "Your SMILES and sequence are converted into the same numeric features the model was trained on.",
            "The selected model estimates a KIBA affinity score for this exact pair.",
            "SHAP identifies which specific features drove that score up or down.",
            "You get a plain-language readout — not just a raw number.",
        ]
        for i, s in enumerate(steps, 1):
            st.markdown(f"**{i}.** {s}")

# ==========================================
# SAMPLE RESULT
# ==========================================
elif page == "Sample result":
    st.markdown('<div class="eyebrow">Prediction output</div>', unsafe_allow_html=True)

    result = st.session_state.last_result
    if result is None:
        st.markdown("## No prediction yet")
        st.markdown(
            '<p class="lede">Run a prediction on the "Try the predictor" page to see a real result here.</p>',
            unsafe_allow_html=True,
        )
        if st.button("Go to Try the predictor →", type="primary"):
            st.session_state.page = "Try the predictor"
            st.rerun()
    else:
        st.markdown(f"## Prediction result")
        st.markdown(
            '<p class="lede">This is a live prediction from the model you ran, with the score, what it means in plain terms, and why the model reached it.</p>',
            unsafe_allow_html=True,
        )

        gauge_pct = result["gauge_pct"]
        # st.html (not st.markdown) — multi-line indented HTML passed to
        # st.markdown gets misread as a Markdown code block (see note at the
        # top design-tokens block) and rendered as literal text.
        st.html(
            f"""
            <div class="gauge-wrap">
              <div style="display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:10px;">
                <div class="gauge-score">{result['score']:.2f} <span style="font-size:14px; color:var(--graphite); font-weight:500;">predicted KIBA score</span></div>
                <div class="gauge-label">{result['label']}</div>
              </div>
              <div class="gauge-track"><div class="gauge-marker" style="left:{gauge_pct:.1f}%;"></div></div>
              <div class="gauge-scale"><span>Weak</span><span>Moderate</span><span>Strong</span></div>
              <p style="font-size:13px; color:var(--graphite); margin-top:16px;">{result['explanation']}</p>
            </div>
            """
        )
        if result.get("std_dev") is not None:
            st.info(f"🛡️ GPR predictive uncertainty (std dev): ±{result['std_dev']:.3f}")

        st.markdown("### Why the model said this")
        for f in result["top_features"]:
            direction = "pos" if f["SHAP Value"] > 0 else "neg"
            sign = "+" if f["SHAP Value"] > 0 else "−"
            st.html(
                f"""
                <div class="reason">
                  <div class="bar {direction}"></div>
                  <div class="rtext"><b>{sign}{abs(f['SHAP Value']):.2f}&nbsp;&nbsp;{f['Feature']}</b>
                  <span>{f['Impact Type']}</span></div>
                </div>
                """
            )

        st.markdown(
            '<div class="mini-note">ℹ️ Reasons are generated using SHAP values mapped to feature names from our featurizer — '
            'plain-language descriptions per feature are a placeholder pending a human-readable feature-name mapping '
            '(see note below).</div>',
            unsafe_allow_html=True,
        )

# ==========================================
# MODEL & SCIENCE
# ==========================================
elif page == "Model & science":
    st.markdown('<div class="eyebrow">For evaluators & the curious</div>', unsafe_allow_html=True)
    st.markdown("## How the models actually compare")
    st.markdown(
        '<p class="lede">Full results across all four algorithms, reported honestly across three evaluation splits — read from our actual saved metrics.</p>',
        unsafe_allow_html=True,
    )

    df_results = get_benchmark_results_data()
    splits = list(df_results["Split"].unique())
    selected_splits = st.multiselect("Filter by split protocol", options=splits, default=splits)
    filtered_df = df_results[df_results["Split"].isin(selected_splits)]

    st.dataframe(
        filtered_df.style.highlight_max(axis=0, subset=["CI", "Pearson r"], color="#D1FAE5")
                         .highlight_min(axis=0, subset=["MSE", "RMSE"], color="#D1FAE5"),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### The honest headline")
    st.markdown(
        '<div class="card"><h3>Every model gets worse on cold splits — and that\'s the finding, not a flaw</h3>'
        '<p>All four algorithms score noticeably lower once drug or protein overlap is removed between train and test. '
        'That gap is the real story: it shows how much of a "good-looking" result on a random split was actually just '
        'memorized similarity, not genuine generalization.</p></div>',
        unsafe_allow_html=True,
    )

# ==========================================
# ABOUT
# ==========================================
elif page == "About":
    st.markdown("## About this project")
    st.markdown(
        '<p class="lede">DTI–ML predicts kinase drug–target binding affinity using classical machine learning, '
        'evaluated under a cold-split protocol and explained with SHAP feature attribution, built on the public KIBA benchmark.</p>',
        unsafe_allow_html=True,
    )

    cols = st.columns(4)
    team = [
        ("A", "Data & Drug Features", "Dataset cleaning, split design, fingerprint & descriptor pipeline."),
        ("B", "Protein Features & Models", "Protein feature pipeline, training and tuning all four algorithms."),
        ("C", "Evaluation & Interpretability", "Metrics, SHAP analysis, pharmacophore sanity checks."),
        ("D", "App & Paper", "Web app build, IEEE paper compilation, presentation."),
    ]
    for col, (letter, title, desc) in zip(cols, team):
        with col:
            st.markdown(
                f'<div class="member"><div class="avatar">{letter}</div><h3>{title}</h3><p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### References this project builds on")
    st.markdown(
        '<p class="lede">Positioned against DeepDTA, GraphDTA, KronRLS, and SimBoost — comparing classical, '
        'feature-engineered ML against deep-learning DTI approaches on the same benchmark.</p>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("DTI–ML")

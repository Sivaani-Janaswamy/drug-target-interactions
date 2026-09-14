from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.helpers import get_benchmark_results_data, get_preset_examples, interpret_affinity_score, load_model_checkpoint
from backend.config import get_settings
from evaluation.shap_utils import compute_shap_attributions, get_top_feature_attributions
from features.featurizer import (
    compute_drug_features,
    get_drug_feature_names,
    get_protein_feature_names,
    featurize_single_pair,
    get_morgan_bit_context,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = get_settings().models_dir
SUPPORTED_MODELS = (
    "Random Forest",
    "XGBoost",
    "Support Vector Regression (SVR)",
    "Gaussian Process Regression (GPR)",
)
MODEL_LABELS = {
    "Random Forest": ("Random Forest", "balanced, fast"),
    "XGBoost": ("XGBoost", "highest accuracy"),
    "Support Vector Regression (SVR)": ("SVR", "smooth estimates"),
    "Gaussian Process Regression (GPR)": ("Gaussian Process", "gives confidence range"),
}


def validate_pair(smiles: str, protein_sequence: str, model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model. Choose one of: {', '.join(SUPPORTED_MODELS)}")
    if not smiles or not smiles.strip():
        raise ValueError("A drug SMILES string is required.")
    if compute_drug_features(smiles.strip()) is None:
        raise ValueError("The SMILES string is invalid or could not be parsed.")
    sequence = "".join(protein_sequence.split()).upper()
    if not sequence:
        raise ValueError("A protein sequence is required.")
    invalid = sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"))
    if invalid:
        raise ValueError(f"Protein sequence contains invalid amino-acid symbols: {', '.join(invalid)}")


@lru_cache(maxsize=len(SUPPORTED_MODELS))
def get_cached_model(model_name: str) -> tuple[Any, bool]:
    return load_model_checkpoint(model_name, str(MODELS_DIR))


def _predict(model: Any, is_mock: bool, model_name: str, smiles: str, sequence: str, features: np.ndarray) -> tuple[float, float | None]:
    if is_mock:
        score = model.predict(smiles, sequence)
        uncertainty = model.predict_uncertainty(smiles, sequence) if hasattr(model, "predict_uncertainty") else None
        return float(score), uncertainty

    if "GPR" in model_name:
        try:
            prediction, standard_deviation = model.predict(features.reshape(1, -1), return_std=True)
            return float(np.asarray(prediction).flatten()[0]), float(np.asarray(standard_deviation).flatten()[0])
        except Exception:
            pass

    prediction = model.predict(features.reshape(1, -1))
    return float(np.asarray(prediction).flatten()[0]), None


def _human_feature(feature_name: str) -> str:
    if feature_name.startswith("Morgan_Bit_"):
        return "Molecular substructure fingerprint"
    if feature_name == "Desc_MolWt":
        return "Molecular weight"
    if feature_name == "Desc_MolLogP":
        return "Molecular lipophilicity"
    if feature_name == "Desc_TPSA":
        return "Polar surface area"
    if feature_name == "Desc_NumHDonors":
        return "Hydrogen bond donors"
    if feature_name == "Desc_NumHAcceptors":
        return "Hydrogen bond acceptors"
    if feature_name == "Desc_NumRotatableBonds":
        return "Rotatable bonds"
    if feature_name.startswith("AAC_"):
        return f"Amino-acid composition ({feature_name[-1]})"
    if feature_name.startswith("CTD_"):
        return "Protein composition profile"
    return feature_name.replace("_", " ")


def _feature_explanation(label: str, value: float) -> str:
    direction = "pushed the score up" if value > 0 else "pulled the score down"
    return f"{label} {direction} in this prediction."


def predict_binding(
    smiles: str,
    protein_sequence: str,
    model_name: str,
    drug_name: str = "Custom drug",
    target_name: str = "Custom target",
) -> dict[str, Any]:
    smiles = smiles.strip()
    sequence = "".join(protein_sequence.split()).upper()
    validate_pair(smiles, sequence, model_name)
    model, is_mock = get_cached_model(model_name)
    features = featurize_single_pair(smiles, sequence)
    score, uncertainty = _predict(model, is_mock, model_name, smiles, sequence, features)
    label, color, explanation = interpret_affinity_score(score)
    feature_names = get_drug_feature_names() + get_protein_feature_names()
    shap_values = compute_shap_attributions(model, features)
    top_features = get_top_feature_attributions(feature_names, shap_values[0], top_k=4)

    bit_context = get_morgan_bit_context(smiles, [row["Feature"] for row in top_features.to_dict("records")])
    feature_payload = []
    for row in top_features.to_dict("records"):
        value = float(row["SHAP Value"])
        human_name = _human_feature(row["Feature"])
        feature_payload.append(
            {
                "name": human_name,
                "technical_name": row["Feature"],
                "value": value,
                "direction": "positive" if value > 0 else "negative",
                "explanation": _feature_explanation(human_name, value),
                "chemical_context": bit_context.get(row["Feature"]),
            }
        )

    gauge_percent = float(np.clip((score - 8.5) / 6.0 * 100.0, 0, 100))
    return {
        "drug": {"name": drug_name, "smiles": smiles},
        "target": {"name": target_name, "sequence_length": len(sequence)},
        "model": model_name,
        "score": round(score, 3),
        "label": label,
        "color": color,
        "explanation": explanation,
        "gauge_percent": round(gauge_percent, 1),
        "uncertainty": uncertainty,
        "features": feature_payload,
    }


def get_api_presets() -> list[dict[str, Any]]:
    presets = []
    for key, value in get_preset_examples().items():
        drug_target, _, target_detail = key.partition(" + ")
        target_name = target_detail.split(" (")[0] if target_detail else "Custom target"
        presets.append(
            {
                "id": key,
                "drug_name": drug_target,
                "target_name": target_name,
                "smiles": value["smiles"],
                "sequence": value["sequence"],
                "description": value["description"],
            }
        )
    presets.insert(
        0,
        {
            "id": "aspirin-abl1",
            "drug_name": "Aspirin",
            "target_name": "ABL1",
            "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
            "sequence": "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTIYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCMPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPAEGLKKPVLPATPKPQSAKEPSGTPISPTPVPSTLAAPAPAPLPPDSKPSMPPQLQPEREETEPASPSPPPPALPEAKPPRPEPPAPQPEPT",
            "description": "Aspirin paired with ABL1 for the sample result flow.",
        },
    )
    return presets


def get_api_benchmarks() -> list[dict[str, Any]]:
    return get_benchmark_results_data(str(MODELS_DIR)).to_dict("records")

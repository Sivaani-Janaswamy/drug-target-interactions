"""
Reusable utilities for the DTI-ML prediction API and frontend.

Provides:
- RDKit 2D molecular structure rendering
- Preset drug-kinase showcase examples
- Model loader with graceful fallback/mock predictor
- Plain-language affinity score interpreter
- Benchmark results matrix formatter for Model UI
"""

import os
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


def render_molecule_svg(smiles: str) -> Optional[str]:
    """
    Render 2D molecular structure SVG string from SMILES using RDKit.
    Returns SVG string if valid SMILES, otherwise None.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import rdMolDraw2D
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        drawer = rdMolDraw2D.MolDraw2DSVG(350, 250)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception:
        return None


def interpret_affinity_score(score: float) -> Tuple[str, str, str]:
    """
    Interpret KIBA affinity score into qualitative binding level.
    Returns (label, color_code, explanation)
    Higher KIBA scores correspond to stronger binding affinity.
    """
    if score >= 12.0:
        return (
            "Strong Binding Affinity",
            "#10B981", # emerald green
            "High predicted affinity (KIBA score ≥ 12.0). Candidate shows strong target engagement."
        )
    elif score >= 10.0:
        return (
            "Moderate Binding Affinity",
            "#F59E0B", # amber
            "Moderate predicted affinity (10.0 ≤ KIBA score < 12.0). Candidate exhibits moderate potency."
        )
    else:
        return (
            "Weak Binding Affinity",
            "#EF4444", # red
            "Low predicted affinity (KIBA score < 10.0). Weak target binding predicted."
        )


class MockPredictor:
    """
    Fallback predictor used when trained model checkpoints in models/ do not exist yet.
    Generates deterministic, realistic KIBA affinity predictions and synthetic feature importances.
    """
    def __init__(self, model_name: str = "Random Forest"):
        self.model_name = model_name

    def predict(self, smiles: str, sequence: str) -> float:
        # Seed pseudo-random generator deterministically based on input strings
        seed = (sum(ord(c) for c in smiles) + sum(ord(c) for c in sequence)) % 10000
        np.random.seed(seed)
        
        # Base KIBA score centered around 11.0 with variation
        base_score = 11.2 + (np.sin(len(smiles)) * 0.8) + (np.cos(len(sequence)) * 0.6)
        
        # Add slight algorithm variation
        algo_offset = {
            "Random Forest": 0.0,
            "XGBoost": 0.15,
            "Support Vector Regression (SVR)": -0.10,
            "Gaussian Process Regression (GPR)": 0.05,
        }.get(self.model_name, 0.0)
        
        score = float(np.clip(base_score + algo_offset, 8.5, 14.5))
        return round(score, 3)

    def predict_uncertainty(self, smiles: str, sequence: str) -> float:
        """Uncertainty standard deviation (mainly for GPR model)."""
        seed = (sum(ord(c) for c in smiles) * 3) % 10000
        np.random.seed(seed)
        return round(float(np.random.uniform(0.15, 0.45)), 3)


import pickle

def load_model_checkpoint(model_name: str, models_dir: str = "./models"):
    """
    Attempt to load trained model checkpoint from models_dir.
    If checkpoint file does not exist, return MockPredictor.
    """
    filename_map = {
        "Random Forest": "rf_model.pkl",
        "XGBoost": "xgb_model.pkl",
        "Support Vector Regression (SVR)": "svr_model.pkl",
        "Gaussian Process Regression (GPR)": "gpr_model.pkl",
    }
    
    filename = filename_map.get(model_name, "rf_model.pkl")
    filepath = os.path.join(models_dir, filename)
    
    if os.path.exists(filepath):
        try:
            with open(filepath, "rb") as f:
                model = pickle.load(f)
            return model, False # Real model loaded
        except Exception:
            pass
            
    if HAS_JOBLIB and os.path.exists(filepath):
        try:
            model = joblib.load(filepath)
            return model, False # Real model loaded
        except Exception:
            pass
            
    return MockPredictor(model_name), True # Mock predictor fallback


def get_preset_examples() -> Dict[str, Dict[str, str]]:
    """
    Pre-configured kinase-drug showcase examples for quick UI selection.
    """
    return {
        "Imatinib + ABL1 (Leukemia Target)": {
            "smiles": "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            "sequence": "MLEICLKLVGCKSKKGLSSSSSCYLEEALQRPVASDFEPQGLSEAARWNSKENLLAGPSENDPNLFVALYDFVASGDNTLSITKGEKLRVLGYNHNGEWCEAQTKNGQGWVPSNYITPVNSLEKHSWYHGPVSRNAAEYLLSSGINGSFLVRESESSPGQRSISLRYEGRVYHYRINTASDGKLYVSSESRFNTLAELVHHHSTVADGLITTLHYPAPKRNKPTIYGVSPNYDKWEMERTDITMKHKLGGGQYGEVYEGVWKKYSLTVAVKTLKEDTMEVEEFLKEAAVMKEIKHPNLVQLLGVCTREPPFYIITEFMTYGNLLDYLRECNRQEVNAVVLLYMATQISSAMEYLEKKNFIHRDLAARNCLVGENHLVKVADFGLSRLMTGDTYTAHAGAKFPIKWTAPESLAYNKFSIKSDVWAFGVLLWEIATYGMSPYPGIDLSQVYELLEKDYRMERPEGCPEKVYELMRACWQWNPSDRPSFAEIHQAFETMFQESSISDEVEKELGKQGVRGAVSTLLQAPELPTKTRTSRRAAEHRDTTDVPEMPHSKGQGESDPLDHEPAVSPLLPRKERGPPEGGLNEDERLLPKDKKTNLFSALIKKKKKTAPTPPKRSSSFREMDGQPERRGAGEEEGRDISNGALAFTPLDTADPAKSPKPSNGAGVPNGALRESGGSGFRSPHLWKKSSTLTSSRLATGEEEGGGSSSKRFLRSCSASCMPHGAKDTEWRSVTLPRDLQSTGRQFDSSTFGGHKSEKPALPRKRAGENRSDQVTRGTVTPPPRLVKKNEEAADEVFKDIMESSPGSSPPNLTPKPLRRQVTVAPASGLPHKEEAGKGSALGTPAAAEPVTPTSKAGSGAPGGTSKGPAEESRVRRHKHSSESPGRDKGKLSRLKPAPPPPPAASAGKAGGKPSQSPSQEAAGEAVLGAKTKATSLVDAVNSDAAKPSQPAEGLKKPVLPATPKPQSAKEPSGTPISPTPVPSTLAAPAPAPLPPDSKPSMPPQLQPEREETEPASPSPPPPALPEAKPPRPEPPAPQPEPT",
            "description": "Imatinib (Gleevec) targeting ABL1 tyrosine kinase — classic targeted therapy for CML."
        },
        "Erlotinib + EGFR (Lung Cancer Target)": {
            "smiles": "COCCOC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC=CC(=C3)C#C)OCCOC",
            "sequence": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYTSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
            "description": "Erlotinib (Tarceva) targeting EGFR kinase — widely used targeted therapy for NSCLC."
        },
        "Gefitinib + EGFR (Kinase Inhibitor)": {
            "smiles": "COC1=C(C=C2C(=C1)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4",
            "sequence": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYTSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
            "description": "Gefitinib (Iressa) selective EGFR tyrosine kinase inhibitor."
        }
    }


def get_benchmark_results_data(models_dir: str = "./models") -> pd.DataFrame:
    """
    Returns benchmark results 12-cell matrix (4 Algos x 3 Splits).
    Reads from models/results.csv if present, otherwise returns LEGACY SYNTHETIC defaults.
    The synthetic fallback includes a 'Source' column marked as 'LEGACY SYNTHETIC'.
    """
    results_path = os.path.join(models_dir, "results.csv")
    if os.path.exists(results_path):
        try:
            return pd.read_csv(results_path)
        except Exception:
            pass
            
    data = [
        {"Split": "Random Split (Baseline)", "Algorithm": "Random Forest", "MSE": 0.215, "RMSE": 0.463, "Pearson r": 0.842, "CI": 0.785, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Random Split (Baseline)", "Algorithm": "XGBoost", "MSE": 0.188, "RMSE": 0.433, "Pearson r": 0.868, "CI": 0.812, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Random Split (Baseline)", "Algorithm": "Support Vector Regression (SVR)", "MSE": 0.245, "RMSE": 0.495, "Pearson r": 0.810, "CI": 0.758, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Random Split (Baseline)", "Algorithm": "Gaussian Process Regression (GPR)", "MSE": 0.230, "RMSE": 0.479, "Pearson r": 0.825, "CI": 0.771, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Cold-Drug Split", "Algorithm": "Random Forest", "MSE": 0.380, "RMSE": 0.616, "Pearson r": 0.635, "CI": 0.665, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Cold-Drug Split", "Algorithm": "XGBoost", "MSE": 0.345, "RMSE": 0.587, "Pearson r": 0.668, "CI": 0.692, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Cold-Drug Split", "Algorithm": "Support Vector Regression (SVR)", "MSE": 0.412, "RMSE": 0.641, "Pearson r": 0.598, "CI": 0.630, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Cold-Drug Split", "Algorithm": "Gaussian Process Regression (GPR)", "MSE": 0.395, "RMSE": 0.628, "Pearson r": 0.612, "CI": 0.648, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Cold-Protein Split", "Algorithm": "Random Forest", "MSE": 0.420, "RMSE": 0.648, "Pearson r": 0.590, "CI": 0.628, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Cold-Protein Split", "Algorithm": "XGBoost", "MSE": 0.390, "RMSE": 0.624, "Pearson r": 0.621, "CI": 0.654, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Cold-Protein Split", "Algorithm": "Support Vector Regression (SVR)", "MSE": 0.465, "RMSE": 0.681, "Pearson r": 0.542, "CI": 0.595, "Source": "LEGACY SYNTHETIC"},
        {"Split": "Cold-Protein Split", "Algorithm": "Gaussian Process Regression (GPR)", "MSE": 0.440, "RMSE": 0.663, "Pearson r": 0.565, "CI": 0.610, "Source": "LEGACY SYNTHETIC"},
    ]
    return pd.DataFrame(data)

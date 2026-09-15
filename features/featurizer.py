"""
Synthetic benchmark featurizer (1,071 dims) — development/testing only.

Feature Engineering Module for Drug and Protein Vector Extraction.
NOTE: This featurizer generates 1,071 dimensions (1,030 drug + 41 protein) used by the
legacy synthetic benchmark and real-time inference prototype.
For real KIBA research experiments (1,227 dims: 1,030 drug + 197 protein), see extract_features.py.

Computes:
- Drug Features (RDKit): 1024-bit Morgan Fingerprint (ECFP4) + 6 Physicochemical Descriptors
- Protein Features: Amino Acid Composition (AAC - 20 dims) + CTD Composition (21 dims)
- Combined Feature Matrix concatenating Drug and Protein representation vectors (1,071 dims)
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional


def get_features_dir() -> str:
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("FEATURES_DIR", "./features")


# =======================================================
# DRUG FEATURIZATION (RDKit Morgan Fingerprint + Descriptors)
# =======================================================
def compute_drug_features(smiles: str) -> Optional[np.ndarray]:
    """
    Featurize a SMILES string into a 1030-dimensional numeric vector:
    - 1024 Morgan Fingerprint bits (radius 2, ECFP4)
    - 6 physicochemical descriptors (MW, LogP, TPSA, HBD, HBA, Rotatable Bonds)
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # 1. 1024-bit Morgan Fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        fp_vec = np.array(fp, dtype=np.float32)
        
        # 2. 6 Physicochemical Descriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot = Descriptors.NumRotatableBonds(mol)
        
        desc_vec = np.array([mw, logp, tpsa, hbd, hba, rot], dtype=np.float32)
        
        # Concatenate fingerprint + descriptors
        return np.concatenate([fp_vec, desc_vec])
    except Exception:
        # Fallback synthetic drug vector if rdkit is unavailable
        seed = int(hashlib.sha256(smiles.encode()).hexdigest(), 16) % 100000
        np.random.seed(seed)
        return np.random.randn(1030).astype(np.float32)


def get_morgan_bit_context(smiles: str, bit_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Return atom-environment metadata for requested Morgan bits."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return {}
        bit_info: Dict[int, list[tuple[int, int]]] = {}
        AllChem.GetMorganFingerprintAsBitVect(molecule, radius=2, nBits=1024, bitInfo=bit_info)
        context: Dict[str, Dict[str, Any]] = {}
        for bit_name in bit_names:
            if not bit_name.startswith("Morgan_Bit_"):
                continue
            bit_id = int(bit_name.rsplit("_", 1)[-1])
            matches = []
            for atom_index, radius in bit_info.get(bit_id, []):
                bond_indices = AllChem.FindAtomEnvironmentOfRadiusN(molecule, radius, atom_index)
                atom_indices = {atom_index}
                for bond_index in bond_indices:
                    bond = molecule.GetBondWithIdx(bond_index)
                    atom_indices.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                matches.append({"atom_indices": sorted(atom_indices), "radius": radius})
            context[bit_name] = {"highlight_supported": bool(matches), "matches": matches}
        return context
    except Exception:
        return {}


def get_drug_feature_names() -> List[str]:
    fp_names = [f"Morgan_Bit_{i}" for i in range(1024)]
    desc_names = ["Desc_MolWt", "Desc_MolLogP", "Desc_TPSA", "Desc_NumHDonors", "Desc_NumHAcceptors", "Desc_NumRotatableBonds"]
    return fp_names + desc_names


# =======================================================
# PROTEIN FEATURIZATION (AAC + CTD Composition)
# =======================================================
def compute_protein_aac(sequence: str) -> np.ndarray:
    """
    Compute 20-dimensional Amino Acid Composition (AAC).
    Frequencies of A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    seq = sequence.upper()
    total_len = len(seq)
    
    if total_len == 0:
        return np.zeros(20, dtype=np.float32)
        
    counts = [seq.count(aa) for aa in amino_acids]
    aac_vec = np.array(counts, dtype=np.float32) / float(total_len)
    return aac_vec


def compute_protein_ctd(sequence: str) -> np.ndarray:
    """
    Compute 21-dimensional CTD (Composition) features across 7 physicochemical properties:
    1. Hydrophobicity
    2. Normalized van der Waals volume
    3. Polarity
    4. Polarizability
    5. Charge
    6. Secondary structure
    7. Solvent accessibility
    """
    groups = {
        "hydrophobicity": ["CLVIMFW", "GASTPHY", "PHEKTED"],
        "volume": ["GASDT", "CPNVEQIL", "KMHFRYW"],
        "polarity": ["LIFWCMVY", "PATGS", "HQRKNED"],
        "polarizability": ["GASDT", "CPNVEQIL", "KMHFRYW"],
        "charge": ["KR", "ANCQGHILMFPSTWVYV", "DE"],
        "secondary_struct": ["EALMQKRH", "VIYCWFT", "GNPSD"],
        "solvent_access": ["ALFCGIVW", "RKQEND", "MSPTHY"],
    }
    
    seq = sequence.upper()
    total_len = len(seq)
    if total_len == 0:
        return np.zeros(21, dtype=np.float32)
        
    ctd_values = []
    for prop, grp_list in groups.items():
        for grp in grp_list:
            count = sum(seq.count(aa) for aa in grp)
            ctd_values.append(count / float(total_len))
            
    return np.array(ctd_values, dtype=np.float32)


def compute_protein_features(sequence: str) -> np.ndarray:
    """
    Featurize protein amino acid sequence into a 41-dimensional numeric vector (AAC + CTD).
    """
    aac = compute_protein_aac(sequence)
    ctd = compute_protein_ctd(sequence)
    return np.concatenate([aac, ctd])


def get_protein_feature_names() -> List[str]:
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    aac_names = [f"AAC_{aa}" for aa in amino_acids]
    ctd_names = [f"CTD_Group_{i+1}" for i in range(21)]
    return aac_names + ctd_names


# =======================================================
# FULL DATASET FEATURIZATION PIPELINE
# =======================================================
def featurize_single_pair(smiles: str, sequence: str) -> np.ndarray:
    """
    Featurize a single drug-target pair into a concatenated feature vector (1071 dims).
    """
    drug_vec = compute_drug_features(smiles)
    if drug_vec is None:
        drug_vec = np.zeros(1030, dtype=np.float32)
        
    protein_vec = compute_protein_features(sequence)
    return np.concatenate([drug_vec, protein_vec])


def run_feature_pipeline():
    from dotenv import load_dotenv
    load_dotenv()
    
    data_dir = os.getenv("DATA_DIR", "./data")
    features_dir = get_features_dir()
    os.makedirs(features_dir, exist_ok=True)
    
    kiba_path = os.path.join(data_dir, "kiba_processed.pkl")
    if not os.path.exists(kiba_path):
        raise FileNotFoundError(f"Processed dataset not found at {kiba_path}. Run data_pipeline.py first.")
        
    df = pd.read_pickle(kiba_path)
    print(f"Loaded {len(df)} interaction samples for feature extraction...")
    
    # 1. Featurize unique SMILES
    unique_smiles = df["smiles"].unique()
    print(f"Featurizing {len(unique_smiles)} unique drug SMILES strings...")
    drug_map = {}
    for s in unique_smiles:
        vec = compute_drug_features(s)
        if vec is not None:
            drug_map[s] = vec
        else:
            drug_map[s] = np.zeros(1030, dtype=np.float32)
            
    # 2. Featurize unique Protein Sequences
    unique_seqs = df["protein_sequence"].unique()
    print(f"Featurizing {len(unique_seqs)} unique protein sequences...")
    protein_map = {}
    for seq in unique_seqs:
        protein_map[seq] = compute_protein_features(seq)
        
    # 3. Construct combined feature matrix X and label vector y
    print("Constructing combined feature matrix X (1071 dimensions per sample)...")
    X_list = []
    y_list = []
    
    for idx, row in df.iterrows():
        d_vec = drug_map[row["smiles"]]
        p_vec = protein_map[row["protein_sequence"]]
        combined_vec = np.concatenate([d_vec, p_vec])
        X_list.append(combined_vec)
        y_list.append(row["kiba_score"])
        
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    feature_names = get_drug_feature_names() + get_protein_feature_names()
    
    # Save output artifacts
    output_dict = {
        "X": X,
        "y": y,
        "feature_names": feature_names,
        "drug_map": drug_map,
        "protein_map": protein_map
    }
    
    output_path = os.path.join(features_dir, "combined_dataset.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(output_dict, f)
        
    print(f"Saved feature matrix X {X.shape} and label y {y.shape} to {output_path}")
    print("Feature extraction pipeline executed successfully!")


if __name__ == "__main__":
    run_feature_pipeline()

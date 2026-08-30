from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem, Descriptors, Lipinski
except Exception as exc:  # pragma: no cover - runtime guard
    Chem = None
    DataStructs = None
    AllChem = None
    Descriptors = None
    Lipinski = None
    RDLogger = None
    _RDKIT_IMPORT_ERROR = exc
else:
    _RDKIT_IMPORT_ERROR = None
    RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"
DRUG_FP_BITS = 1024
PSEAAC_LAMBDA = 10
PSEAAC_WEIGHT = 0.05
CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"

AA_INDEX = {aa: idx for idx, aa in enumerate(CANONICAL_AA)}
AA_HYDROPHOBICITY = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}
AA_HYDROPHILICITY = {
    "A": -0.5,
    "C": -1.0,
    "D": 3.0,
    "E": 3.0,
    "F": -2.5,
    "G": 0.0,
    "H": -0.5,
    "I": -1.8,
    "K": 3.0,
    "L": -1.8,
    "M": -1.3,
    "N": 0.2,
    "P": 0.0,
    "Q": 0.2,
    "R": 3.0,
    "S": 0.3,
    "T": -0.4,
    "V": -1.5,
    "W": -3.4,
    "Y": -2.3,
}
AA_SIDE_CHAIN = {
    "A": 1.0,
    "C": 2.0,
    "D": 3.0,
    "E": 3.5,
    "F": 4.5,
    "G": 0.5,
    "H": 4.0,
    "I": 3.5,
    "K": 4.5,
    "L": 3.5,
    "M": 3.5,
    "N": 2.5,
    "P": 2.5,
    "Q": 3.0,
    "R": 5.0,
    "S": 1.5,
    "T": 1.5,
    "V": 3.0,
    "W": 5.0,
    "Y": 4.5,
}

CTD_GROUPS = {
    "hydrophobicity": (set("RKEDQN"), set("GASTPHY"), set("CLVIMFW")),
    "normwaalsvolume": (set("GASTPD"), set("NVEQIL"), set("MHKFRYW")),
    "polarity": (set("LIFWCMVY"), set("PATGS"), set("HQRKNED")),
    "charge": (set("KR"), set("ANCQGHILMFPSTWYV"), set("DE")),
    "secondarystruct": (set("EALMQKRH"), set("VIYCWFT"), set("GNPSD")),
    "solventaccess": (set("ALFCGIVW"), set("RKQEND"), set("MPSTHY")),
    "polarizability": (set("GASDT"), set("CPNVEQIL"), set("KMHFRYW")),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sanitize_sequence(seq: str) -> str:
    return "".join(aa for aa in seq.upper() if aa in AA_INDEX)


def _safe_rank(length: int, fraction: float) -> int:
    if length <= 0:
        return 0
    return max(1, min(length, int(np.ceil(length * fraction))))


def _group_labels(seq: str, groups: tuple[set[str], set[str], set[str]]) -> list[int]:
    labels: list[int] = []
    for aa in seq:
        for idx, group in enumerate(groups, start=1):
            if aa in group:
                labels.append(idx)
                break
    return labels


def _distribution_for_group(labels: list[int], group_idx: int) -> list[float]:
    positions = [idx + 1 for idx, label in enumerate(labels) if label == group_idx]
    if not positions:
        return [0.0] * 5

    total = float(len(labels))
    out: list[float] = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        rank = _safe_rank(len(positions), fraction) - 1
        out.append(positions[rank] / total)
    return out


def calculate_aac(seq: str) -> np.ndarray:
    seq = sanitize_sequence(seq)
    if not seq:
        return np.zeros(len(CANONICAL_AA), dtype=np.float32)

    counts = np.array([seq.count(aa) for aa in CANONICAL_AA], dtype=np.float32)
    return counts / float(len(seq))


def calculate_ctd(seq: str) -> np.ndarray:
    seq = sanitize_sequence(seq)
    if len(seq) < 2:
        return np.zeros(len(CTD_GROUPS) * 21, dtype=np.float32)

    features: list[float] = []
    for groups in CTD_GROUPS.values():
        labels = _group_labels(seq, groups)
        if not labels:
            features.extend([0.0] * 21)
            continue

        total = float(len(labels))
        comp = [labels.count(group_idx) / total for group_idx in (1, 2, 3)]

        transition_counts = [0, 0, 0]
        for left, right in zip(labels, labels[1:]):
            pair = {left, right}
            if pair == {1, 2}:
                transition_counts[0] += 1
            elif pair == {1, 3}:
                transition_counts[1] += 1
            elif pair == {2, 3}:
                transition_counts[2] += 1
        transitions = [count / float(len(labels) - 1) for count in transition_counts]

        distribution = []
        for group_idx in (1, 2, 3):
            distribution.extend(_distribution_for_group(labels, group_idx))

        features.extend(comp + transitions + distribution)

    return np.asarray(features, dtype=np.float32)


def _theta_lag(seq: str, lag: int) -> float:
    if len(seq) <= lag:
        return 0.0

    values: list[float] = []
    for idx in range(len(seq) - lag):
        left = seq[idx]
        right = seq[idx + lag]
        if left not in AA_INDEX or right not in AA_INDEX:
            continue
        diff1 = AA_HYDROPHOBICITY[left] - AA_HYDROPHOBICITY[right]
        diff2 = AA_HYDROPHILICITY[left] - AA_HYDROPHILICITY[right]
        diff3 = AA_SIDE_CHAIN[left] - AA_SIDE_CHAIN[right]
        values.append((diff1 * diff1 + diff2 * diff2 + diff3 * diff3) / 3.0)
    if not values:
        return 0.0
    return float(np.mean(values))


def calculate_pseaac(seq: str, lambda_value: int = PSEAAC_LAMBDA, weight: float = PSEAAC_WEIGHT) -> np.ndarray:
    seq = sanitize_sequence(seq)
    if not seq:
        return np.zeros(len(CANONICAL_AA) + lambda_value, dtype=np.float32)

    aac = calculate_aac(seq)
    theta = np.array([_theta_lag(seq, lag) for lag in range(1, lambda_value + 1)], dtype=np.float32)
    denominator = 1.0 + weight * float(np.sum(theta))
    if denominator == 0.0:
        denominator = 1.0

    comp = aac / denominator
    order = (weight * theta) / denominator
    return np.concatenate([comp, order]).astype(np.float32)


def extract_drug_features(smiles: str) -> np.ndarray | None:
    if _RDKIT_IMPORT_ERROR is not None:
        raise RuntimeError(
            "RDKit could not be imported from the current environment. "
            "Reinstall the dependency set before running feature extraction."
        ) from _RDKIT_IMPORT_ERROR

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=DRUG_FP_BITS)
        fp_arr = np.zeros((DRUG_FP_BITS,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, fp_arr)

        desc_vals = np.array(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Lipinski.NumHDonors(mol),
                Lipinski.NumHAcceptors(mol),
                Lipinski.NumRotatableBonds(mol),
            ],
            dtype=np.float32,
        )
        return np.concatenate([fp_arr, desc_vals])
    except Exception:
        return None


def extract_protein_features(seq: str) -> np.ndarray | None:
    try:
        seq = sanitize_sequence(seq)
        if not seq:
            return None
        aac = calculate_aac(seq)
        ctd = calculate_ctd(seq)
        pseaac = calculate_pseaac(seq)
        return np.concatenate([aac, ctd, pseaac]).astype(np.float32)
    except Exception:
        return None


def featurize_protein_item(protein_id: str, seq: str) -> tuple[str, np.ndarray | None]:
    return protein_id, extract_protein_features(seq)


def _load_mappings() -> tuple[dict[str, str], dict[str, str]]:
    ligand_file = DATA_DIR / "ligands_can.txt"
    protein_file = DATA_DIR / "proteins.txt"
    with ligand_file.open("r", encoding="utf-8") as f:
        ligands_dict = json.load(f)
    with protein_file.open("r", encoding="utf-8") as f:
        proteins_dict = json.load(f)
    return ligands_dict, proteins_dict


def main() -> None:
    ensure_dir(FEATURES_DIR)

    ligands_dict, proteins_dict = _load_mappings()
    print(f"Loaded {len(ligands_dict)} drugs and {len(proteins_dict)} proteins.")

    t0 = time.time()
    drug_features: dict[str, np.ndarray] = {}
    failed_drugs = 0
    for drug_id, smiles in ligands_dict.items():
        feats = extract_drug_features(smiles)
        if feats is None:
            failed_drugs += 1
            continue
        drug_features[drug_id] = feats
    print(
        f"Successfully featurized {len(drug_features)} drugs "
        f"({failed_drugs} failed) in {time.time() - t0:.2f} seconds."
    )

    t0 = time.time()
    protein_features: dict[str, np.ndarray] = {}
    failed_proteins = 0
    for protein_id, seq in proteins_dict.items():
        _, feats = featurize_protein_item(protein_id, seq)
        if feats is None:
            failed_proteins += 1
            continue
        protein_features[protein_id] = feats
    print(
        f"Successfully featurized {len(protein_features)} proteins "
        f"({failed_proteins} failed) in {time.time() - t0:.2f} seconds."
    )

    drug_out = FEATURES_DIR / "drug_features.pkl"
    protein_out = FEATURES_DIR / "protein_features.pkl"
    manifest_out = FEATURES_DIR / "feature_manifest.json"

    with drug_out.open("wb") as f:
        pickle.dump(drug_features, f)
    with protein_out.open("wb") as f:
        pickle.dump(protein_features, f)

    manifest = {
        "drug_feature_dim": int(next(iter(drug_features.values())).shape[0]) if drug_features else 0,
        "protein_feature_dim": int(next(iter(protein_features.values())).shape[0]) if protein_features else 0,
        "drug_feature_count": int(len(drug_features)),
        "protein_feature_count": int(len(protein_features)),
        "drug_failed": int(failed_drugs),
        "protein_failed": int(failed_proteins),
        "drug_features": ["morgan_ecfp4_1024", "molwt", "logp", "tpsa", "hbd", "hba", "rotatable_bonds"],
        "protein_features": ["aac_20", "ctd_147", f"pseaac_{len(CANONICAL_AA) + PSEAAC_LAMBDA}"],
    }
    manifest_out.write_text(json.dumps(manifest, indent=2))

    print("Feature extraction complete. Saved:")
    print(f"  - {drug_out}")
    print(f"  - {protein_out}")
    print(f"  - {manifest_out}")

    if drug_features:
        print(f"Drug feature dimension: {next(iter(drug_features.values())).shape}")
    if protein_features:
        print(f"Protein feature dimension: {next(iter(protein_features.values())).shape}")


if __name__ == "__main__":
    main()

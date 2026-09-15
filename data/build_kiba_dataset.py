from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_BASE = "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data/kiba"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(name: str) -> Path:
    ensure_dir(DATA_DIR)
    target = DATA_DIR / name
    if target.exists():
        return target

    import urllib.request

    url = f"{RAW_BASE}/{name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        content = response.read()
    target.write_bytes(content)
    return target


def load_kiba_matrix() -> pd.DataFrame:
    ligand_file = download_file("ligands_can.txt")
    protein_file = download_file("proteins.txt")
    affinity_file = download_file("kiba_binding_affinity_v2.txt")

    with ligand_file.open("r", encoding="utf-8") as f:
        ligands_dict = json.load(f)
    with protein_file.open("r", encoding="utf-8") as f:
        proteins_dict = json.load(f)

    ligands = list(ligands_dict.keys())
    proteins = list(proteins_dict.keys())

    affinity_matrix = np.genfromtxt(affinity_file, delimiter="\t", dtype=float, comments=None)
    if affinity_matrix.shape[1] == len(proteins) + 1:
        affinity_matrix = affinity_matrix[:, :len(proteins)]

    if affinity_matrix.shape != (len(ligands), len(proteins)):
        raise ValueError(
            f"Unexpected KIBA matrix shape: {affinity_matrix.shape}; expected {(len(ligands), len(proteins))}."
        )

    df = pd.DataFrame(affinity_matrix, index=ligands, columns=proteins)
    melted = df.reset_index().melt(id_vars="index", var_name="protein_id", value_name="affinity")
    melted = melted.rename(columns={"index": "drug_id"})
    melted = melted.dropna(subset=["affinity"]).reset_index(drop=True)
    melted["affinity"] = melted["affinity"].astype(float)
    return melted


def save_split(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    train_path = DATA_DIR / f"{name}_train.csv"
    test_path = DATA_DIR / f"{name}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    return train_path, test_path


def random_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_idx, test_idx = train_test_split(df.index.to_numpy(), test_size=test_size, random_state=seed)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def cold_drug_split(
    df: pd.DataFrame,
    ligands_dict: dict[str, str],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Cold-Drug split by partitioning unique canonical SMILES structures.
    This guarantees 0 drug ID overlap AND 0 canonical SMILES overlap, preventing
    identical chemical structures with different ChEMBL IDs from leaking into test.
    """
    # Find unique canonical SMILES present in the dataset
    drug_to_smiles = {d: ligands_dict[d] for d in df["drug_id"].unique()}
    unique_smiles = sorted(list(set(drug_to_smiles.values())))

    rng = np.random.RandomState(seed)
    n_test_smiles = max(1, int(len(unique_smiles) * test_size))
    test_smiles = set(rng.choice(unique_smiles, size=n_test_smiles, replace=False).tolist())

    test_drugs = set(d for d, s in drug_to_smiles.items() if s in test_smiles)
    test_df = df[df["drug_id"].isin(test_drugs)].reset_index(drop=True)
    train_df = df[~df["drug_id"].isin(test_drugs)].reset_index(drop=True)
    return train_df, test_df


def cold_protein_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate Cold-Protein split holding out 20% of unique proteins.
    """
    all_proteins = sorted(df["protein_id"].drop_duplicates().tolist())
    rng = np.random.RandomState(seed)
    n_test_proteins = max(1, int(len(all_proteins) * test_size))
    test_proteins = set(rng.choice(all_proteins, size=n_test_proteins, replace=False).tolist())
    test_df = df[df["protein_id"].isin(test_proteins)].reset_index(drop=True)
    train_df = df[~df["protein_id"].isin(test_proteins)].reset_index(drop=True)
    return train_df, test_df


def validate_splits(
    splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    ligands_dict: dict[str, str],
    proteins_dict: dict[str, str],
) -> dict[str, Any]:
    """
    Validate that all split partitions satisfy their cold-split invariants:
    - cold_drug: drug ID overlap == 0 and canonical SMILES overlap == 0
    - cold_protein: protein ID overlap == 0 and sequence overlap == 0
    """
    split_meta: dict[str, Any] = {}

    for split_name, (train_df, test_df) in splits.items():
        train_drugs = set(train_df["drug_id"])
        test_drugs = set(test_df["drug_id"])
        drug_id_overlap = len(train_drugs & test_drugs)

        train_smiles = {ligands_dict[d] for d in train_drugs}
        test_smiles = {ligands_dict[d] for d in test_drugs}
        smiles_overlap = len(train_smiles & test_smiles)

        train_prots = set(train_df["protein_id"])
        test_prots = set(test_df["protein_id"])
        protein_id_overlap = len(train_prots & test_prots)

        train_seqs = {proteins_dict[p] for p in train_prots}
        test_seqs = {proteins_dict[p] for p in test_prots}
        seq_overlap = len(train_seqs & test_seqs)

        split_meta[split_name] = {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_drugs": int(len(train_drugs)),
            "test_drugs": int(len(test_drugs)),
            "drug_id_overlap": int(drug_id_overlap),
            "train_smiles": int(len(train_smiles)),
            "test_smiles": int(len(test_smiles)),
            "smiles_overlap": int(smiles_overlap),
            "train_proteins": int(len(train_prots)),
            "test_proteins": int(len(test_prots)),
            "protein_id_overlap": int(protein_id_overlap),
            "train_sequences": int(len(train_seqs)),
            "test_sequences": int(len(test_seqs)),
            "sequence_overlap": int(seq_overlap),
        }

        print(f"\n--- Split Validation: {split_name} ---")
        print(f"  Rows: train={len(train_df):,}, test={len(test_df):,}")
        print(f"  Drugs: train={len(train_drugs)}, test={len(test_drugs)}, ID overlap={drug_id_overlap}, SMILES overlap={smiles_overlap}")
        print(f"  Proteins: train={len(train_prots)}, test={len(test_prots)}, ID overlap={protein_id_overlap}, Sequence overlap={seq_overlap}")

        if split_name == "cold_drug":
            assert drug_id_overlap == 0, f"Cold-drug split has {drug_id_overlap} overlapping drug IDs!"
            assert smiles_overlap == 0, f"Cold-drug split has {smiles_overlap} overlapping SMILES structures!"
        elif split_name == "cold_protein":
            assert protein_id_overlap == 0, f"Cold-protein split has {protein_id_overlap} overlapping protein IDs!"
            assert seq_overlap == 0, f"Cold-protein split has {seq_overlap} overlapping protein sequences!"

    return split_meta


def main() -> None:
    ligand_file = DATA_DIR / "ligands_can.txt"
    protein_file = DATA_DIR / "proteins.txt"
    with ligand_file.open("r", encoding="utf-8") as f:
        ligands_dict = json.load(f)
    with protein_file.open("r", encoding="utf-8") as f:
        proteins_dict = json.load(f)

    pairs = load_kiba_matrix()
    pairs_path = DATA_DIR / "kiba_clean.csv"
    pairs.to_csv(pairs_path, index=False)

    random_train, random_test = random_split(pairs)
    cold_drug_train, cold_drug_test = cold_drug_split(pairs, ligands_dict)
    cold_protein_train, cold_protein_test = cold_protein_split(pairs)

    splits = {
        "random": (random_train, random_test),
        "cold_drug": (cold_drug_train, cold_drug_test),
        "cold_protein": (cold_protein_train, cold_protein_test),
    }

    print("Validating all split invariants...")
    split_meta = validate_splits(splits, ligands_dict, proteins_dict)

    save_split("random", random_train, random_test)
    save_split("cold_drug", cold_drug_train, cold_drug_test)
    save_split("cold_protein", cold_protein_train, cold_protein_test)

    summary = {
        "dataset": "KIBA",
        "description": "Kinase Inhibitor BioActivity benchmark dataset",
        "rows_total": int(len(pairs)),
        "unique_drugs": int(pairs["drug_id"].nunique()),
        "unique_canonical_smiles": int(len(set(ligands_dict[d] for d in pairs["drug_id"].unique()))),
        "unique_proteins": int(pairs["protein_id"].nunique()),
        "unique_protein_sequences": int(len(set(proteins_dict[p] for p in pairs["protein_id"].unique()))),
        "affinity_min": float(pairs["affinity"].min()),
        "affinity_max": float(pairs["affinity"].max()),
        "affinity_mean": float(pairs["affinity"].mean()),
        "random_train_rows": int(len(random_train)),
        "random_test_rows": int(len(random_test)),
        "cold_drug_train_rows": int(len(cold_drug_train)),
        "cold_drug_test_rows": int(len(cold_drug_test)),
        "cold_protein_train_rows": int(len(cold_protein_train)),
        "cold_protein_test_rows": int(len(cold_protein_test)),
    }

    summary_path = DATA_DIR / "kiba_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    split_meta_path = DATA_DIR / "kiba_split_metadata.json"
    split_meta_path.write_text(json.dumps(split_meta, indent=2))

    print("\nKIBA dataset and leak-free cold splits prepared successfully.")
    print(f"Summary saved to: {summary_path}")
    print(f"Split metadata saved to: {split_meta_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


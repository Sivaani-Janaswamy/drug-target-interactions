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


def random_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    train_idx, test_idx = train_test_split(df.index.to_numpy(), test_size=test_size, random_state=seed)
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def cold_drug_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    all_drugs = df["drug_id"].drop_duplicates().tolist()
    rng = np.random.RandomState(seed)
    test_drugs = set(rng.choice(all_drugs, size=max(1, int(len(all_drugs) * test_size)), replace=False).tolist())
    test_df = df[df["drug_id"].isin(test_drugs)].reset_index(drop=True)
    train_df = df[~df["drug_id"].isin(test_drugs)].reset_index(drop=True)
    return train_df, test_df


def cold_protein_split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    all_proteins = df["protein_id"].drop_duplicates().tolist()
    rng = np.random.RandomState(seed)
    test_proteins = set(rng.choice(all_proteins, size=max(1, int(len(all_proteins) * test_size)), replace=False).tolist())
    test_df = df[df["protein_id"].isin(test_proteins)].reset_index(drop=True)
    train_df = df[~df["protein_id"].isin(test_proteins)].reset_index(drop=True)
    return train_df, test_df


def main() -> None:
    pairs = load_kiba_matrix()
    pairs_path = DATA_DIR / "kiba_clean.csv"
    pairs.to_csv(pairs_path, index=False)

    random_train, random_test = random_split(pairs)
    cold_drug_train, cold_drug_test = cold_drug_split(pairs)
    cold_protein_train, cold_protein_test = cold_protein_split(pairs)

    save_split("random", random_train, random_test)
    save_split("cold_drug", cold_drug_train, cold_drug_test)
    save_split("cold_protein", cold_protein_train, cold_protein_test)

    summary = {
        "rows_total": int(len(pairs)),
        "unique_drugs": int(pairs["drug_id"].nunique()),
        "unique_proteins": int(pairs["protein_id"].nunique()),
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

    print("KIBA dataset prepared successfully.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

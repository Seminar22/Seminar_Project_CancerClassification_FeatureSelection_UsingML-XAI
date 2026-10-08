"""
Download the TCGA RNA-seq data (BRCA, LUAD, THCA, OV, COAD) directly from the GDC API,
the same source used in Dalmolin et al., AI 2025, 6(1), 2.  Pure Python (no R needed).

Put this file in:  src/download_gdc.py
Run from the project root (venv active):   python src\\download_gdc.py

Needs one extra package:   pip install requests

v3: every batch is validated and retried (handles truncated downloads) and cached
on disk, so if the connection drops you can just re-run and it resumes.
Produces TWO versions of the dataset (same samples, different gene sets):
    allgenes       - all genes after the 25% low-expression filter (as in the paper)
    proteincoding  - protein-coding genes only, then the same filter

Output:
    data/raw/cache_<CANCER>/batch_XXXX.pkl                  (per-batch cache)
    data/raw/<CANCER>_tpm.pkl                               (one file per cancer)
    data/raw/gene_types.csv                                 (gene_name -> gene_type)
    data/processed/tcga_5cancers_allgenes_full.csv
    data/processed/tcga_5cancers_allgenes_balanced.csv
    data/processed/tcga_5cancers_proteincoding_full.csv
    data/processed/tcga_5cancers_proteincoding_balanced.csv
"""
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API = "https://api.gdc.cancer.gov"
PROJECTS = {
    "BRCA": "TCGA-BRCA",
    "LUAD": "TCGA-LUAD",
    "THCA": "TCGA-THCA",
    "OV": "TCGA-OV",
    "COAD": "TCGA-COAD",
}
RAW = Path("data/raw")
OUT = Path("data/processed")
BATCH = 20          # files per download request (small = fewer truncated downloads)
RETRIES = 6
N_PER_CLASS = 421   # value used in the paper
QNT_CUT = 0.25      # same filtering threshold as the paper
SEED = 42
RAW.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def list_files(project_id: str) -> pd.DataFrame:
    """Query GDC for open-access STAR-Counts files of Primary Tumor samples."""
    filters = {"op": "and", "content": [
        {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
        {"op": "in", "content": {"field": "data_category", "value": ["Transcriptome Profiling"]}},
        {"op": "in", "content": {"field": "data_type", "value": ["Gene Expression Quantification"]}},
        {"op": "in", "content": {"field": "analysis.workflow_type", "value": ["STAR - Counts"]}},
        {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]}},
        {"op": "in", "content": {"field": "access", "value": ["open"]}},
    ]}
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,cases.submitter_id,cases.samples.submitter_id,cases.samples.sample_type",
        "format": "JSON",
        "size": "3000",
    }
    for attempt in range(RETRIES):
        try:
            r = requests.get(f"{API}/files", params=params, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print("  retry (file list):", e)
            time.sleep(5)
    else:
        raise RuntimeError("Could not list files from GDC")
    rows = []
    for hit in r.json()["data"]["hits"]:
        case = hit["cases"][0]
        tumor = [s for s in case["samples"] if s["sample_type"] == "Primary Tumor"]
        if not tumor:
            continue
        rows.append({"file_id": hit["file_id"],
                     "case": case["submitter_id"],
                     "sample": tumor[0]["submitter_id"]})
    return pd.DataFrame(rows).drop_duplicates("case")   # one sample per patient


def parse_tar(blob: bytes, id_to_sample: dict) -> pd.DataFrame:
    """Read every .tsv inside the downloaded tar.gz -> samples x genes (TPM)."""
    cols, names = {}, None
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.name.endswith(".tsv"):
                continue
            file_id = m.name.split("/")[0]
            df = pd.read_csv(tar.extractfile(m), sep="\t", comment="#")
            df = df[~df["gene_id"].str.startswith("N_")]          # drop summary rows
            if names is None:
                names = df["gene_name"].values
            cols[id_to_sample[file_id]] = df["tpm_unstranded"].values
    if len(cols) == 0:
        raise ValueError("no expression files found in archive")
    mat = pd.DataFrame(cols, index=names).T.astype("float32")    # samples x gene symbols
    return mat.loc[:, ~mat.columns.duplicated()]


def fetch_batch(chunk: list, id_to_sample: dict, cache_file: Path) -> pd.DataFrame:
    if cache_file.exists():
        return pd.read_pickle(cache_file)
    for attempt in range(1, RETRIES + 1):
        try:
            r = requests.post(f"{API}/data", data=json.dumps({"ids": chunk}),
                              headers={"Content-Type": "application/json"},
                              timeout=(30, 600))
            r.raise_for_status()
            mat = parse_tar(r.content, id_to_sample)             # fails if download truncated
            if mat.shape[0] != len(chunk):
                raise ValueError(f"expected {len(chunk)} samples, got {mat.shape[0]}")
            mat.to_pickle(cache_file)
            return mat
        except (requests.RequestException, EOFError, tarfile.TarError, OSError, ValueError) as e:
            print(f"    attempt {attempt}/{RETRIES} failed: {type(e).__name__}: {e}")
            time.sleep(5 * attempt)
    raise RuntimeError("Batch failed after all retries - just re-run the script to resume")


def fetch_cancer(cancer: str) -> pd.DataFrame:
    final = RAW / f"{cancer}_tpm.pkl"
    if final.exists():
        print(f"{cancer}: loading cached result")
        return pd.read_pickle(final)

    files = list_files(PROJECTS[cancer])
    print(f"{cancer}: {len(files)} primary tumor files")
    id_to_sample = dict(zip(files.file_id, files["sample"]))
    ids = list(files.file_id)
    cache_dir = RAW / f"cache_{cancer}"
    cache_dir.mkdir(exist_ok=True)

    parts = []
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        cache_file = cache_dir / f"batch_{i // BATCH:04d}.pkl"
        status = "cached" if cache_file.exists() else "downloading"
        print(f"  {status} {i + 1}-{i + len(chunk)} / {len(ids)}")
        parts.append(fetch_batch(chunk, id_to_sample, cache_file))

    mat = pd.concat(parts)
    mat.insert(0, "label", cancer)
    mat.to_pickle(final)
    return mat


def get_gene_types() -> pd.Series:
    """gene_name -> gene_type (protein_coding, lncRNA, ...). Downloads one file, then caches."""
    f = RAW / "gene_types.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0)["gene_type"]
    fid = list_files(PROJECTS["BRCA"]).file_id.iloc[0]
    r = requests.get(f"{API}/data/{fid}", timeout=(30, 300))
    r.raise_for_status()
    df = pd.read_csv(io.BytesIO(r.content), sep="\t", comment="#")
    df = df[~df["gene_id"].str.startswith("N_")].drop_duplicates("gene_name")
    s = df.set_index("gene_name")["gene_type"]
    s.to_csv(f)
    return s


def qfilter(genes: pd.DataFrame) -> pd.DataFrame:
    """Drop genes with low mean expression (like TCGAanalyzeFiltering qnt.cut=0.25)."""
    means = genes.mean(axis=0)
    return genes.loc[:, means > means.quantile(QNT_CUT)]


def save_version(name: str, labels: pd.Series, genes: pd.DataFrame):
    data = pd.concat([labels, genes], axis=1)
    print(f"[{name}] genes: {genes.shape[1]} | {data['label'].value_counts().to_dict()}")
    data.to_csv(OUT / f"tcga_5cancers_{name}_full.csv")

    # undersample so every class has the same number of samples
    n = min(N_PER_CLASS, data["label"].value_counts().min())
    balanced = pd.concat([g.sample(n=n, random_state=SEED) for _, g in data.groupby("label")])
    balanced.to_csv(OUT / f"tcga_5cancers_{name}_balanced.csv")
    print(f"[{name}] balanced:", balanced.shape)


def main():
    frames = [fetch_cancer(c) for c in PROJECTS]
    data = pd.concat(frames, join="inner")
    print("Total samples:", data.shape[0], "| genes:", data.shape[1] - 1)
    labels = data["label"]
    logged = np.log2(data.drop(columns="label") + 1)

    # version A: all genes (as in the paper)
    save_version("allgenes", labels, qfilter(logged))

    # version B: protein-coding only
    gtypes = get_gene_types()
    pc = [g for g in logged.columns if gtypes.get(g) == "protein_coding"]
    save_version("proteincoding", labels, qfilter(logged[pc]))


if __name__ == "__main__":
    main()
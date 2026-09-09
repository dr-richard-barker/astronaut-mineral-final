#!/usr/bin/env python3
"""
01_fetch_osdr_data.py
Stage 1: Discover and download astronaut + rodent omics data from NASA OSDR API.

Outputs:
  - /mnt/shared-workspace/shared/osdr_raw/dataset_inventory.csv
  - /mnt/shared-workspace/shared/osdr_raw/<accession>_<datatype>.csv  (raw data)
  - /mnt/shared-workspace/shared/osdr_raw/<accession>_metadata.csv   (sample metadata)
"""
import os
import sys
import json
import time
import urllib.parse
import urllib.request
import csv
import io
import re

BASE = "https://visualization.osdr.nasa.gov/biodata/api"
RAW_DIR = "/mnt/shared-workspace/shared/osdr_raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ── I4 astronaut studies of interest ──
I4_STUDIES = {
    "OSD-569": "Whole blood RNA-seq, CBC, WGS",
    "OSD-570": "PBMC snRNA-seq, snATAC-seq, V(D)J",
    "OSD-571": "Plasma proteomics, metabolomics, cfDNA, cfRNA",
    "OSD-572": "Skin/oral/nasal swabs",
    "OSD-574": "Skin biopsy spatial transcriptomics",
    "OSD-575": "Serum comprehensive metabolic panel + cytokines",
    "OSD-630": "Stool metagenomics",
    "OSD-656": "Urine",
}

# Rodent tissues of interest
RODENT_TISSUES = ["liver", "blood", "muscle", "bone", "kidney", "heart", "spleen", "thymus"]


def api_get(path, fmt="csv", timeout=120):
    """Fetch from OSDR API. Returns text content."""
    url = f"{BASE}{path}"
    if "?" in url:
        url += f"&format={fmt}"
    else:
        url += f"?format={fmt}"
    req = urllib.request.Request(url, headers={"User-Agent": "Biomni-OSDR-Miner/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"  retry {attempt+1}/3 for {path}: {e}")
            time.sleep(5)
    return None


def parse_csv(text):
    """Parse CSV text into list of dicts."""
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def save_text(text, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


def discover_all_spaceflight_assays():
    """Get full assay inventory for all spaceflight studies."""
    print("=== Discovering all spaceflight assays ===")
    text = api_get("/v2/query/assays/?=study.characteristics.organism"
                   "&=study.factor%20value.spaceflight"
                   "&=investigation.study%20assays.study%20assay%20technology%20type"
                   "&=study.characteristics.organism%20part", fmt="csv")
    rows = parse_csv(text)
    print(f"  Found {len(rows)} assay-organism-factor rows")
    return rows


def filter_rodent_rnaseq(rows):
    """Filter for Mus musculus RNA-seq studies."""
    rodent = []
    for r in rows:
        org = r.get("study.characteristics.organism", "")
        tech = r.get("investigation.study assays.study assay technology type", "")
        acc = r.get("id.accession", "")
        if "Mus musculus" in org and ("rna-seq" in tech.lower() or "RNA sequencing" in tech.lower() or "rna sequencing" in tech.lower()):
            rodent.append(r)
    # deduplicate by accession
    seen = set()
    unique = []
    for r in rodent:
        acc = r["id.accession"]
        if acc not in seen:
            seen.add(acc)
            unique.append(r)
    print(f"  Mus musculus RNA-seq studies: {len(unique)}")
    return unique


def get_dataset_files(accession):
    """List files associated with a dataset."""
    text = api_get(f"/v2/dataset/{accession}/files/", fmt="json")
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    # files are under dataset.files
    if isinstance(data, dict):
        files = data.get("dataset", {}).get("files", [])
    elif isinstance(data, list):
        files = data
    else:
        files = []
    return files


def get_sample_metadata(accession):
    """Get sample-level metadata for a dataset."""
    text = api_get(f"/v2/query/metadata/?id.accession={accession}"
                   "&=study.characteristics.organism"
                   "&=study.characteristics.tissue"
                   "&=study.factor%20value.spaceflight"
                   "&=study.characteristics.strain"
                   "&=study.parameter%20value.mission", fmt="csv")
    return text


def download_data_file(accession, data_type):
    """Try to download a specific data type for a dataset."""
    text = api_get(f"/v2/query/data/?id.accession={accession}"
                   f"&file.data%20type={urllib.parse.quote(data_type)}", fmt="csv")
    return text


def main():
    # ── Step 1: Discover all spaceflight assays ──
    all_assays = discover_all_spaceflight_assays()
    save_text("\n".join([",".join(all_assays[0].keys())] + [",".join(r.values()) for r in all_assays]) if all_assays else "",
              os.path.join(RAW_DIR, "all_spaceflight_assays.csv"))

    # ── Step 2: Filter rodent RNA-seq ──
    rodent_studies = filter_rodent_rnaseq(all_assays)
    print("\n=== Rodent RNA-seq studies ===")
    for r in rodent_studies:
        print(f"  {r['id.accession']}: {r.get('investigation.study assays.study assay technology type','')}")

    # ── Step 3: Get metadata for I4 studies + rodent studies ──
    inventory = []
    all_studies = list(I4_STUDIES.keys()) + [r["id.accession"] for r in rodent_studies]

    print("\n=== Downloading metadata + data for target studies ===")
    for acc in all_studies:
        print(f"\n--- {acc} ---")
        is_i4 = acc in I4_STUDIES
        desc = I4_STUDIES.get(acc, "Rodent RNA-seq")

        # Get sample metadata
        meta_text = get_sample_metadata(acc)
        if meta_text:
            meta_path = os.path.join(RAW_DIR, f"{acc}_metadata.csv")
            save_text(meta_text, meta_path)
            meta_rows = parse_csv(meta_text)
            n_samples = len(meta_rows)
            tissues = set()
            for m in meta_rows:
                t = m.get("study.characteristics.tissue", "")
                if t:
                    tissues.add(t)
            print(f"  metadata: {n_samples} samples, tissues: {tissues}")
        else:
            n_samples = 0
            tissues = set()
            print(f"  WARNING: no metadata retrieved")

        # Get file listing
        files = get_dataset_files(acc)
        print(f"  files found: {len(files)}")

        # Try to download key data types
        data_types_to_try = [
            "unnormalized counts",
            "normalized counts",
            "differential expression",
            "differential abundance",
            "differential analysis",
            "metabolite abundance",
            "comprehensive metabolic panel",
            "DESeq2 differential expression",
            "gene expression",
            "protein expression",
            "metabolite expression",
        ]

        downloaded = []
        for dt in data_types_to_try:
            data_text = download_data_file(acc, dt)
            if data_text and len(data_text) > 100 and not data_text.startswith("error"):
                safe_dt = re.sub(r'[^a-zA-Z0-9]', '_', dt)
                data_path = os.path.join(RAW_DIR, f"{acc}_{safe_dt}.csv")
                save_text(data_text, data_path)
                n_rows = data_text.count("\n")
                downloaded.append(dt)
                print(f"  downloaded '{dt}': {n_rows} rows")

        # Also try to get file listing and download specific files directly
        for f in files:
            fname = f.get("file_name", f.get("fileName", ""))
            furl = f.get("file_url", f.get("URL", f.get("url", "")))
            ftype = f.get("data_type", f.get("dataType", ""))
            if not furl:
                # construct download URL
                furl = f"{BASE}/v2/dataset/{acc}/file/{urllib.parse.quote(fname)}/"
            # Only download CSV/TSV/TXT files that look like data
            if any(fname.endswith(ext) for ext in [".csv", ".tsv", ".txt"]) and not fname.endswith("README"):
                if not any(dt in fname.lower() for dt in downloaded):
                    try:
                        req = urllib.request.Request(furl, headers={"User-Agent": "Biomni-OSDR-Miner/1.0"})
                        with urllib.request.urlopen(req, timeout=120) as resp:
                            content = resp.read().decode("utf-8", errors="replace")
                        if len(content) > 100:
                            safe_fname = re.sub(r'[^a-zA-Z0-9._]', '_', fname)
                            fpath = os.path.join(RAW_DIR, f"{acc}_{safe_fname}")
                            save_text(content, fpath)
                            downloaded.append(fname)
                            print(f"  downloaded file: {fname} ({len(content)} chars)")
                    except Exception as e:
                        pass  # skip files that fail

        inventory.append({
            "accession": acc,
            "description": desc,
            "is_i4": is_i4,
            "n_samples": n_samples,
            "tissues": "; ".join(tissues),
            "n_files": len(files),
            "downloaded_types": "; ".join(downloaded),
        })
        time.sleep(1)  # be polite

    # ── Step 4: Save inventory ──
    inv_path = os.path.join(RAW_DIR, "dataset_inventory.csv")
    with open(inv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["accession", "description", "is_i4",
                                                "n_samples", "tissues", "n_files",
                                                "downloaded_types"])
        writer.writeheader()
        writer.writerows(inventory)
    print(f"\n=== Inventory saved to {inv_path} ===")
    print(f"Total studies processed: {len(inventory)}")

    # Print summary
    print("\n=== SUMMARY ===")
    for inv in inventory:
        print(f"  {inv['accession']}: {inv['n_samples']} samples, "
              f"downloaded: {inv['downloaded_types'][:80]}")


if __name__ == "__main__":
    main()

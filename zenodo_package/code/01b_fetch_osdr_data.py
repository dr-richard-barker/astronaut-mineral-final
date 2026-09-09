#!/usr/bin/env python3
"""
01b_fetch_osdr_data.py — Stage 1 (revised): Discover and download astronaut + rodent
omics data from NASA OSDR using the REST interface (more reliable than query API).

Outputs:
  /mnt/shared-workspace/shared/osdr_raw/dataset_inventory.csv
  /mnt/shared-workspace/shared/osdr_raw/<accession>_<filename>  (downloaded files)
"""
import os, sys, json, time, urllib.parse, urllib.request, csv, io, re

BASE = "https://visualization.osdr.nasa.gov/biodata/api"
RAW_DIR = "/mnt/shared-workspace/shared/osdr_raw"
os.makedirs(RAW_DIR, exist_ok=True)

I4_STUDIES = ["OSD-569","OSD-570","OSD-571","OSD-572","OSD-574","OSD-575","OSD-630","OSD-656"]
TARGET_TISSUES = ["liver","blood","muscle","bone","kidney","heart","spleen","thymus","eye","brain","adipose","pancreas","lung","skin"]
# Keywords for files we want to download
DOWNLOAD_KEYWORDS = ["expression","differential","abundance","metabolic","count","deseq","edger","processed","transformed","gene","protein","metabolite","cmp","panel","cytokine"]
SKIP_KEYWORDS = ["multiqc","md5sum",".zip","fastq","bam","bai","vcf","report","image","pdf","png","jpg","jpeg","README",".log",".html",".txt"]


def rest_get(path, timeout=120):
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Biomni-OSDR-Miner/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"  ERROR: {e}")
                return None


def download_file(url, filepath, timeout=180):
    req = urllib.request.Request(url, headers={"User-Agent": "Biomni-OSDR-Miner/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(filepath, "wb") as f:
            f.write(data)
        return len(data)
    except Exception as e:
        print(f"    download failed: {e}")
        return 0


def get_dataset_meta(accession):
    """Get REST metadata for a dataset — returns dict with tissue, title, etc."""
    text = rest_get(f"/v2/dataset/{accession}/")
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data.get(accession, {}).get("metadata", {})
    except json.JSONDecodeError:
        return {}


def get_file_list(accession):
    """Get file listing for a dataset — returns dict {filename: download_url}."""
    text = rest_get(f"/v2/dataset/{accession}/files/")
    if not text:
        return {}
    try:
        data = json.loads(text)
        files = data.get(accession, {}).get("files", {})
        result = {}
        for fname, finfo in files.items():
            url = finfo.get("URL", "")
            result[fname] = url
        return result
    except json.JSONDecodeError:
        return {}


def should_download(fname):
    """Decide if a file should be downloaded based on name."""
    fl = fname.lower()
    # Skip unwanted
    for kw in SKIP_KEYWORDS:
        if kw in fl:
            return False
    # Want data files
    for kw in DOWNLOAD_KEYWORDS:
        if kw in fl:
            return True
    return False


def main():
    # ── Step 1: Read rodent RNA-seq accessions ──
    with open("/tmp/rodent_rnaseq.txt") as f:
        rodent_accs = [l.strip() for l in f if l.strip()]
    print(f"Rodent RNA-seq studies to screen: {len(rodent_accs)}")

    # ── Step 2: Screen rodent studies by tissue ──
    print("\n=== Screening rodent studies by tissue ===")
    selected_rodent = []
    all_rodent_meta = {}

    for acc in rodent_accs:
        meta = get_dataset_meta(acc)
        tissue = meta.get("material type", "")
        if isinstance(tissue, list):
            tissue = "; ".join(str(t) for t in tissue)
        title = meta.get("study title", "")
        if isinstance(title, list):
            title = "; ".join(str(t) for t in title)
        organism = meta.get("organism", "")
        if isinstance(organism, list):
            organism = "; ".join(str(t) for t in organism)
        all_rodent_meta[acc] = {"tissue": str(tissue), "title": str(title), "organism": str(organism)}

        tissue_lower = tissue.lower() if tissue else ""
        if any(t in tissue_lower for t in TARGET_TISSUES):
            selected_rodent.append(acc)
            print(f"  ✓ {acc}: tissue={tissue}")
        time.sleep(0.3)

    print(f"\nSelected {len(selected_rodent)} rodent studies with target tissues")

    # Also keep studies with no tissue listed but interesting titles
    for acc, m in all_rodent_meta.items():
        if acc not in selected_rodent and not m["tissue"]:
            title_lower = m["title"].lower()
            if any(t in title_lower for t in TARGET_TISSUES):
                selected_rodent.append(acc)
                print(f"  + {acc}: title mentions tissue: {m['title'][:60]}")

    print(f"\nFinal selected rodent studies: {len(selected_rodent)}")

    # ── Step 3: Download files for I4 + selected rodent studies ──
    all_studies = I4_STUDIES + selected_rodent
    inventory = []

    print(f"\n=== Downloading files for {len(all_studies)} studies ===")
    for acc in all_studies:
        is_i4 = acc in I4_STUDIES
        meta = all_rodent_meta.get(acc) or get_dataset_meta(acc)
        tissue = meta.get("tissue", "") if isinstance(meta, dict) else ""
        if not tissue and isinstance(meta, dict):
            t = meta.get("material type", "")
            tissue = "; ".join(str(x) for x in t) if isinstance(t, list) else str(t)
        title = meta.get("title", "") if isinstance(meta, dict) else ""

        print(f"\n--- {acc} ({'I4' if is_i4 else 'rodent'}) tissue={tissue} ---")
        files = get_file_list(acc)
        print(f"  Files: {len(files)}")

        downloaded = []
        for fname, url in files.items():
            if should_download(fname) and url:
                safe_fname = re.sub(r'[^a-zA-Z0-9._-]', '_', fname)
                fpath = os.path.join(RAW_DIR, f"{acc}_{safe_fname}")
                if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
                    downloaded.append(fname)
                    continue
                size = download_file(url, fpath)
                if size > 100:
                    downloaded.append(fname)
                    print(f"    ✓ {fname} ({size} bytes)")
                else:
                    print(f"    ✗ {fname} (too small or failed)")
            time.sleep(0.2)

        inventory.append({
            "accession": acc,
            "is_i4": is_i4,
            "tissue": tissue,
            "title": title[:80],
            "n_files_total": len(files),
            "n_files_downloaded": len(downloaded),
            "downloaded_files": "; ".join(downloaded[:10]),
        })
        time.sleep(0.5)

    # ── Step 4: Save inventory ──
    inv_path = os.path.join(RAW_DIR, "dataset_inventory.csv")
    with open(inv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["accession","is_i4","tissue","title",
                                                "n_files_total","n_files_downloaded","downloaded_files"])
        writer.writeheader()
        writer.writerows(inventory)
    print(f"\n=== Inventory saved: {inv_path} ===")
    print(f"Total studies: {len(inventory)}")
    total_dl = sum(i["n_files_downloaded"] for i in inventory)
    print(f"Total files downloaded: {total_dl}")


if __name__ == "__main__":
    main()

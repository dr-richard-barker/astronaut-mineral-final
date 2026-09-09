#!/usr/bin/env python3
"""01c_rodent_download.py — Screen rodent RNA-seq studies by tissue and download files."""
import os, json, time, urllib.request, urllib.parse, re, csv

BASE = "https://visualization.osdr.nasa.gov/biodata/api"
RAW_DIR = "/mnt/shared-workspace/shared/osdr_raw"
TARGET_TISSUES = ["liver","blood","muscle","bone","kidney","heart","spleen","thymus","eye","brain","adipose","pancreas","lung","skin","leukocyte","pbmc","mononuclear"]
SKIP = ["multiqc","md5sum",".zip","fastq","bam","bai","vcf","report","image","pdf","png","jpg","jpeg","readme",".log",".html",".faa",".fasta",".gff",".fa",".gtf",".bed"]
WANT = ["expression","differential","abundance","count","deseq","edger","processed","transformed","gene","tpm","fpkm","normalized","unnormalized"]

def rest_get(path, timeout=90):
    url = f"{BASE}{path}"
    req = urllib.request.Request(url, headers={"User-Agent":"Biomni-OSDR-Miner/1.0"})
    for a in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r: return r.read().decode("utf-8","replace")
        except: time.sleep(3) if a<2 else None
    return None

def dl(url, fp, timeout=120):
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"Biomni-OSDR-Miner/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r: d=r.read()
        with open(fp,"wb") as f: f.write(d)
        return len(d)
    except: return 0

with open("/tmp/rodent_rnaseq.txt") as f:
    accs = [l.strip() for l in f if l.strip()]

inventory = []
selected = []

# Screen by tissue
for acc in accs:
    text = rest_get(f"/v2/dataset/{acc}/")
    if not text: continue
    try: meta = json.loads(text).get(acc,{}).get("metadata",{})
    except: continue
    tissue = meta.get("material type","")
    if isinstance(tissue, list): tissue = "; ".join(str(t) for t in tissue)
    title = meta.get("study title","")
    if isinstance(title, list): title = "; ".join(str(t) for t in title)
    tl = str(tissue).lower()
    if any(t in tl for t in TARGET_TISSUES) or any(t in str(title).lower() for t in TARGET_TISSUES):
        selected.append(acc)
        inventory.append({"accession":acc,"tissue":str(tissue),"title":str(title)[:80],"n_files":0,"n_dl":0})
    time.sleep(0.2)

print(f"Selected {len(selected)} rodent studies")

# Download files for selected studies
for i, acc in enumerate(selected):
    text = rest_get(f"/v2/dataset/{acc}/files/")
    if not text: continue
    try: files = json.loads(text).get(acc,{}).get("files",{})
    except: continue
    dl_count = 0
    for fname, finfo in files.items():
        url = finfo.get("URL","")
        fl = fname.lower()
        if not url: continue
        if any(s in fl for s in SKIP): continue
        if not any(w in fl for w in WANT): continue
        safe = re.sub(r'[^a-zA-Z0-9._-]','_',fname)
        fp = os.path.join(RAW_DIR, f"{acc}_{safe}")
        if os.path.exists(fp) and os.path.getsize(fp)>100: dl_count+=1; continue
        sz = dl(url, fp)
        if sz > 100: dl_count += 1
        time.sleep(0.1)
    inventory[i]["n_files"] = len(files)
    inventory[i]["n_dl"] = dl_count
    print(f"  {acc}: {dl_count}/{len(files)} files downloaded")
    time.sleep(0.3)

with open(os.path.join(RAW_DIR,"rodent_inventory.csv"),"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=["accession","tissue","title","n_files","n_dl"])
    w.writeheader(); w.writerows(inventory)
print(f"\nDone. Inventory saved. {sum(i['n_dl'] for i in inventory)} files downloaded.")

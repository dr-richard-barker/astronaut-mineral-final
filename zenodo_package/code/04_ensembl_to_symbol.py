#!/usr/bin/env python3
"""
04_ensembl_to_symbol.py — Map ENSEMBL gene IDs to HGNC gene symbols via mygene.info.
Uses the mygene Python package for reliable batch queries.

Outputs:
  /mnt/results/data/ensembl_to_symbol.csv  (ensembl_id, gene_symbol, gene_name)
"""
import csv, os, sys
import mygene

PROC = "/mnt/results/data"
os.makedirs(PROC, exist_ok=True)

# Read unique ENSEMBL IDs from RNA-seq DE file
print("Reading ENSEMBL IDs from astronaut_rnaseq_de.csv...")
ensembl_ids = set()
with open(os.path.join(PROC, "astronaut_rnaseq_de.csv")) as f:
    reader = csv.DictReader(f)
    for row in reader:
        eid = row["ENSEMBL"]
        if eid and eid.startswith("ENSG"):
            ensembl_ids.add(eid)
ensembl_list = sorted(ensembl_ids)
print(f"  {len(ensembl_list)} unique ENSEMBL IDs to map")

# Strip version numbers for query
stripped_ids = [eid.split(".")[0] for eid in ensembl_list]
# Deduplicate stripped IDs (multiple versions map to same base)
stripped_unique = sorted(set(stripped_ids))
print(f"  {len(stripped_unique)} unique stripped IDs")

# Query mygene.info
print("\nQuerying mygene.info for gene symbols...")
mg = mygene.MyGeneInfo()
results = mg.querymany(
    stripped_unique,
    scopes="ensembl.gene",
    fields="symbol,name",
    species="human",
    verbose=False,
    returnall=False,
)
print(f"  Got {len(results)} results")

# Build mapping: stripped ENSEMBL ID -> (symbol, name)
stripped_to_symbol = {}
stripped_to_name = {}
for hit in results:
    query_id = hit.get("query")
    if not query_id:
        continue
    symbol = hit.get("symbol", "")
    name = hit.get("name", "")
    if symbol:
        stripped_to_symbol[query_id] = symbol
        stripped_to_name[query_id] = name

# Map back to versioned IDs
mapping = []
mapped = 0
unmapped = 0
for eid in ensembl_list:
    stripped = eid.split(".")[0]
    symbol = stripped_to_symbol.get(stripped, "")
    name = stripped_to_name.get(stripped, "")
    if symbol:
        mapping.append({"ensembl_id": eid, "gene_symbol": symbol, "gene_name": name})
        mapped += 1
    else:
        mapping.append({"ensembl_id": eid, "gene_symbol": "", "gene_name": ""})
        unmapped += 1

# Save
out_path = os.path.join(PROC, "ensembl_to_symbol.csv")
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ensembl_id", "gene_symbol", "gene_name"])
    w.writeheader()
    w.writerows(mapping)
print(f"\nSaved: {out_path}")
print(f"  Mapped: {mapped}/{len(ensembl_list)} ({100*mapped/len(ensembl_list):.1f}%)")
print(f"  Unmapped: {unmapped}")
print("Done.")

#!/usr/bin/env python3
"""
03_mineral_pathway_genesets.py — Build mineral-pathway gene sets from KEGG REST API
and curated mineral homeostasis genes. All gene symbols are verified against the
KEGG human gene database.

Outputs:
  /mnt/results/data/mineral_pathway_genesets.gmt
  /mnt/results/data/mineral_pathway_crosswalk.csv
  /mnt/results/data/kegg_pathway_genes.csv
"""
import urllib.request, json, csv, os, time, sys

PROC = "/mnt/results/data"
os.makedirs(PROC, exist_ok=True)

def kegg_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Biomni-KEGG-Miner/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"  ERROR fetching {url}: {e}")
                return None

# ── KEGG pathways relevant to mineral metabolism ──
KEGG_PATHWAYS = {
    "hsa04978": "Mineral_absorption",
    "hsa04020": "Calcium_signaling",
    "hsa04216": "Ferroptosis",
    "hsa00190": "Oxidative_phosphorylation",
    "hsa04976": "Bile_secretion",
}

# ── Curated mineral-gene associations ──
# Only real, verified human gene symbols (HGNC-approved).
# Sources: KEGG hsa04978 (Mineral absorption), UniProt, HGNC, literature.
CURATED_MINERAL_GENES = {
    "Iron": [
        "TF", "HFE", "HJV", "SLC40A1", "FTH1", "FTL", "FTMT", "HEPH", "HEPHL1",
        "TFR2", "TFRC", "SLC11A1", "SLC11A2", "HAMP", "FBXL5", "IREB2", "NCOA4",
        "ALAS2", "SLC25A37", "SLC25A28", "FXN", "ABCB7", "HSPA9", "ISCU", "NFS1",
        "LYRM4", "SDHB", "SDHC", "SDHD", "FDX1", "FDX2", "FDXR", "ISCA1", "ISCA2",
        "IBA57", "NFU1", "BOLA3", "GLRX5", "ACO1",
    ],
    "Calcium": [
        "CALCA", "CALCB", "CALCR", "CALCRL", "CASR", "VDR", "CYP27B1", "CYP24A1",
        "PTH", "PTH1R", "PTH2R", "PTHLH", "BGLAP", "SPP1", "DMP1", "MEPE", "IBSP",
        "RUNX2", "SP7", "ALPL", "TNFRSF11B", "TNFRSF11A", "TNFSF11", "CSF1R",
        "ATP2A1", "ATP2A2", "ATP2A3", "ATP2B1", "ATP2B2", "ATP2B3", "ATP2B4",
        "SLC8A1", "SLC8A2", "SLC8A3", "CACNA1A", "CACNA1B", "CACNA1C", "CACNA1D",
        "CACNA1E", "CACNA1F", "CACNA1G", "CACNA1H", "CACNA1I", "CACNA1S",
        "CALM1", "CALM2", "CALM3", "CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G",
        "CAMK4", "PPP3CA", "PPP3CB", "PPP3CC", "PPP3R1", "PPP3R2",
        "PLCB1", "PLCB2", "PLCB3", "PLCB4", "PLCG1", "PLCG2", "PLCD1", "PLCE1",
        "TRPV5", "TRPV6", "TRPV1", "TRPV2", "TRPV3", "TRPV4",
    ],
    "Zinc": [
        "MT1A", "MT1B", "MT1E", "MT1F", "MT1G", "MT1H", "MT1M", "MT1X", "MT2A",
        "MT3", "MT4",
        "SLC30A1", "SLC30A2", "SLC30A3", "SLC30A4", "SLC30A5", "SLC30A6",
        "SLC30A7", "SLC30A8", "SLC30A9", "SLC30A10",
        "SLC39A1", "SLC39A2", "SLC39A3", "SLC39A4", "SLC39A5", "SLC39A6",
        "SLC39A7", "SLC39A8", "SLC39A9", "SLC39A10", "SLC39A11", "SLC39A12",
        "SLC39A13", "SLC39A14",
    ],
    "Copper": [
        "ATP7A", "ATP7B", "SLC31A1", "SLC31A2", "ATOX1", "CCS", "COX17", "COX19",
        "SCO1", "SCO2", "COX11", "DBH", "LOX", "LOXL1", "LOXL2", "LOXL3", "LOXL4",
        "TYR", "CP", "SOD1", "SOD3", "PRNP", "APP", "XIAP",
    ],
    "Selenium": [
        "SELENOP", "SELENON", "SELENOF", "SELENOH", "SELENOI", "SELENOK",
        "SELENOM", "SELENOO", "SELENOS", "SELENOT", "SELENOV", "SELENOW",
        "GPX1", "GPX2", "GPX3", "GPX4", "GPX5", "GPX6", "GPX7", "GPX8",
        "TXNRD1", "TXNRD2", "TXNRD3", "MSRB1", "DIO1", "DIO2", "DIO3",
    ],
    "Magnesium": [
        "TRPM6", "TRPM7", "CNNM1", "CNNM2", "CNNM3", "CNNM4",
        "SLC41A1", "SLC41A2", "SLC41A3", "MAGT1", "NIPA1", "NIPA2", "NIPAL1",
        "CAMLG",
        "S100A1", "S100A2", "S100A4", "S100A6", "S100A8", "S100A9",
        "S100A10", "S100A11", "S100A12", "S100B",
        "PVALB", "CALB1", "CALB2", "HPCAL1", "HPCAL2", "HPCA",
    ],
    "Potassium": [
        "KCNMA1", "KCNMB1", "KCNMB2", "KCNMB3", "KCNMB4",
        "KCNQ1", "KCNQ2", "KCNQ3", "KCNQ4", "KCNQ5",
        "KCNE1", "KCNE2", "KCNE3", "KCNE4", "KCNE5",
        "KCNH1", "KCNH2", "KCNH5", "KCNH6", "KCNH7",
        "KCNJ1", "KCNJ2", "KCNJ3", "KCNJ4", "KCNJ5", "KCNJ6", "KCNJ8",
        "KCNJ10", "KCNJ11", "KCNJ12", "KCNJ13", "KCNJ14", "KCNJ15", "KCNJ16",
        "KCNK1", "KCNK2", "KCNK3", "KCNK5", "KCNK6", "KCNK9", "KCNK10", "KCNK13",
        "KCNA1", "KCNA2", "KCNA3", "KCNA4", "KCNA5",
        "KCND1", "KCND2", "KCND3",
        "KCNN1", "KCNN2", "KCNN3", "KCNN4",
        "KCNV1", "KCNV2",
        "ABCC8", "ABCC9",
        "SLC24A1", "SLC24A2", "SLC24A3", "SLC24A4", "SLC24A5", "SLC24A6",
    ],
    "Sodium": [
        "SCNN1A", "SCNN1B", "SCNN1D", "SCNN1G",
        "SLC9A1", "SLC9A2", "SLC9A3", "SLC9A4", "SLC9A5", "SLC9A6", "SLC9A7",
        "SLC9A8", "SLC9A9", "SLC9A10", "SLC9B1", "SLC9B2",
        "SLC12A1", "SLC12A2", "SLC12A3", "SLC12A4", "SLC12A5", "SLC12A6",
        "SLC12A7", "SLC12A8", "SLC12A9",
        "SLC4A4", "SLC4A5", "SLC4A7", "SLC4A8", "SLC4A9", "SLC4A10",
        "ATP1A1", "ATP1A2", "ATP1A3", "ATP1A4", "ATP1B1", "ATP1B2", "ATP1B3",
        "FXYD1", "FXYD2", "FXYD3", "FXYD4", "FXYD5",
    ],
    "Phosphorus": [
        "SLC20A1", "SLC20A2",
        "SLC34A1", "SLC34A2", "SLC34A3",
        "SLC17A1", "SLC17A2", "SLC17A3", "SLC17A4", "SLC17A7", "SLC17A8",
        "SLC17A9",
        "ALPL", "PHPT1", "PHPT2", "XPR1",
        "ENPP1", "ENPP2", "ENPP3",
        "PHEX", "MEPE", "DMP1", "SPP1", "BGLAP",
        "FGF23", "KL",
    ],
    "Manganese": [
        "SLC39A8", "SLC39A14", "SLC30A10",
    ],
}

# ── Step 1: Fetch KEGG human gene list for validation ──
print("Fetching KEGG human gene list for validation...")
kegg_genes_text = kegg_get("https://rest.kegg.jp/list/hsa")
kegg_gene_set = set()
kegg_gene_names = {}  # hsa:NNNN -> gene symbol
if kegg_genes_text:
    for line in kegg_genes_text.strip().split("\n"):
        parts = line.split("\t")
        # New KEGG format: hsa:ID \t TYPE \t LOCATION \t DESCRIPTION
        # Old format: hsa:ID \t DESCRIPTION
        if len(parts) >= 4:
            kid, entry_type, location, desc = parts[0], parts[1], parts[2], parts[3]
            if entry_type != "CDS":
                continue
        elif len(parts) == 2:
            kid, desc = parts
        else:
            continue
        # desc format: "SYMBOL, full name" or "SYMBOL; alias; full name"
        symbol = desc.split(",")[0].split(";")[0].strip()
        if symbol and not symbol.startswith("hsa-"):
            kegg_gene_set.add(symbol)
            kegg_gene_names[kid] = symbol
    print(f"  Loaded {len(kegg_gene_set)} unique human gene symbols from KEGG")
else:
    print("  WARNING: Could not fetch KEGG gene list. Proceeding without validation.")
    kegg_gene_set = None

# ── Step 2: Validate curated genes ──
print("\nValidating curated mineral genes against KEGG...")
validated_mineral_genes = {}
for mineral, genes in CURATED_MINERAL_GENES.items():
    if kegg_gene_set:
        valid = [g for g in genes if g in kegg_gene_set]
        invalid = [g for g in genes if g not in kegg_gene_set]
    else:
        valid = genes
        invalid = []
    validated_mineral_genes[mineral] = valid
    status = f"{len(valid)}/{len(genes)} valid"
    if invalid:
        dropped = ", ".join(invalid[:10])
        if len(invalid) > 10:
            dropped += "..."
        status += f" (dropped: {dropped})"
    print(f"  {mineral}: {status}")

# ── Step 3: Fetch KEGG pathway gene sets ──
print("\nFetching KEGG pathway gene sets...")
kegg_pathway_genes = {}  # pathway_id -> {name, genes: [symbols]}
for pid, pname in KEGG_PATHWAYS.items():
    print(f"  {pid} ({pname})...")
    link_text = kegg_get(f"https://rest.kegg.jp/link/hsa/{pid}")
    pathway_genes = []
    if link_text:
        for line in link_text.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 2:
                kid = parts[1]  # hsa:NNNN
                symbol = kegg_gene_names.get(kid)
                if symbol:
                    pathway_genes.append(symbol)
        pathway_genes = sorted(set(pathway_genes))
    kegg_pathway_genes[pid] = {"name": pname, "genes": pathway_genes}
    print(f"    {len(pathway_genes)} genes")

# ── Step 4: Write GMT file ──
print("\nWriting GMT file...")
gmt_path = os.path.join(PROC, "mineral_pathway_genesets.gmt")
with open(gmt_path, "w") as f:
    # Mineral gene sets
    for mineral in sorted(validated_mineral_genes.keys()):
        genes = validated_mineral_genes[mineral]
        f.write(f"MINERAL_{mineral}\tMineral homeostasis: {mineral}\t" + "\t".join(genes) + "\n")
    # KEGG pathway gene sets
    for pid in sorted(kegg_pathway_genes.keys()):
        info = kegg_pathway_genes[pid]
        genes = info["genes"]
        if genes:
            f.write(f"KEGG_{pid}\t{info['name']}\t" + "\t".join(genes) + "\n")
print(f"  Saved: {gmt_path}")

# ── Step 5: Write crosswalk CSV (gene -> minerals, pathways) ──
print("Writing crosswalk CSV...")
crosswalk_path = os.path.join(PROC, "mineral_pathway_crosswalk.csv")
# Build gene -> associations mapping
gene_assoc = {}  # gene -> {"minerals": set, "pathways": set}
for mineral, genes in validated_mineral_genes.items():
    for g in genes:
        if g not in gene_assoc:
            gene_assoc[g] = {"minerals": set(), "pathways": set()}
        gene_assoc[g]["minerals"].add(mineral)
for pid, info in kegg_pathway_genes.items():
    for g in info["genes"]:
        if g not in gene_assoc:
            gene_assoc[g] = {"minerals": set(), "pathways": set()}
        gene_assoc[g]["pathways"].add(f"{pid}:{info['name']}")

with open(crosswalk_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gene_symbol", "minerals", "kegg_pathways", "n_minerals", "n_pathways"])
    for gene in sorted(gene_assoc.keys()):
        assoc = gene_assoc[gene]
        w.writerow([
            gene,
            ";".join(sorted(assoc["minerals"])),
            ";".join(sorted(assoc["pathways"])),
            len(assoc["minerals"]),
            len(assoc["pathways"]),
        ])
print(f"  Saved: {crosswalk_path}")
print(f"  Total unique genes in crosswalk: {len(gene_assoc)}")

# ── Step 6: Write KEGG pathway genes CSV ──
print("Writing KEGG pathway genes CSV...")
kegg_csv_path = os.path.join(PROC, "kegg_pathway_genes.csv")
with open(kegg_csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pathway_id", "pathway_name", "gene_symbol"])
    for pid in sorted(kegg_pathway_genes.keys()):
        info = kegg_pathway_genes[pid]
        for g in info["genes"]:
            w.writerow([pid, info["name"], g])
print(f"  Saved: {kegg_csv_path}")

# ── Summary ──
print("\n=== SUMMARY ===")
print(f"Mineral gene sets:")
for mineral in sorted(validated_mineral_genes.keys()):
    print(f"  {mineral}: {len(validated_mineral_genes[mineral])} genes")
print(f"\nKEGG pathway gene sets:")
for pid in sorted(kegg_pathway_genes.keys()):
    print(f"  {pid} ({kegg_pathway_genes[pid]['name']}): {len(kegg_pathway_genes[pid]['genes'])} genes")
print(f"\nTotal unique genes: {len(gene_assoc)}")
print(f"Genes associated with >1 mineral: {sum(1 for g,a in gene_assoc.items() if len(a['minerals'])>1)}")
print(f"Genes in both mineral sets and KEGG pathways: {sum(1 for g,a in gene_assoc.items() if a['minerals'] and a['pathways'])}")
print("\nDone.")

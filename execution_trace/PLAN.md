# Plan: Graphical Abstract for NHANES-First Analytical Framework

## Summary
Generate two graphical abstracts (landscape + portrait) summarizing the NHANES-first multi-omics spaceflight analysis framework and all major findings. Both are purely schematic (boxes, arrows, icons, short labels — no embedded data plots or numbers). Scientific minimal color scheme: white background, muted blues/grays for structure, red/green accents for up/down regulation.

## Tool
`GenerateImage` (deferred — load via ToolSearch first). This is a conceptual/schematic figure, not a data plot, so GenerateImage is the correct tool per visualization guidelines.

## Design Specification

### Color Scheme (Scientific Minimal)
- Background: white (#FFFFFF)
- Structure (boxes, arrows, borders): muted slate blue (#3B6E8F) and gray (#6B7280)
- Up-regulation / elevated: muted red (#C0392B)
- Down-regulation / depressed: muted green (#27AE60)
- NHANES reference layer: light blue fill (#D6E4F0)
- Astronaut layer: light gray fill (#F0F0F0)
- Text: dark gray (#1F2937)

### Version 1: Landscape (Left-to-Right Pipeline)
Five vertical columns connected by left-to-right arrows, each column representing an analysis stage:

**Column 1 — NHANES Population Reference**
- Box: "NHANES 2013–2018, n=5,748 (age 40–60)"
- Sub-box: "15 clinical variables + 4 iron status markers"
- Icon: population/demographics symbol

**Column 2 — Astronaut Clinical Deviations**
- Box: "Inspiration4, n=4 astronauts, 7 timepoints"
- Sub-boxes (with up/down arrows):
  - ↑ MPV (elevated)
  - ↑ Calcium (elevated)
  - ↓ Bicarbonate (depressed)
  - Hemoglobin (modest deviation)
- Arrow from Column 1 labeled "z-scores vs population norms"

**Column 3 — Mineral-Pathway Transcriptomics**
- Box: "801 mineral-pathway genes, 50 DE genes"
- Sub-boxes (red/green):
  - ↓ ALAS2 (iron/heme biosynthesis)
  - ↑ HFE (iron sensing)
  - ↓ CACNA1D (calcium channel)
  - ↑ APP, ↓ LOX/LOXL1 (copper)
  - ↓ S100A8/A9 (magnesium)
- Side box: "Iron status reference: Hb–iron r=0.42"
- Arrow from Column 2 labeled "clinical → molecular"

**Column 4 — ML & Deep Learning**
- Box: "Mineral-pathway genes predict serum minerals"
- Sub-boxes:
  - "K⁺ r=0.50, Na⁺ r=0.46 (LOO-CV)"
  - "Stability selection: HFE 88%, KCNE1 94%"
- Box: "DAE latent space (I4+AX-1, n=6)"
- Sub-box: "Flight vs ground AUC=0.73, ensemble AUC=0.75"
- Arrow from Column 3 labeled "predictive modeling"

**Column 5 — Cross-Mission & Cross-Species Validation**
- Box: "JAXA6 cross-mission (n=6)"
- Sub-box: "ALAS2 FDR=0.021 (meta-analysis)"
- Box: "28 rodent datasets"
- Sub-box: "Muscle concordance (soleus, p=0.009)"
- Arrow from Column 4 labeled "validation"

### Version 2: Portrait (Top-Down Waterfall)
Same five stages as vertical layers stacked top-to-bottom, with downward arrows connecting them. Same content and color scheme, just reoriented for portrait/standard journal format. Key findings listed within each layer as labeled sub-boxes with up/down arrows.

### Output
- Save both as PNG to `/mnt/results/figures/graphical_abstract_landscape.png` and `/mnt/results/figures/graphical_abstract_portrait.png`
- Copy both to `/mnt/results/manuscript/figures/`
- No SVG needed (GenerateImage produces raster output)

## Assumptions
- GenerateImage produces a single raster image per call; two separate calls for landscape and portrait
- The schematic style uses clean boxes, arrows, and short text labels — no data plots embedded
- "All major findings" means representing every key result as a labeled element, but keeping text concise (schematic, not a results table)
- No manuscript LaTeX edits needed for this task — graphical abstracts are standalone deliverables (can be added to manuscript later if requested)

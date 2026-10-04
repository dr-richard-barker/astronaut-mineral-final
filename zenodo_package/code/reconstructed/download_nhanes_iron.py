"""Downloads NHANES serum iron, ferritin and TIBC files; some cycles returned 404.
Serum iron (LBXSIR) comes from the Standard Biochemistry Profile (BIOPRO);
ferritin (LBXFER) from the Ferritin files (FERTIN); transferrin saturation
(LBDPCT) and TIBC (LBDTIB) from the TIBC/Transferrin Saturation file (FETIB).
The NHANES 2013-2014 cycle has no FETIB or FERTIN files (404). Reference
statistics (age 40-60, all sexes) are appended to nhanes_reference_stats.csv.

RECONSTRUCTION (Oct 2026) of a lost original; see FIDELITY.md.
"""
import argparse
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# NHANES cycle letter -> begin year (URL pattern uses the cycle BEGIN year)
CYCLES = {"H": "2013", "I": "2015", "J": "2017"}
BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{file}.{ext}"

# file stem -> cycles that exist (others 404)
FILES = {"BIOPRO": ["H", "I", "J"], "CBC": ["H", "I", "J"], "DEMO": ["H", "I", "J"],
         "FETIB": ["J"], "FERTIN": ["I", "J"]}

IRON_VARS = {
    "SERUM IRON": ("LBXSIR", "Serum Iron", "ug/dL"),
    "FERRITIN": ("LBXFER", "Ferritin", "ng/mL"),
    "TRANSFERRIN SATURATION": ("LBDPCT", "Transferrin Saturation", "%"),
    "TIBC": ("LBDTIB", "Total Iron Binding Capacity", "ug/dL"),
}


def download(raw_dir: Path) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for stem, cycles in FILES.items():
        for c in cycles:
            fname = f"{stem}_{c}"
            dest = raw_dir / f"{fname}.XPT"
            if not dest.exists():
                got = False
                for ext in ("XPT", "xpt"):  # server is case-sensitive per file
                    url = BASE.format(year=CYCLES[c], file=fname, ext=ext)
                    try:
                        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                        body = urlopen(req, timeout=120).read()
                        if body[:4] != b"<!DO":  # guard against HTML error pages
                            dest.write_bytes(body)
                            print(f"Downloaded {fname}.XPT ({len(body):,} bytes)")
                            got = True
                            break
                        print(f"FAILED (HTML error page): {fname}.{ext}")
                    except HTTPError as e:
                        print(f"FAILED ({e.code}): {fname}.{ext}")
                if not got:
                    continue
            paths.setdefault(stem, []).append((c, dest))
    return paths


def read_xpt(path: Path) -> pd.DataFrame:
    return pd.read_sas(path, format="xport")


def main():
    ap = argparse.ArgumentParser()
    from _common import DATA, OUT
    ap.add_argument("--raw-dir", default=str(OUT / "nhanes_raw"))
    ap.add_argument("--data-dir", default=str(DATA), help="deposited nhanes_reference_stats.csv (read)")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    raw, data, out = Path(a.raw_dir), Path(a.data_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = download(raw)

    # Merge serum iron (BIOPRO) with demographics for the age filter
    frames = []
    for c, p in paths.get("BIOPRO", []):
        bio = read_xpt(p)[["SEQN", "LBXSIR"]]
        demo = read_xpt(raw / f"DEMO_{c}.XPT")[["SEQN", "RIDAGEYR"]]
        frames.append(bio.merge(demo, on="SEQN"))
    iron = pd.concat(frames, ignore_index=True)
    iron = iron[(iron.RIDAGEYR >= 40) & (iron.RIDAGEYR <= 60)]

    # Ferritin (FERTIN)
    fer = pd.concat([read_xpt(p)[["SEQN", "LBXFER"]] for c, p in paths.get("FERTIN", [])],
                    ignore_index=True)
    iron = iron.merge(fer, on="SEQN", how="left")

    # TIBC + transferrin saturation (FETIB, 2017-2018 only)
    if paths.get("FETIB"):
        fet = read_xpt(paths["FETIB"][0][1])[["SEQN", "LBDPCT", "LBDTIB"]]
        iron = iron.merge(fet, on="SEQN", how="left")

    stats_path = data / "nhanes_reference_stats.csv"
    stats = (pd.read_csv(stats_path) if stats_path.exists()
             else pd.DataFrame(columns=["astronaut_variable", "nhanes_code", "label",
                                        "units", "sex", "N", "mean", "sd", "median",
                                        "P1", "P2.5", "P5", "P10", "P25", "P50",
                                        "P75", "P90", "P95", "P97.5", "P99"]))
    stats = stats[stats.astronaut_variable.isin(IRON_VARS) == False].copy()  # noqa: E712

    pct = [1, 2.5, 5, 10, 25, 50, 75, 90, 95, 97.5, 99]
    for var, (code, label, units) in IRON_VARS.items():
        v = iron[code].dropna()
        row = {"astronaut_variable": var, "nhanes_code": code, "label": label,
               "units": units, "sex": "All", "N": len(v), "mean": v.mean(),
               "sd": v.std(), "median": v.median()}
        row.update({f"P{p}": np.percentile(v, p) for p in pct})
        stats = pd.concat([stats, pd.DataFrame([row])], ignore_index=True)
        print(f"{var}: N={len(v)}, mean={v.mean():.2f}, SD={v.std():.2f}, "
              f"P2.5={np.percentile(v, 2.5):.2f}, P97.5={np.percentile(v, 97.5):.2f}")

    stats.to_csv(out / "nhanes_reference_stats.csv", index=False)
    print(f"Saved {out / 'nhanes_reference_stats.csv'} ({len(stats)} rows)")


if __name__ == "__main__":
    main()

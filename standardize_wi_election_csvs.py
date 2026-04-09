#!/usr/bin/env python3
"""Create standardized county-level election CSVs for Wisconsin races in this repository.

Standardized schema:
county,election_date,race,dem_candidate,rep_candidate,dem_votes,rep_votes,other_votes,total_votes,source_csv
"""

from __future__ import annotations

import csv
from pathlib import Path

DATA_DIR = Path("data")

TEMPLATE_PATH = DATA_DIR / "county_results_standardized_template.csv"
PRES_IN = DATA_DIR / "Wisconsin Election Results - 2024 President.csv"
SENATE_IN = DATA_DIR / "wi_county_results_2024_2025.csv"
SUPREME_IN = DATA_DIR / "Wisconsin Election Results - 2025 Supreme Court.csv"

PRES_OUT = DATA_DIR / "wi_2024_president_standardized.csv"
SENATE_OUT = DATA_DIR / "wi_2024_senate_standardized.csv"
SUPREME_OUT = DATA_DIR / "wi_2025_supreme_standardized.csv"

FIELDS = [
    "county",
    "election_date",
    "race",
    "dem_candidate",
    "rep_candidate",
    "dem_votes",
    "rep_votes",
    "other_votes",
    "total_votes",
    "source_csv",
]


def to_int(value: str) -> int:
    return int(value.replace(",", "").strip())


def write_template() -> None:
    TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TEMPLATE_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        # Example row to document the expected schema.
        w.writerow(
            {
                "county": "Example County",
                "election_date": "YYYY-MM-DD",
                "race": "Example Race",
                "dem_candidate": "Dem Candidate",
                "rep_candidate": "Rep Candidate",
                "dem_votes": 0,
                "rep_votes": 0,
                "other_votes": 0,
                "total_votes": 0,
                "source_csv": "source-file.csv",
            }
        )


def standardize_president() -> None:
    rows = []
    with PRES_IN.open(newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        data = list(r)

    for row in data[2:]:
        if not row or not row[0].strip():
            continue
        county = row[0].strip()
        rep_votes = to_int(row[1])
        dem_votes = to_int(row[3])
        other_votes = to_int(row[5])
        total_votes = to_int(row[9])
        rows.append(
            {
                "county": county,
                "election_date": "2024-11-05",
                "race": "2024 Wisconsin Presidential",
                "dem_candidate": "Kamala Harris",
                "rep_candidate": "Donald Trump",
                "dem_votes": dem_votes,
                "rep_votes": rep_votes,
                "other_votes": other_votes,
                "total_votes": total_votes,
                "source_csv": PRES_IN.name,
            }
        )

    with PRES_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["county"]))


def standardize_senate() -> None:
    rows = []
    with SENATE_IN.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["race"].strip() != "2024 Wisconsin U.S. Senate":
                continue
            dem_votes = to_int(row["dem_votes"])
            rep_votes = to_int(row["rep_votes"])
            total_votes = to_int(row["total_votes"])
            rows.append(
                {
                    "county": row["county"].strip(),
                    "election_date": row["election_date"].strip(),
                    "race": row["race"].strip(),
                    "dem_candidate": row["dem_candidate"].strip(),
                    "rep_candidate": row["rep_candidate"].strip(),
                    "dem_votes": dem_votes,
                    "rep_votes": rep_votes,
                    "other_votes": max(0, total_votes - dem_votes - rep_votes),
                    "total_votes": total_votes,
                    "source_csv": SENATE_IN.name,
                }
            )

    with SENATE_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["county"]))


def standardize_supreme() -> None:
    rows = []
    with SUPREME_IN.open(newline="", encoding="utf-8-sig") as f:
        r = csv.reader(f)
        data = list(r)

    for row in data[2:]:
        if not row or not row[0].strip():
            continue
        county = row[0].strip()
        dem_votes = to_int(row[1])
        rep_votes = to_int(row[3])
        other_votes = to_int(row[5])
        total_votes = to_int(row[9])
        rows.append(
            {
                "county": county,
                "election_date": "2025-04-01",
                "race": "2025 Wisconsin Supreme Court",
                "dem_candidate": "Susan Crawford",
                "rep_candidate": "Brad Schimel",
                "dem_votes": dem_votes,
                "rep_votes": rep_votes,
                "other_votes": other_votes,
                "total_votes": total_votes,
                "source_csv": SUPREME_IN.name,
            }
        )

    with SUPREME_OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["county"]))


def main() -> None:
    write_template()
    standardize_president()
    standardize_senate()
    standardize_supreme()
    print("Wrote standardized template and race CSVs.")


if __name__ == "__main__":
    main()

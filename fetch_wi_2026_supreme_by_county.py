#!/usr/bin/env python3
"""Fetch 2026 Wisconsin Supreme Court county results from AP's election data feed.

Writes standardized CSV:
  data/wi_2026_supreme_standardized.csv

Also performs consistency checks between county sums and AP race summary/metadata.
"""

from __future__ import annotations

import csv
import gzip
import json
import urllib.request
from pathlib import Path

DETAIL_URL = "https://interactives.apelections.org/election-results/data-live/2026-04-07/results/races/WI/20260407WI50888/detail.json"
METADATA_URL = "https://interactives.apelections.org/election-results/data-live/2026-04-07/results/races/WI/20260407WI50888/metadata.json"
OUTPUT = Path("data/wi_2026_supreme_standardized.csv")

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


def fetch_json(url: str) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip",
        },
    )
    raw = urllib.request.urlopen(req, timeout=30).read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def main() -> None:
    detail = fetch_json(DETAIL_URL)
    metadata = fetch_json(METADATA_URL)

    # Candidate IDs from AP metadata.
    candidates = metadata["candidates"]
    name_to_id = {
        f"{v.get('first', '').strip()} {v.get('last', '').strip()}".strip(): k
        for k, v in candidates.items()
    }

    # Party-aligned assignment for modeling consistency with prior files.
    dem_name = "Chris Taylor"
    rep_name = "Maria Lazar"
    dem_id = name_to_id[dem_name]
    rep_id = name_to_id[rep_name]

    rows = []
    sum_dem = 0
    sum_rep = 0
    sum_total = 0

    for key, county in detail.items():
        if key == "summary":
            continue
        if county.get("reportingunitLevel") != 2:
            continue

        vote_by_candidate = {c["candidateID"]: int(c.get("voteCount", 0)) for c in county["candidates"]}
        total_votes = int(county["parameters"]["vote"]["total"])
        dem_votes = vote_by_candidate.get(dem_id, 0)
        rep_votes = vote_by_candidate.get(rep_id, 0)
        other_votes = max(0, total_votes - dem_votes - rep_votes)

        rows.append(
            {
                "county": county["reportingunitName"],
                "election_date": "2026-04-07",
                "race": "2026 Wisconsin Supreme Court",
                "dem_candidate": dem_name,
                "rep_candidate": rep_name,
                "dem_votes": dem_votes,
                "rep_votes": rep_votes,
                "other_votes": other_votes,
                "total_votes": total_votes,
                "source_csv": DETAIL_URL,
            }
        )

        sum_dem += dem_votes
        sum_rep += rep_votes
        sum_total += total_votes

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda x: x["county"]))

    # Verification checks against AP summary and metadata totals.
    summary = detail["summary"]
    summary_votes = {c["candidateID"]: int(c.get("voteCount", 0)) for c in summary["candidates"]}
    metadata_total = int(metadata["parameters"]["vote"]["total"])

    assert len(rows) == 72, f"Expected 72 counties, got {len(rows)}"
    assert sum_dem == summary_votes[dem_id], (sum_dem, summary_votes[dem_id])
    assert sum_rep == summary_votes[rep_id], (sum_rep, summary_votes[rep_id])
    assert sum_total == metadata_total, (sum_total, metadata_total)

    print("Wrote", OUTPUT)
    print("Verification passed:")
    print("  counties:", len(rows))
    print("  dem sum:", sum_dem)
    print("  rep sum:", sum_rep)
    print("  total sum:", sum_total)


if __name__ == "__main__":
    main()

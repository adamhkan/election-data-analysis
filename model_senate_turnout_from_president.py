#!/usr/bin/env python3
"""Estimate county Senate turnout as a 2-parameter function of Trump/Harris presidential votes.

Model assumptions (as requested):
- Ignore third-party/write-in presidential voters entirely.
- Every Senate voter is assumed to also have voted in the presidential election.
- No intercept term.
- Parameters are constrained to [0, 1].

For each county c:
    senate_turnout_c ~= a * harris_votes_c + b * trump_votes_c

where:
    a = fraction of Harris voters who also voted in Senate race
    b = fraction of Trump voters who also voted in Senate race
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


EXPECTED_WI_COUNTIES = 72


@dataclass(frozen=True)
class CountyRecord:
    county: str
    harris_votes: int
    trump_votes: int
    senate_turnout: int


def _normalize_county_name(county: str) -> str:
    return " ".join(county.strip().lower().split())


def _to_int(value: str) -> int:
    cleaned = value.replace(",", "").strip()
    if cleaned == "":
        raise ValueError("Encountered empty numeric field")
    return int(cleaned)


def load_presidential_votes(path: str) -> Dict[str, Tuple[str, int, int]]:
    """Return norm_county -> (display_county, harris_votes, trump_votes)."""
    results: Dict[str, Tuple[str, int, int]] = {}

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) < 3:
        raise ValueError(f"Presidential file looks too short: {path}")

    # Expected format has a two-row header, then county data:
    # col 0: County, col 1: Trump #, col 3: Harris #
    for row in rows[2:]:
        if not row or len(row) < 4:
            continue
        county = row[0].strip()
        if county == "":
            continue

        trump_votes = _to_int(row[1])
        harris_votes = _to_int(row[3])

        norm_county = _normalize_county_name(county)
        if norm_county in results:
            raise ValueError(f"Duplicate county in presidential file: {county}")
        results[norm_county] = (county, harris_votes, trump_votes)

    return results


def load_senate_turnout(path: str) -> Dict[str, Tuple[str, int]]:
    """Return norm_county -> (display_county, senate_turnout)."""
    results: Dict[str, Tuple[str, int]] = {}

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_cols = {"race", "county", "total_votes"}
        missing = required_cols - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Senate file missing required columns: {sorted(missing)}")

        for row in reader:
            if row["race"].strip() != "2024 Wisconsin U.S. Senate":
                continue
            county = row["county"].strip()
            turnout = _to_int(row["total_votes"])
            norm_county = _normalize_county_name(county)
            if norm_county in results:
                raise ValueError(f"Duplicate county in senate rows: {county}")
            results[norm_county] = (county, turnout)

    return results


def build_dataset(
    senate_turnout_by_county: Dict[str, Tuple[str, int]],
    presidential_votes_by_county: Dict[str, Tuple[str, int, int]],
) -> List[CountyRecord]:
    senate_counties = set(senate_turnout_by_county)
    presidential_counties = set(presidential_votes_by_county)

    if len(senate_counties) != EXPECTED_WI_COUNTIES:
        raise ValueError(
            f"Expected {EXPECTED_WI_COUNTIES} counties in senate file for 2024 Senate race, "
            f"found {len(senate_counties)}"
        )
    if len(presidential_counties) != EXPECTED_WI_COUNTIES:
        raise ValueError(
            f"Expected {EXPECTED_WI_COUNTIES} counties in presidential file, "
            f"found {len(presidential_counties)}"
        )

    if senate_counties != presidential_counties:
        only_senate = sorted(senate_counties - presidential_counties)
        only_pres = sorted(presidential_counties - senate_counties)
        raise ValueError(
            "County mismatch between files. "
            f"Only in senate={only_senate}; only in presidential={only_pres}"
        )

    records: List[CountyRecord] = []
    for county in sorted(senate_counties):
        senate_display_county, senate_turnout = senate_turnout_by_county[county]
        _, harris, trump = presidential_votes_by_county[county]
        records.append(
            CountyRecord(
                county=senate_display_county,
                harris_votes=harris,
                trump_votes=trump,
                senate_turnout=senate_turnout,
            )
        )
    return records


def _objective(records: Sequence[CountyRecord], a: float, b: float) -> float:
    n = len(records)
    if n == 0:
        raise ValueError("No county records available")

    total_sq = 0.0
    for r in records:
        pred = a * r.harris_votes + b * r.trump_votes
        err = pred - r.senate_turnout
        total_sq += err * err
    return total_sq / n


def fit_bounded_two_parameter_model(records: Sequence[CountyRecord]) -> Tuple[float, float, float]:
    """Fit min MSE model y ~= a*x_harris + b*x_trump with a,b in [0,1].

    Uses exact convex optimization logic for 2D box-constrained quadratic:
    - Check unconstrained OLS solution.
    - If not feasible, evaluate boundary optima on each edge and corners.
    """
    # Build sufficient statistics.
    s_hh = s_tt = s_ht = s_hy = s_ty = 0.0
    for r in records:
        h = float(r.harris_votes)
        t = float(r.trump_votes)
        y = float(r.senate_turnout)
        s_hh += h * h
        s_tt += t * t
        s_ht += h * t
        s_hy += h * y
        s_ty += t * y

    det = s_hh * s_tt - s_ht * s_ht
    candidates: List[Tuple[float, float]] = []

    # Interior (unconstrained) optimum when matrix is invertible.
    if abs(det) > 1e-12:
        a_star = (s_hy * s_tt - s_ty * s_ht) / det
        b_star = (s_ty * s_hh - s_hy * s_ht) / det
        if 0.0 <= a_star <= 1.0 and 0.0 <= b_star <= 1.0:
            candidates.append((a_star, b_star))

    # Boundary optima on lines a=0, a=1, b=0, b=1.
    if s_tt > 0:
        b_a0 = max(0.0, min(1.0, s_ty / s_tt))
        candidates.append((0.0, b_a0))
        b_a1 = max(0.0, min(1.0, (s_ty - s_ht) / s_tt))
        candidates.append((1.0, b_a1))

    if s_hh > 0:
        a_b0 = max(0.0, min(1.0, s_hy / s_hh))
        candidates.append((a_b0, 0.0))
        a_b1 = max(0.0, min(1.0, (s_hy - s_ht) / s_hh))
        candidates.append((a_b1, 1.0))

    # Corners.
    candidates.extend([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])

    best_a, best_b = 0.0, 0.0
    best_mse = float("inf")
    for a, b in candidates:
        mse = _objective(records, a, b)
        if mse < best_mse:
            best_mse = mse
            best_a, best_b = a, b

    return best_a, best_b, best_mse


def compute_metrics(records: Sequence[CountyRecord], a: float, b: float) -> Dict[str, float]:
    n = len(records)
    y_values = [float(r.senate_turnout) for r in records]
    y_mean = sum(y_values) / n

    sse = 0.0
    sae = 0.0
    for r in records:
        pred = a * r.harris_votes + b * r.trump_votes
        err = pred - r.senate_turnout
        sse += err * err
        sae += abs(err)

    mse = sse / n
    rmse = math.sqrt(mse)
    mae = sae / n

    sst = sum((y - y_mean) ** 2 for y in y_values)
    r2 = float("nan") if sst == 0 else 1.0 - (sse / sst)

    return {
        "num_counties": float(n),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "sse": sse,
        "r2": r2,
    }


def write_county_predictions(
    records: Sequence[CountyRecord],
    a: float,
    b: float,
    output_path: str,
) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "county",
                "harris_votes",
                "trump_votes",
                "actual_senate_turnout",
                "predicted_senate_turnout",
                "residual_pred_minus_actual",
                "abs_error",
            ]
        )
        for r in sorted(records, key=lambda x: x.county):
            pred = a * r.harris_votes + b * r.trump_votes
            resid = pred - r.senate_turnout
            writer.writerow(
                [
                    r.county,
                    r.harris_votes,
                    r.trump_votes,
                    r.senate_turnout,
                    f"{pred:.6f}",
                    f"{resid:.6f}",
                    f"{abs(resid):.6f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--senate-csv",
        default="data/wi_county_results_2024_2025.csv",
        help="Path to combined county results CSV (used ONLY for 2024 Senate rows).",
    )
    parser.add_argument(
        "--presidential-csv",
        default="data/Wisconsin Election Results - 2024 President.csv",
        help="Path to presidential county CSV (used for Harris/Trump votes).",
    )
    parser.add_argument(
        "--county-output",
        default="data/senate_turnout_model_county_predictions.csv",
        help="Where to write county-level actual vs predicted turnout table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        senate_turnout = load_senate_turnout(args.senate_csv)
        presidential_votes = load_presidential_votes(args.presidential_csv)
        records = build_dataset(senate_turnout, presidential_votes)
        a, b, _ = fit_bounded_two_parameter_model(records)
        metrics = compute_metrics(records, a, b)
        write_county_predictions(records, a, b, args.county_output)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Model fit complete.")
    print(f"Counties used: {int(metrics['num_counties'])}")
    print("\nLearned parameters (bounded to [0,1]):")
    print(f"  pct_harris_voters_in_senate = {a:.6f} ({a * 100:.3f}%)")
    print(f"  pct_trump_voters_in_senate  = {b:.6f} ({b * 100:.3f}%)")

    print("\nFit metrics (county-level turnout prediction):")
    print(f"  MSE  = {metrics['mse']:.6f}")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  MAE  = {metrics['mae']:.6f}")
    print(f"  SSE  = {metrics['sse']:.6f}")
    print(f"  R^2  = {metrics['r2']:.6f}")

    print(f"\nCounty-level predictions written to: {args.county_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

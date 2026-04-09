#!/usr/bin/env python3
"""Fit two-stage voter-retention + switching models between two standardized county CSVs.

Standardized input schema (required columns):
county,election_date,race,dem_candidate,rep_candidate,dem_votes,rep_votes,other_votes,total_votes,source_csv

Stage 1 (retention):
    predicted_target_major_votes ~= a * source_dem_votes + b * source_rep_votes

Stage 2 (switching among returners only):
    returned_dem = a * source_dem_votes
    returned_rep = b * source_rep_votes

    predicted_target_dem = h_loyal * returned_dem + (1 - t_loyal) * returned_rep
    predicted_target_rep = (1 - h_loyal) * returned_dem + t_loyal * returned_rep

All parameters are constrained to [0,1].

Two fitting modes are supported:
- sequential: fit stage-1 retention first (major-party vote-count MSE), then fit stage-2 loyalty
              using weighted MSE of percent errors for candidate votes.
- joint: fit all four parameters directly using weighted MSE of percent errors for candidate votes.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple

REQUIRED_COLUMNS = {
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
}


@dataclass(frozen=True)
class CountyRecord:
    county: str
    source_dem_votes: int
    source_rep_votes: int
    target_dem_votes: int
    target_rep_votes: int


@dataclass(frozen=True)
class FitResult:
    a_retention_dem: float
    b_retention_rep: float
    h_loyal_dem: float
    t_loyal_rep: float


@dataclass(frozen=True)
class CounterfactualResult:
    total_dem_votes: float
    total_rep_votes: float
    dem_share_percent: float
    rep_share_percent: float
    dem_margin_votes: float
    dem_margin_percent: float


def _to_int(value: str) -> int:
    return int(value.replace(",", "").strip())


def _normalize_county_name(county: str) -> str:
    return " ".join(county.strip().lower().split())


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_standardized_csv(path: str) -> Dict[str, Dict[str, str]]:
    """Load standardized rows keyed by normalized county name."""
    rows: Dict[str, Dict[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing required columns: {sorted(missing)}")

        for row in reader:
            county = row["county"].strip()
            if not county:
                continue
            norm = _normalize_county_name(county)
            if norm in rows:
                raise ValueError(f"Duplicate county '{county}' in {path}")
            rows[norm] = row

    return rows


def build_dataset(
    source_rows: Dict[str, Dict[str, str]],
    target_rows: Dict[str, Dict[str, str]],
    expected_counties: int | None,
) -> Tuple[List[CountyRecord], str, str, str, str]:
    source_counties = set(source_rows)
    target_counties = set(target_rows)

    if expected_counties is not None:
        if len(source_counties) != expected_counties:
            raise ValueError(f"Source file has {len(source_counties)} counties, expected {expected_counties}")
        if len(target_counties) != expected_counties:
            raise ValueError(f"Target file has {len(target_counties)} counties, expected {expected_counties}")

    if source_counties != target_counties:
        only_source = sorted(source_counties - target_counties)
        only_target = sorted(target_counties - source_counties)
        raise ValueError(
            "County mismatch between files. "
            f"only_source={only_source}; only_target={only_target}"
        )

    source_races = {row["race"].strip() for row in source_rows.values()}
    target_races = {row["race"].strip() for row in target_rows.values()}
    source_race = next(iter(source_races)) if source_races else ""
    target_race = next(iter(target_races)) if target_races else ""
    target_dem_candidate = next(iter({row["dem_candidate"].strip() for row in target_rows.values()}), "")
    target_rep_candidate = next(iter({row["rep_candidate"].strip() for row in target_rows.values()}), "")

    records: List[CountyRecord] = []
    for county in sorted(source_counties):
        s = source_rows[county]
        t = target_rows[county]

        records.append(
            CountyRecord(
                county=s["county"].strip(),
                source_dem_votes=_to_int(s["dem_votes"]),
                source_rep_votes=_to_int(s["rep_votes"]),
                target_dem_votes=_to_int(t["dem_votes"]),
                target_rep_votes=_to_int(t["rep_votes"]),
            )
        )

    return records, source_race, target_race, target_dem_candidate, target_rep_candidate


def stage1_turnout_mse(records: Sequence[CountyRecord], a: float, b: float) -> float:
    n = len(records)
    sse = 0.0
    for r in records:
        pred_major = a * r.source_dem_votes + b * r.source_rep_votes
        actual_major = r.target_dem_votes + r.target_rep_votes
        err = pred_major - actual_major
        sse += err * err
    return sse / n


def _predict_candidate_votes(r: CountyRecord, fit: FitResult) -> Tuple[float, float]:
    returned_dem = fit.a_retention_dem * r.source_dem_votes
    returned_rep = fit.b_retention_rep * r.source_rep_votes

    pred_dem = fit.h_loyal_dem * returned_dem + (1.0 - fit.t_loyal_rep) * returned_rep
    pred_rep = (1.0 - fit.h_loyal_dem) * returned_dem + fit.t_loyal_rep * returned_rep
    return pred_dem, pred_rep


def stage2_weighted_percent_mse(records: Sequence[CountyRecord], fit: FitResult) -> float:
    """Weighted MSE of percent errors for target-candidate votes.

    For county c with actual major-party total W_c = dem_c + rep_c,
    weight w_c = W_c / sum(W_c).

    Candidate percent error is computed on a 0-100 scale:
        pe_dem = 100 * (pred_dem - dem) / dem
        pe_rep = 100 * (pred_rep - rep) / rep

    County contribution:
        w_c * (pe_dem^2 + pe_rep^2) / 2
    """
    total_real_votes = sum(r.target_dem_votes + r.target_rep_votes for r in records)
    if total_real_votes <= 0:
        raise ValueError("Target major-party vote totals are non-positive")

    loss = 0.0
    for r in records:
        if r.target_dem_votes <= 0 or r.target_rep_votes <= 0:
            raise ValueError(
                f"County '{r.county}' has non-positive candidate votes; percent-error loss undefined"
            )

        pred_dem, pred_rep = _predict_candidate_votes(r, fit)

        pe_dem = 100.0 * (pred_dem - r.target_dem_votes) / r.target_dem_votes
        pe_rep = 100.0 * (pred_rep - r.target_rep_votes) / r.target_rep_votes

        county_weight = (r.target_dem_votes + r.target_rep_votes) / total_real_votes
        loss += county_weight * ((pe_dem * pe_dem + pe_rep * pe_rep) / 2.0)

    return loss


def fit_stage1_retention(records: Sequence[CountyRecord]) -> Tuple[float, float]:
    """Exact box-constrained solve for stage-1 2-parameter turnout MSE."""
    s_dd = s_rr = s_dr = s_dy = s_ry = 0.0
    for r in records:
        d = float(r.source_dem_votes)
        rep = float(r.source_rep_votes)
        y = float(r.target_dem_votes + r.target_rep_votes)
        s_dd += d * d
        s_rr += rep * rep
        s_dr += d * rep
        s_dy += d * y
        s_ry += rep * y

    det = s_dd * s_rr - s_dr * s_dr
    candidates: List[Tuple[float, float]] = []

    if abs(det) > 1e-12:
        a_star = (s_dy * s_rr - s_ry * s_dr) / det
        b_star = (s_ry * s_dd - s_dy * s_dr) / det
        if 0.0 <= a_star <= 1.0 and 0.0 <= b_star <= 1.0:
            candidates.append((a_star, b_star))

    if s_rr > 0:
        b_a0 = _clip01(s_ry / s_rr)
        b_a1 = _clip01((s_ry - s_dr) / s_rr)
        candidates.extend([(0.0, b_a0), (1.0, b_a1)])

    if s_dd > 0:
        a_b0 = _clip01(s_dy / s_dd)
        a_b1 = _clip01((s_dy - s_dr) / s_dd)
        candidates.extend([(a_b0, 0.0), (a_b1, 1.0)])

    candidates.extend([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])

    best_a, best_b, best_mse = 0.0, 0.0, float("inf")
    for a, b in candidates:
        mse = stage1_turnout_mse(records, a, b)
        if mse < best_mse:
            best_a, best_b, best_mse = a, b, mse

    return best_a, best_b


def _projected_gradient_descent(
    loss_fn: Callable[[List[float]], float],
    init: List[float],
    *,
    steps: int = 5000,
    lr: float = 0.01,
    tol: float = 1e-10,
) -> List[float]:
    """Simple projected gradient descent with finite-difference gradients on [0,1]^k."""
    x = [_clip01(v) for v in init]
    eps = 1e-6
    prev = loss_fn(x)

    for _ in range(steps):
        grad = []
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] = _clip01(x_plus[i] + eps)
            x_minus[i] = _clip01(x_minus[i] - eps)
            f_plus = loss_fn(x_plus)
            f_minus = loss_fn(x_minus)
            denom = x_plus[i] - x_minus[i]
            g = 0.0 if abs(denom) < 1e-15 else (f_plus - f_minus) / denom
            grad.append(g)

        candidate = [_clip01(v - lr * g) for v, g in zip(x, grad)]
        cur = loss_fn(candidate)

        # Backtracking when a step increases loss.
        local_lr = lr
        while cur > prev and local_lr > 1e-8:
            local_lr *= 0.5
            candidate = [_clip01(v - local_lr * g) for v, g in zip(x, grad)]
            cur = loss_fn(candidate)

        if abs(prev - cur) < tol:
            x = candidate
            break

        x = candidate
        prev = cur

    return x


def fit_sequential(records: Sequence[CountyRecord]) -> FitResult:
    a, b = fit_stage1_retention(records)

    def loss_stage2(params: List[float]) -> float:
        h_loyal, t_loyal = params
        fit = FitResult(a_retention_dem=a, b_retention_rep=b, h_loyal_dem=h_loyal, t_loyal_rep=t_loyal)
        return stage2_weighted_percent_mse(records, fit)

    h_opt, t_opt = _projected_gradient_descent(loss_stage2, init=[0.9, 0.9], lr=0.02)
    return FitResult(a_retention_dem=a, b_retention_rep=b, h_loyal_dem=h_opt, t_loyal_rep=t_opt)


def fit_joint(records: Sequence[CountyRecord]) -> FitResult:
    a0, b0 = fit_stage1_retention(records)

    def loss_joint(params: List[float]) -> float:
        a, b, h_loyal, t_loyal = params
        fit = FitResult(a_retention_dem=a, b_retention_rep=b, h_loyal_dem=h_loyal, t_loyal_rep=t_loyal)
        return stage2_weighted_percent_mse(records, fit)

    a_opt, b_opt, h_opt, t_opt = _projected_gradient_descent(loss_joint, init=[a0, b0, 0.9, 0.9], lr=0.01)
    return FitResult(a_retention_dem=a_opt, b_retention_rep=b_opt, h_loyal_dem=h_opt, t_loyal_rep=t_opt)


def compute_stage2_diagnostics(records: Sequence[CountyRecord], fit: FitResult) -> Dict[str, float]:
    total_real_votes = sum(r.target_dem_votes + r.target_rep_votes for r in records)

    mse_votes = 0.0
    mae_votes = 0.0
    weighted_pct_mse = 0.0

    for r in records:
        pred_dem, pred_rep = _predict_candidate_votes(r, fit)
        err_dem = pred_dem - r.target_dem_votes
        err_rep = pred_rep - r.target_rep_votes

        mse_votes += (err_dem * err_dem + err_rep * err_rep) / 2.0
        mae_votes += (abs(err_dem) + abs(err_rep)) / 2.0

        pe_dem = 100.0 * err_dem / r.target_dem_votes
        pe_rep = 100.0 * err_rep / r.target_rep_votes
        w = (r.target_dem_votes + r.target_rep_votes) / total_real_votes
        weighted_pct_mse += w * ((pe_dem * pe_dem + pe_rep * pe_rep) / 2.0)

    n = len(records)
    mse_votes /= n
    mae_votes /= n

    return {
        "stage1_mse_major_votes": stage1_turnout_mse(records, fit.a_retention_dem, fit.b_retention_rep),
        "stage2_weighted_percent_mse": weighted_pct_mse,
        "stage2_weighted_percent_rmse": math.sqrt(weighted_pct_mse),
        "stage2_unweighted_vote_mse": mse_votes,
        "stage2_unweighted_vote_rmse": math.sqrt(mse_votes),
        "stage2_unweighted_vote_mae": mae_votes,
    }


def compute_full_return_counterfactual(records: Sequence[CountyRecord], fit: FitResult) -> CounterfactualResult:
    """Counterfactual where all source two-party voters return with learned switching behavior."""
    total_dem = 0.0
    total_rep = 0.0
    for r in records:
        # Set retention to 100% for both source-party groups, while keeping learned loyalty/switching.
        dem = fit.h_loyal_dem * r.source_dem_votes + (1.0 - fit.t_loyal_rep) * r.source_rep_votes
        rep = (1.0 - fit.h_loyal_dem) * r.source_dem_votes + fit.t_loyal_rep * r.source_rep_votes
        total_dem += dem
        total_rep += rep

    total = total_dem + total_rep
    if total <= 0:
        raise ValueError("Counterfactual total votes are non-positive")

    dem_share = 100.0 * total_dem / total
    rep_share = 100.0 * total_rep / total
    dem_margin_votes = total_dem - total_rep
    dem_margin_percent = dem_share - rep_share
    return CounterfactualResult(
        total_dem_votes=total_dem,
        total_rep_votes=total_rep,
        dem_share_percent=dem_share,
        rep_share_percent=rep_share,
        dem_margin_votes=dem_margin_votes,
        dem_margin_percent=dem_margin_percent,
    )


def write_county_predictions(records: Sequence[CountyRecord], fit: FitResult, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "county",
                "source_dem_votes",
                "source_rep_votes",
                "actual_target_dem_votes",
                "actual_target_rep_votes",
                "predicted_target_dem_votes",
                "predicted_target_rep_votes",
                "counterfactual_full_return_dem_votes",
                "counterfactual_full_return_rep_votes",
                "dem_percent_error_0_to_100_scale",
                "rep_percent_error_0_to_100_scale",
            ]
        )
        for r in sorted(records, key=lambda x: x.county):
            pred_dem, pred_rep = _predict_candidate_votes(r, fit)
            cf_dem = fit.h_loyal_dem * r.source_dem_votes + (1.0 - fit.t_loyal_rep) * r.source_rep_votes
            cf_rep = (1.0 - fit.h_loyal_dem) * r.source_dem_votes + fit.t_loyal_rep * r.source_rep_votes
            pe_dem = 100.0 * (pred_dem - r.target_dem_votes) / r.target_dem_votes
            pe_rep = 100.0 * (pred_rep - r.target_rep_votes) / r.target_rep_votes
            w.writerow(
                [
                    r.county,
                    r.source_dem_votes,
                    r.source_rep_votes,
                    r.target_dem_votes,
                    r.target_rep_votes,
                    f"{pred_dem:.6f}",
                    f"{pred_rep:.6f}",
                    f"{cf_dem:.6f}",
                    f"{cf_rep:.6f}",
                    f"{pe_dem:.6f}",
                    f"{pe_rep:.6f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-csv", required=True, help="Standardized source-election CSV (predictor race).")
    parser.add_argument("--target-csv", required=True, help="Standardized target-election CSV (second race).")
    parser.add_argument(
        "--expected-counties",
        type=int,
        default=None,
        help="Optional strict county count check (e.g., 72 for Wisconsin).",
    )
    parser.add_argument(
        "--fit-mode",
        choices=["sequential", "joint"],
        default="sequential",
        help="Fit sequentially (stage1 then stage2) or jointly (all 4 params at once on stage2 loss).",
    )
    parser.add_argument(
        "--county-output",
        default="data/generalized_switching_model_county_predictions.csv",
        help="Where to write county-level prediction details.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source_rows = load_standardized_csv(args.source_csv)
        target_rows = load_standardized_csv(args.target_csv)
        records, source_race, target_race, target_dem_candidate, target_rep_candidate = build_dataset(
            source_rows, target_rows, expected_counties=args.expected_counties
        )

        if args.fit_mode == "sequential":
            fit = fit_sequential(records)
        else:
            fit = fit_joint(records)

        diagnostics = compute_stage2_diagnostics(records, fit)
        counterfactual = compute_full_return_counterfactual(records, fit)
        write_county_predictions(records, fit, args.county_output)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Model fit complete.")
    print(f"Fit mode: {args.fit_mode}")
    print(f"Source race: {source_race}")
    print(f"Target race: {target_race}")
    print(f"Target candidates: Dem='{target_dem_candidate}', Rep='{target_rep_candidate}'")
    print(f"Counties used: {len(records)}")

    print("\nLearned retention parameters (stage 1):")
    print(f"  a_retention_dem = {fit.a_retention_dem:.6f} ({fit.a_retention_dem * 100:.3f}%)")
    print(f"  b_retention_rep = {fit.b_retention_rep:.6f} ({fit.b_retention_rep * 100:.3f}%)")

    print("\nLearned loyalty/switching parameters (stage 2):")
    print(f"  h_loyal_dem = {fit.h_loyal_dem:.6f} ({fit.h_loyal_dem * 100:.3f}%)")
    print(f"  t_loyal_rep = {fit.t_loyal_rep:.6f} ({fit.t_loyal_rep * 100:.3f}%)")
    print(f"  h_switch_to_rep = {1.0 - fit.h_loyal_dem:.6f} ({(1.0 - fit.h_loyal_dem) * 100:.3f}%)")
    print(f"  t_switch_to_dem = {1.0 - fit.t_loyal_rep:.6f} ({(1.0 - fit.t_loyal_rep) * 100:.3f}%)")

    print("\nDiagnostics:")
    print(f"  stage1_mse_major_votes            = {diagnostics['stage1_mse_major_votes']:.6f}")
    print(f"  stage2_weighted_percent_mse       = {diagnostics['stage2_weighted_percent_mse']:.6f}")
    print(f"  stage2_weighted_percent_rmse      = {diagnostics['stage2_weighted_percent_rmse']:.6f}")
    print(f"  stage2_unweighted_vote_mse        = {diagnostics['stage2_unweighted_vote_mse']:.6f}")
    print(f"  stage2_unweighted_vote_rmse       = {diagnostics['stage2_unweighted_vote_rmse']:.6f}")
    print(f"  stage2_unweighted_vote_mae        = {diagnostics['stage2_unweighted_vote_mae']:.6f}")

    print("\nCounterfactual result if 100% of source two-party voters returned:")
    print(f"  total_dem_votes  = {counterfactual.total_dem_votes:.3f}")
    print(f"  total_rep_votes  = {counterfactual.total_rep_votes:.3f}")
    print(f"  dem_share_pct    = {counterfactual.dem_share_percent:.6f}")
    print(f"  rep_share_pct    = {counterfactual.rep_share_percent:.6f}")
    print(f"  dem_margin_votes = {counterfactual.dem_margin_votes:.3f}")
    print(f"  dem_margin_pct   = {counterfactual.dem_margin_percent:.6f}")

    print(f"\nCounty-level predictions written to: {args.county_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

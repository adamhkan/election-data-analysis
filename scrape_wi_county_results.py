#!/usr/bin/env python3
"""Scrape/collect Wisconsin county-level election results for selected races.

Sources:
- 2024 presidential county-by-county PDF (Wisconsin Elections Commission)
- 2024 U.S. Senate county-by-county PDF (Wisconsin Elections Commission)
- 2025 Wisconsin Supreme Court county table from Wikipedia wikitext
  (which cites WEC certified canvass summary)
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List
from urllib.request import Request, urlopen

WI_COUNTIES_UPPER = [
    "ADAMS", "ASHLAND", "BARRON", "BAYFIELD", "BROWN", "BUFFALO", "BURNETT", "CALUMET",
    "CHIPPEWA", "CLARK", "COLUMBIA", "CRAWFORD", "DANE", "DODGE", "DOOR", "DOUGLAS", "DUNN",
    "EAU CLAIRE", "FLORENCE", "FOND DU LAC", "FOREST", "GRANT", "GREEN", "GREEN LAKE", "IOWA",
    "IRON", "JACKSON", "JEFFERSON", "JUNEAU", "KENOSHA", "KEWAUNEE", "LA CROSSE", "LAFAYETTE",
    "LANGLADE", "LINCOLN", "MANITOWOC", "MARATHON", "MARINETTE", "MARQUETTE", "MENOMINEE",
    "MILWAUKEE", "MONROE", "OCONTO", "ONEIDA", "OUTAGAMIE", "OZAUKEE", "PEPIN", "PIERCE",
    "POLK", "PORTAGE", "PRICE", "RACINE", "RICHLAND", "ROCK", "RUSK", "SAINT CROIX", "ST. CROIX", "SAUK",
    "SAWYER", "SHAWANO", "SHEBOYGAN", "TAYLOR", "TREMPEALEAU", "VERNON", "VILAS", "WALWORTH",
    "WASHBURN", "WASHINGTON", "WAUKESHA", "WAUPACA", "WAUSHARA", "WINNEBAGO", "WOOD",
]
COUNTY_SET = set(WI_COUNTIES_UPPER)

SOURCES = {
    "2024_presidential": "https://elections.wi.gov/sites/default/files/documents/County%20by%20County%20Report_POTUS.pdf",
    "2024_senate": "https://elections.wi.gov/sites/default/files/documents/County%20by%20County%20Report_US%20Senate_1.pdf",
    "2025_supreme_wiki": "https://en.wikipedia.org/w/index.php?title=2025_Wisconsin_Supreme_Court_election&action=raw",
}


def _download_with_curl(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-fsSL", "-o", str(out_path), url], check=True)


def _extract_pdf_text(pdf_path: Path) -> str:
    from pypdf import PdfReader  # lazy import

    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_county_rows_from_pdf_text(text: str) -> Dict[str, Dict[str, int]]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    results: Dict[str, Dict[str, int]] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        if line in COUNTY_SET:
            # In these WEC county-by-county reports, first numeric columns are:
            # total votes, dem votes, rep votes.
            if i + 3 >= len(lines):
                raise ValueError(f"Unexpected end of lines around county {line}")
            total = _to_int(lines[i + 1])
            dem = _to_int(lines[i + 2])
            rep = _to_int(lines[i + 3])
            county_name = _title_county(line)
            if county_name not in results:
                results[county_name] = {"total_votes": total, "dem_votes": dem, "rep_votes": rep}
            i += 4
            continue
        i += 1

    if len(results) != 72:
        raise ValueError(f"Expected 72 counties, parsed {len(results)} from PDF text")
    return results


def _to_int(s: str) -> int:
    cleaned = re.sub(r"[^0-9-]", "", s)
    if cleaned == "":
        return 0
    return int(cleaned)


def _title_county(county_upper: str) -> str:
    if county_upper in {"SAINT CROIX", "ST. CROIX"}:
        return "St. Croix"
    return county_upper.title()


def _fetch_text(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp:
        return resp.read().decode("utf-8")


def _parse_2025_supreme_from_wikitext(text: str) -> Dict[str, Dict[str, int]]:
    marker = "===By county==="
    start = text.find(marker)
    if start == -1:
        raise ValueError("Could not locate 'By county' section in wikitext")
    table_start = text.find("{|", start)
    table_end = text.find("|}", table_start)
    if table_start == -1 or table_end == -1:
        raise ValueError("Could not locate county table in wikitext")
    table = text[table_start:table_end]

    rows = table.split("|- style=\"text-align:center;\"")
    out: Dict[str, Dict[str, int]] = {}

    for chunk in rows[1:]:
        cells = []
        for raw_line in chunk.splitlines():
            line = raw_line.strip()
            if not line.startswith("|"):
                continue
            if line.startswith("|-"):
                continue
            value = line.rsplit("|", 1)[-1].strip()
            value = re.sub(r"\[\[[^\]|]+\|([^\]]+)\]\]", r"\1", value)
            value = re.sub(r"\[\[([^\]]+)\]\]", r"\1", value)
            value = re.sub(r"\{\{[^}]+\}\}", "", value).strip()
            cells.append(value)

        if len(cells) < 10:
            continue

        county = cells[0].replace("County, Wisconsin", "").strip()
        if county == "Saint Croix":
            county = "St. Croix"

        out[county] = {
            "dem_votes": _to_int(cells[1]),
            "rep_votes": _to_int(cells[3]),
            "total_votes": _to_int(cells[9]),
        }

    if len(out) != 72:
        raise ValueError(f"Expected 72 counties, parsed {len(out)} from 2025 county table")
    return out


def _records_for_race(
    race_key: str,
    county_data: Dict[str, Dict[str, int]],
    source_url: str,
    dem_candidate: str,
    rep_candidate: str,
) -> Iterable[Dict[str, str | int]]:
    meta = {
        "2024_presidential": ("2024-11-05", "2024 Wisconsin Presidential"),
        "2024_senate": ("2024-11-05", "2024 Wisconsin U.S. Senate"),
        "2025_supreme": ("2025-04-01", "2025 Wisconsin Supreme Court"),
    }
    election_date, race_name = meta[race_key]

    for county in sorted(county_data):
        row = county_data[county]
        yield {
            "election_date": election_date,
            "race": race_name,
            "county": county,
            "dem_candidate": dem_candidate,
            "rep_candidate": rep_candidate,
            "dem_votes": row["dem_votes"],
            "rep_votes": row["rep_votes"],
            "total_votes": row["total_votes"],
            "source_url": source_url,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/wi_county_results_2024_2025.csv")
    parser.add_argument("--cache-dir", default="data/raw")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)

    pres_pdf = cache_dir / "wi_2024_presidential.pdf"
    senate_pdf = cache_dir / "wi_2024_senate.pdf"

    _download_with_curl(SOURCES["2024_presidential"], pres_pdf)
    _download_with_curl(SOURCES["2024_senate"], senate_pdf)

    pres = _parse_county_rows_from_pdf_text(_extract_pdf_text(pres_pdf))
    senate = _parse_county_rows_from_pdf_text(_extract_pdf_text(senate_pdf))

    wiki_text = _fetch_text(SOURCES["2025_supreme_wiki"])
    supreme = _parse_2025_supreme_from_wikitext(wiki_text)

    records: List[Dict[str, str | int]] = []
    records.extend(_records_for_race(
        "2024_presidential", pres, SOURCES["2024_presidential"], "Kamala Harris", "Donald Trump"
    ))
    records.extend(_records_for_race(
        "2024_senate", senate, SOURCES["2024_senate"], "Tammy Baldwin", "Eric Hovde"
    ))
    records.extend(_records_for_race(
        "2025_supreme", supreme, SOURCES["2025_supreme_wiki"], "Susan Crawford", "Brad Schimel"
    ))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "election_date",
                "race",
                "county",
                "dem_candidate",
                "rep_candidate",
                "dem_votes",
                "rep_votes",
                "total_votes",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Wrote {len(records)} rows to {output_path}")


if __name__ == "__main__":
    main()

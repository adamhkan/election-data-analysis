## Wisconsin county-level election scraper (2024/2025)

This repository includes a script that collects county-level vote totals for:

- 2024 Wisconsin Presidential election (Dem vs Rep)
- 2024 Wisconsin U.S. Senate election (Dem vs Rep)
- 2025 Wisconsin Supreme Court election (Crawford vs Schimel)

### Run

```bash
python -m pip install pypdf
python scrape_wi_county_results.py
```

Output CSV:

- `data/wi_county_results_2024_2025.csv`

## Standardized county CSV format + generalized model

Use this script to create standardized county-level files for modeling:

```bash
python standardize_wi_election_csvs.py
```

This writes:

- `data/county_results_standardized_template.csv` (schema template)
- `data/wi_2024_president_standardized.csv`
- `data/wi_2024_senate_standardized.csv`
- `data/wi_2025_supreme_standardized.csv`

Then run the generalized retention model:

```bash
python model_turnout_from_two_standardized_csvs.py \
  --source-csv data/wi_2024_president_standardized.csv \
  --target-csv data/wi_2024_senate_standardized.csv \
  --expected-counties 72 \
  --county-output data/generalized_model_2024_senate_predictions.csv
```

To include stage-2 party-switching estimation (sequential or joint fit), use:

```bash
python model_turnout_from_two_standardized_csvs.py \
  --source-csv data/wi_2024_president_standardized.csv \
  --target-csv data/wi_2025_supreme_standardized.csv \
  --expected-counties 72 \
  --fit-mode sequential \
  --county-output data/generalized_switching_model_2025_supreme_sequential.csv
```

The model output now also prints a counterfactual statewide result where 100% of source two-party voters return in the second race while following the learned switching tendencies.

To fetch and standardize 2026 Wisconsin Supreme Court county results from AP's election data feed, run:

```bash
python fetch_wi_2026_supreme_by_county.py
```

### Data sources used by the script

- WEC county-by-county PDF (2024 President):
  `https://elections.wi.gov/sites/default/files/documents/County%20by%20County%20Report_POTUS.pdf`
- WEC county-by-county PDF (2024 U.S. Senate):
  `https://elections.wi.gov/sites/default/files/documents/County%20by%20County%20Report_US%20Senate_1.pdf`
- Wikipedia raw table for 2025 Wisconsin Supreme Court county results (which cites WEC canvass certification):
  `https://en.wikipedia.org/w/index.php?title=2025_Wisconsin_Supreme_Court_election&action=raw`

> Note: If Wisconsin Elections Commission publishes a direct county-by-county PDF/CSV specifically for the 2025 Supreme Court race, you can swap that URL into the script for an all-official-source pipeline.

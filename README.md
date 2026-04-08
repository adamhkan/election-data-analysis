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

### Data sources used by the script

- WEC county-by-county PDF (2024 President):
  `https://elections.wi.gov/sites/default/files/documents/County%20by%20County%20Report_POTUS.pdf`
- WEC county-by-county PDF (2024 U.S. Senate):
  `https://elections.wi.gov/sites/default/files/documents/County%20by%20County%20Report_US%20Senate_1.pdf`
- Wikipedia raw table for 2025 Wisconsin Supreme Court county results (which cites WEC canvass certification):
  `https://en.wikipedia.org/w/index.php?title=2025_Wisconsin_Supreme_Court_election&action=raw`

> Note: If Wisconsin Elections Commission publishes a direct county-by-county PDF/CSV specifically for the 2025 Supreme Court race, you can swap that URL into the script for an all-official-source pipeline.

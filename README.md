# Maricopa Housing Stats

An interactive Streamlit dashboard exploring residential property sales in
Maricopa County, Arizona using public Assessor data.

> Companion project to [Maricopa Housing Predictor](../maricopa-housing-predictor),
> which uses the same data to build an ML price model.

## What you can do in the app

- **Sale-price trends by year** — see how the market moved 2018-2026
- **Geography** — compare BOOKs (Assessor's geographic clusters)
- **Property class** — distributions and medians for R1 through R7
- **Sqft vs. price scatter** — interactive, sampled, with filters
- **Quick stats cards** — median, mean, count, year-over-year change

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS / Linux

pip install -r requirements.txt
```

### Get the data

The app reads `data/processed/residential_with_sales.parquet`. To generate it,
follow the same data-ingestion path used in the predictor project:

1. Download the Maricopa Assessor **Residential Master (R116)** export from
   `mcassessor.maricopa.gov` → Public Data / Downloads.
2. Unzip into `data/raw/Residential_Master/`.
3. Run the ingestion script:

```bash
python notebooks/01_get_data.py
```

That writes the filtered Parquet file the app expects.

### Run the dashboard

```bash
streamlit run stats_app.py
```

Browser tab opens at `http://localhost:8501`.

## Folder layout

```
maricopa-housing-stats/
├── stats_app.py             # Streamlit dashboard (entry point)
├── notebooks/
│   ├── 01_get_data.py       # raw assessor file → Parquet
│   └── 02_eda.py            # offline EDA → plots in data/processed/plots/
├── data/
│   ├── raw/                 # drop the Assessor download here (gitignored)
│   └── processed/           # Parquet + plots (gitignored)
├── src/                     # reusable helpers (eventually)
├── requirements.txt
├── README.md
└── LICENSE
```

## Notes on the data

- The Assessor's delivered file is pipe-delimited, no header, 24 columns.
- `SALE_DATE` ships ISO `YYYY-MM-DD` despite the legend claiming `MM/DD/YYYY` —
  trust the data.
- Sale prices below $50,000 (quit-claim deeds, intra-family transfers) and
  above $5,000,000 (luxury outliers) are filtered out.
- `RoofStyle` is 100% empty; we drop it.

## License

Code is MIT-licensed ([LICENSE](LICENSE)).
Data belongs to the Maricopa County Assessor's Office and is not redistributed
here.

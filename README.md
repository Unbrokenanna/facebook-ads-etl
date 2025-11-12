# Facebook Ads ETL

Reads raw JSON files from `./raw`, normalizes them to a canonical schema, cleans & enriches the data (dedupe, types, features), writes a CSV to `./result`, and serves ranked tables via a small Flask app.


- **Docker Hub image:** `annayarik/facebook-ads-etl:v1`  
- **Container port:** `8800`  
- **Open UI:** http://localhost:8800

## Features
- Parse & normalize JSON (`pandas.json_normalize`)
- Canonical schema with clear names
- Cleaning:
  - Missing values (text → `"Unknown"`, numeric → `0`)
  - Duplicate removal by (ad_archive_id, start_date, page_name)
  - Safe dtypes & datetime parsing
- Enrichment:
  - `duration_hours` from seconds
  - `media_mix` (`video-only`, `image-only`, `both`, `none`)
  - Language detection (`langdetect`)
- Ranked views (HTML):
  - Top 10 by `ad_archive_id` + `page_name`
  - Top 10 by `page_name`
  - Aggregation by `language`
  - Top 10 IMAGE ads
- Export: `./result/clean_dataset_facebook_ads.csv`

---

## Run with Docker

```bash
docker pull annayarik/facebook-ads-etl:v1

docker run --rm \
  -p 8800:8800 \
  -v "$PWD/raw:/app/raw" \
  -v "$PWD/result:/app/result" \
  annayarik/facebook-ads-etl:v1

"""Facebook Ads ETL + Flask Viewer

Reads raw JSON files from ./raw, normalizes them to a canonical schema,
cleans and enriches the data (dedupe, types, features), writes a CSV,
and serves 4 ranking tables as an HTML page.
"""

from flask import Flask, render_template_string
import pandas as pd
import glob
import json
from pathlib import Path
from langdetect import detect, DetectorFactory

app = Flask(__name__)

# Set langdetect seed once for deterministic language detection across runs
DetectorFactory.seed = 0

TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Facebook Ads – Tables</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }
    h2 { margin: 0 0 8px; }
    .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .table-wrap { max-height: 70vh; overflow: auto; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; font-size: 12px; }
    th { background: #f9fafb; position: sticky; top: 0; }
  </style>
</head>
<body>
  <div class="grid">
    <div class="card"><h2>Top 10 by ad_archive_id</h2><div class="table-wrap">{{ table1|safe }}</div></div>
    <div class="card"><h2>Top 10 by page_name</h2><div class="table-wrap">{{ table2|safe }}</div></div>
    <div class="card"><h2>By language</h2><div class="table-wrap">{{ table3|safe }}</div></div>
    <div class="card"><h2>Top 10 IMAGE ads</h2><div class="table-wrap">{{ table4|safe }}</div></div>
  </div>
</body>
</html>
"""

# Raw → selected → renamed columns
select_columns = [
    'ad_archive_id',
    'ads_count',
    'snapshot.page_name',
    'snapshot.body.text',
    'snapshot.display_format',
    'snapshot.page_like_count',
    'total_active_time',
    'start_date_formatted',
    'end_date_formatted',
]

new_column_names = [
    'ad_archive_id',
    'ads_count',
    'page_name',
    'body_text',
    'display_format',
    'page_like_count',
    'total_active_time',
    'start_date',
    'end_date',
]


def extract_from_json() -> pd.DataFrame:
    """Load and normalize all JSON files from ./raw into a single DataFrame.

    - Uses pandas.json_normalize to handle nested fields.
    - Ensures a consistent column set with reindex (missing columns become NaN).
    - Renames columns to the canonical schema.

    Returns:
        pd.DataFrame: Concatenated raw data mapped to `new_column_names`.
    """
    files = list(glob.glob('raw/*.json'))
    if not files:
        print('No JSON files found in ./raw')
        return pd.DataFrame(columns=new_column_names)

    frames = []
    for file in files:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.json_normalize(data)
        # Keep only the known fields; add any missing as NaN
        df = df.reindex(columns=select_columns)
        frames.append(df)

    raw_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=select_columns)
    raw_df.columns = new_column_names  # rename to canonical names
    return raw_df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate ads based on a stable business key.

    Args:
        df: Input DataFrame in canonical schema.

    Returns:
        pd.DataFrame: Frame with duplicates dropped by (ad_archive_id, start_date, page_name).
    """
    return df.drop_duplicates(subset=['ad_archive_id', 'start_date', 'page_name'], keep='first')


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing text/numeric values with sensible defaults.

    - Textual: 'Unknown'
    - Numeric: 0

    Args:
        df: Input DataFrame in canonical schema.

    Returns:
        pd.DataFrame: Frame with NA values imputed.
    """
    text_cols = ['page_name', 'body_text', 'display_format']
    numeric_cols = ['ads_count', 'page_like_count', 'total_active_time']
    df[text_cols] = df[text_cols].fillna('Unknown')
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def apply_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce types for canonical schema and safely parse datetimes.

    - start_date, end_date → datetime64[ns] (coerce errors)
    - IDs and counts → numeric/nullable integer
    - Strings → pandas 'string' dtype

    Args:
        df: Input DataFrame after missing value handling.

    Returns:
        pd.DataFrame: Typed DataFrame.
    """
    # Datetime parsing is lossy-safe: invalid → NaT
    df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
    df['end_date'] = pd.to_datetime(df['end_date'], errors='coerce')

    # Numeric conversions with coercion, then cast to desired dtypes
    df['ad_archive_id'] = pd.to_numeric(df['ad_archive_id'], errors='coerce').astype('Int64')
    df['ads_count'] = pd.to_numeric(df['ads_count'], errors='coerce').fillna(0).astype('Int32')
    df['page_like_count'] = pd.to_numeric(df['page_like_count'], errors='coerce').fillna(0).astype('Int64')
    df['total_active_time'] = pd.to_numeric(df['total_active_time'], errors='coerce').fillna(0.0).astype('float64')

    # Canonical string dtype (nullable)
    for c in ['page_name', 'body_text', 'display_format']:
        df[c] = df[c].astype('string')

    return df


def language_detection(text: str) -> str:
    """Detect ISO 639-1 language code for ad text (best effort).

    - Returns 'unknown' for empty/very short/undetectable strings.

    Args:
        text: Input text from `body_text`.

    Returns:
        str: Two-letter code like 'en', 'fr', or 'unknown'.
    """
    try:
        t = (text or "").strip()
        if len(t) < 3:
            return 'unknown'
        return detect(t)
    except Exception:
        return 'unknown'


def media_mix_detection(fmt: str) -> str:
    """Map display format to a simplified media-mix tag.

    Args:
        fmt: Original display format (e.g., 'VIDEO', 'IMAGE', 'CAROUSEL').

    Returns:
        str: One of {'video-only','image-only','both','none'}.
    """
    fmt = (fmt or '').upper()
    if fmt == 'VIDEO':
        return 'video-only'
    if fmt in ('IMAGE', 'MULTI_IMAGES'):
        return 'image-only'
    if fmt == 'CAROUSEL':
        return 'both'
    return 'none'


def duration_hours(seconds) -> float:
    """Convert seconds to hours (1 decimal).

    Args:
        seconds: Duration in seconds.

    Returns:
        float: Hours rounded to 1 decimal (invalid input → 0.0).
    """
    try:
        return float(round(float(seconds) / 3600.0, 1))
    except Exception:
        return 0.0


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning + enrichment pipeline.

    Steps:
        1) Impute missing values
        2) Drop duplicates by business key
        3) Enforce schema types
        4) Compute features:
           - duration_hours (from total_active_time)
           - media_mix (from display_format)
           - language (from body_text)

    Args:
        df: Raw extracted DataFrame.

    Returns:
        pd.DataFrame: Cleaned, typed, and feature-enriched DataFrame.
    """
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = apply_schema(df)
    df['duration_hours'] = df['total_active_time'].apply(duration_hours)
    df['media_mix'] = df['display_format'].apply(media_mix_detection)
    df['language'] = df['body_text'].apply(language_detection)
    return df


def rank_performance_by_columns(
    df: pd.DataFrame,
    key_cols: list[str],
    filter_expr: str | None = None
) -> pd.DataFrame:
    """Rank top 10 entities by multiple measures with optional filtering.

    Ranking logic:
      - Filter with a pandas .query expression (if provided).
      - Group by `key_cols`, summing measures.
      - Sort descending by page_like_count, duration_hours, ads_count (lexicographic).
      - Return top 10 rows.

    Args:
        df: Enriched DataFrame.
        key_cols: Columns to group and rank by (e.g., ['ad_archive_id','page_name']).
        filter_expr: Optional pandas query string (e.g., 'display_format == "IMAGE"').

    Returns:
        pd.DataFrame: Top 10 ranking with measures and keys.
    """
    measures = ['page_like_count', 'duration_hours', 'ads_count']
    tmp = df
    if filter_expr:
        tmp = tmp.query(filter_expr)

    grouped = (
        tmp[key_cols + measures]
        .groupby(key_cols, dropna=False, as_index=False)
        .sum(numeric_only=True)
    )
    ranked = grouped.sort_values(by=measures, ascending=[False, False, False])
    return ranked.head(10)


def write_result(df: pd.DataFrame, file_name: str) -> None:
    """Write the cleaned dataset to CSV, creating folders as needed.

    Args:
        df: DataFrame to persist.
        file_name: Output path (e.g., 'result/clean_dataset_facebook_ads.csv').
    """
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_name, index=False)


@app.route("/")
def home():
    """Flask route: runs ETL and renders 4 ranking tables."""
    # 1) Extract → 2) Transform → 3) Persist cleaned data
    extracted_data = extract_from_json()
    transformed_df = transform(extracted_data)
    write_result(transformed_df, 'result/clean_dataset_facebook_ads.csv')

    # 4) Produce ranked views for the HTML page
    rank_by_ads_id = rank_performance_by_columns(transformed_df, ['ad_archive_id', 'page_name'])
    rank_by_name = rank_performance_by_columns(transformed_df, ['page_name'])
    rank_by_language = rank_performance_by_columns(transformed_df, ['language'])
    rank_by_image_format = rank_performance_by_columns(
        transformed_df, ['page_name'], 'display_format == "IMAGE"'
    )

    
    table_html1 = rank_by_ads_id.to_html(index=False, border=0)
    table_html2 = rank_by_name.to_html(index=False, border=0)
    table_html3 = rank_by_language.to_html(index=False, border=0)
    table_html4 = rank_by_image_format.to_html(index=False, border=0)

    return render_template_string(
        TEMPLATE,
        table1=table_html1,
        table2=table_html2,
        table3=table_html3,
        table4=table_html4
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8800, debug=True)

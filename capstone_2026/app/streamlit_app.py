import streamlit as st
from pathlib import Path
import pandas as pd

SAMPLES_DIR = Path("data/samples")

def available_months():
    months = []
    for p in sorted(SAMPLES_DIR.glob("month=*/demo_sample.parquet")):
        # p is like data/samples/month=2021-06/demo_sample.parquet
        month = p.parent.name.replace("month=", "")
        months.append(month)
    return months

@st.cache_data(show_spinner=False)
def load_month_sample(month: str) -> pd.DataFrame:
    fp = SAMPLES_DIR / f"month={month}" / "demo_sample.parquet"
    return pd.read_parquet(fp)

st.title("Dynamic Ride Pricing System — Demo")

months = available_months()
if not months:
    st.error("No monthly samples found. Run: python scripts/make_sample.py --all_months")
    st.stop()

month = st.selectbox("Select month", months, index=months.index("2021-06") if "2021-06" in months else 0)

df = load_month_sample(month)

st.caption(f"Loaded sample for **{month}** — rows: {len(df):,}")
st.dataframe(df.head(50), use_container_width=True)

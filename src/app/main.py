import streamlit as st
from pathlib import Path

#Let's build the navigation of the dashboard
ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

home = st.Page(ROOT / "src/app/presentation.py", title="Portfolio Optimizer")
configuration = st.Page(ROOT / "src/app/configuration.py", title="Configuration")
analytics = st.Page(ROOT / "src/app/analytics.py", title="Analytics")

nav = st.navigation(
    {
        "Presentation": [home],
        "Portfolio": [configuration, analytics],
    }
)

nav.run()

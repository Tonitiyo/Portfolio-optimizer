import streamlit as st

#Let's build the navigation of the dashboard

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

home = st.Page("/Users/charlieduret/Portfolio-optimizer/src/app/home.py", title="Home", icon="🏠")
portfolio = st.Page("/Users/charlieduret/Portfolio-optimizer/src/app/input.py", title="Input")

nav = st.navigation(
    {
        "Presentation": [home],
        "Portfolio Construction": [portfolio],
    }
)

nav.run()

import streamlit as st
from configuration import asset, start_date, end_date
from data import get_prices

st.title("Portfolio Analytics")
st.write("In this section the goal is to analyse through graphics and metrics the overall portfolio performance")

data = get_prices(asset["ticker"], start_date, end_date)


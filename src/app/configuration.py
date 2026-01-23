import streamlit as st

st.title("Portfolio Configuration")
st.write("In this section the goal is to choose every single parameters that you will need for the portfolio optimization.")

# This section is to add assets for the user by settings several information.
st.subheader("Portfolio Assets")

# Initialize session state
if "assets" not in st.session_state:
    st.session_state.assets = [
        {"type": "Equity", "ticker": "", "allocation": ""}
    ]
else:
    for asset in st.session_state.assets:
        asset.setdefault("allocation", "")

col_type, col_ticker, col_allocation, col_delete = st.columns([1, 1, 1, 1])

# Columns titles
with col_type:
    st.write("Type")

with col_ticker:
    st.write("Ticker Name")

with col_allocation:
    st.write("Allocation(%)")

with col_delete:
    st.write(" ")

# Render asset rows
for i, asset in enumerate(st.session_state.assets):


    with col_type:
        asset["type"] = st.selectbox(
            "",
            ["Equity", "Index"],
            index=["Equity", "Index"].index(asset["type"]),
            key=f"type_{i}",
            label_visibility="hidden",
        )

    with col_ticker:
        asset["ticker"] = st.text_input(
            "",
            value=asset["ticker"],
            key=f"ticker_{i}",
            label_visibility="hidden",
        )
    
    with col_allocation:
        asset["allocation"] = st.text_input(
            "",
            value=asset["allocation"],
            key=f"allocation_{i}",
            label_visibility="hidden",
        )

    with col_delete:
        if st.button("Delete", key=f"delete_{i}"):
            st.session_state.assets.pop(i)
            st.rerun()

# Add new row
if st.button("Add asset"):
    st.session_state.assets.append(
        {"type": "Equity", "ticker": "", "allocation": ""}
    )
    st.rerun()


st.divider()
st.subheader("Initial Settings")

col_startdate, col_enddate = st.columns(2)

with col_startdate:
    start_date = st.date_input("Start Date")

with col_enddate:
    end_date = st.date_input("End Date")

initial_amount = st.text_input("Initial Amount")

cashflow = st.selectbox("Cashflow", ["None", "Contribute Fixed Amount", "Withdraw fixed amount"])

if cashflow == "Contribute Fixed Amount":
    amount = st.text_input("Amount")
    contribution_frequency = st.selectbox("Contribution Frequency", ["Annualy", "Monthly", "Quarterly"])
elif cashflow =="Withdraw fixed amount":
    amount = st.text_input("Amount")
    withdrawal_frequency = st.selectbox("Withdrawal Frequency", ["Annualy", "Monthly", "Quarterly"])

rebalancing = st.selectbox("Rebalancing", ["Annualy", "Semi-Annually", "Quarterly", "Monthly"])

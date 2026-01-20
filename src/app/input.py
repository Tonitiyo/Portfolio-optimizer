import streamlit as st

st.title("Portfolio Input")
st.write("In this section the goal is to choose every single parameters that you will need for the portfolio optimization.")


st.subheader("Portfolio Assets")

# Initialize session state
if "assets" not in st.session_state:
    st.session_state.assets = [
        {"type": "Equity", "ticker": "", "allocation": ""}
    ]
else:
    for asset in st.session_state.assets:
        asset.setdefault("allocation", "")


# Render asset rows
for i, asset in enumerate(st.session_state.assets):
    col_type, col_ticker, col_allocation, col_delete = st.columns([1, 1, 1, 1])

    with col_type:
        asset["type"] = st.selectbox(
            "",
            ["Equity", "ETF", "Crypto"],
            index=["Equity", "ETF", "Crypto"].index(asset["type"]),
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
        st.write("")
        st.write("")
        if st.button("Delete", key=f"delete_{i}"):
            st.session_state.assets.pop(i)
            st.rerun()

# Add new row
if st.button("Add asset"):
    st.session_state.assets.append(
        {"type": "Equity", "ticker": "", "allocation": ""}
    )
    st.rerun()

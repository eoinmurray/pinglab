import streamlit as st

from pages.components.calibrator_form import render_calibrator_form
from pages.components.calibrator_plot import render_calibrator_plot

st.set_page_config(
    page_title="Calibrator",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("Calibrator")
st.caption("Stub page")

col_left, col_right = st.columns([1, 2])

with col_left:
    render_calibrator_form()

with col_right:
    render_calibrator_plot()

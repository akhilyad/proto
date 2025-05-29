import streamlit as st
from typing import Optional

def show_error(message: str, user_message: Optional[str] = None) -> None:
    """
    Display an error message in Streamlit and log it.
    """
    st.error(user_message or f"An error occurred: {message}. Please try again or contact support.")


def show_tooltip(text: str) -> None:
    """
    Display a tooltip (info box) in Streamlit.
    """
    st.info(text)


def dark_mode_toggle() -> None:
    """
    Add a dark mode toggle to the sidebar (Streamlit native dark mode is experimental).
    """
    st.sidebar.markdown("""
        <style>
        body[data-theme="dark"] {
            background-color: #222 !important;
            color: #eee !important;
        }
        </style>
    """, unsafe_allow_html=True)
    # Streamlit's dark mode is handled in settings, but you can add a toggle for custom CSS if needed.


def validate_positive_number(value: float, field_name: str) -> None:
    """
    Validate that a number is positive, else show an error.
    """
    if value <= 0:
        st.error(f"{field_name} must be positive.") 
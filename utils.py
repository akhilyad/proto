import streamlit as st
from typing import Callable, Any, Dict
import datetime

@st.cache_data(show_spinner=False)
def cached_function(func: Callable, *args, **kwargs) -> Any:
    """
    Cache the result of an expensive function using Streamlit's cache_data.
    """
    return func(*args, **kwargs)


def localize(text: str, lang: str = 'en', translations: Dict[str, Dict[str, str]] = None) -> str:
    """
    Simple localization utility. Provide a translations dict: {lang: {text: translation}}.
    """
    if translations and lang in translations and text in translations[lang]:
        return translations[lang][text]
    return text


def format_date(dt: datetime.datetime, fmt: str = '%Y-%m-%d') -> str:
    """
    Format a datetime object as a string.
    """
    return dt.strftime(fmt) 
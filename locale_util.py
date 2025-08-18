import streamlit as st


def init_language_selector():
    """Render language selector in sidebar and store selection in session state.
    Values: 'en' (English), 'et' (Estonian). Defaults to 'en'.
    """
    with st.sidebar:
        st.markdown("---")
        st.header("Language")
        current = st.session_state.get('lang', 'en')
        lang = st.radio("Select language", options=["en", "et"], index=0 if current == 'en' else 1)
        st.session_state['lang'] = lang


def get_lang() -> str:
    return st.session_state.get('lang', 'en')


def tr(en_text: str, et_text: str) -> str:
    """Translate inline strings based on current language selection."""
    return et_text if get_lang() == 'et' else en_text



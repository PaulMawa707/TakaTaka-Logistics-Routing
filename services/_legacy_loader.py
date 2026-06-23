"""Load legacy Streamlit modules with a no-op streamlit stub."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _noop(*args, **kwargs):
    return None


def _mock_streamlit():
    st = ModuleType("streamlit")
    st.info = st.error = st.warning = st.write = st.dataframe = st.markdown = _noop
    st.subheader = st.caption = st.success = st.spinner = st.balloons = _noop
    st.session_state = {}
    st.form = lambda *a, **k: SimpleNamespace(__enter__=lambda s: s, __exit__=lambda *a: None)
    st.columns = lambda n: [SimpleNamespace() for _ in range(n)]
    st.file_uploader = st.date_input = st.slider = st.checkbox = st.selectbox = st.text_input = _noop
    st.form_submit_button = st.button = lambda *a, **k: False
    sys.modules["streamlit"] = st
    return st


def load_module(filename: str, module_name: str):
    _mock_streamlit()
    path = Path(__file__).resolve().parent.parent / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required module file missing: {filename}. "
            f"Ensure it is deployed and not listed in .vercelignore."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

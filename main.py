import streamlit as st
import os
import base64
import importlib.util
from pathlib import Path

st.set_page_config(page_title="TakaTaka Logistics Tools", layout="wide")


def load_local_module(module_name, candidate_filenames):
    base_dir = Path(__file__).parent

    # Prefer explicit local files first (avoids importing stdlib modules like `test`).
    for filename in candidate_filenames:
        module_path = base_dir / filename
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    # Fallback to regular import if local files are not found.
    try:
        return __import__(module_name)
    except Exception:
        pass

    raise ModuleNotFoundError(
        f"Could not load module '{module_name}'. Tried: {', '.join(candidate_filenames)}"
    )


test = load_local_module("test", ["test.py", "test (1).py"])
test3 = load_local_module("test3", ["test3.py", "test3 (1).py"])
weekly_dispatch = load_local_module("weekly_dispatch", ["weekly_dispatch.py", "weekly_dispatch (1).py"])

# =========================================================
# Page Config
# =========================================================
# Must be called before any imported module runs Streamlit commands.

# =========================================================
# Shared background and logo
# =========================================================
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

def set_background(image_path="TakaTaka_thumbnail_solution.jpg"):
    bg_image = get_base64_image(image_path)
    if bg_image:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{bg_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

def show_logo_top_right(image_path="taka-taka-solutions-logo (1).jpg", width=120):
    logo_base64 = get_base64_image(image_path)
    if logo_base64:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div></div>
                <div style="margin-right: 1rem;">
                    <img src="data:image/png;base64,{logo_base64}" width="{width}">
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# UI Setup
# =========================================================
set_background()
show_logo_top_right()
st.markdown("<br>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🚚 Navigation")
page = st.sidebar.radio(
    "Select Application",
    [
        "Optimized Orders Uploader (Nearest-First)",
        "Strict Orders Uploader (Excel Order)",
        "Scheduled Weekly Dispatch (Template Routes)",
    ]
)

# =========================================================
# Page routing
# =========================================================
if page == "Optimized Orders Uploader (Nearest-First)":
    st.header("📦 Logistics Optimized Orders Uploader")
    test.run_wialon_uploader()

elif page == "Strict Orders Uploader (Excel Order)":
    st.header("📋 Logistics Strict Orders Uploader")
    test3.run_wialon_uploader()

elif page == "Scheduled Weekly Dispatch (Template Routes)":
    st.header("🗓️ Scheduled Weekly Dispatch")
    weekly_dispatch.run_weekly_dispatch()

# =========================================================
# Footer (optional)
# =========================================================
st.markdown(
    "<hr><center>© 2025 TakaTaka Solutions | Powered by Streamlit</center>",
    unsafe_allow_html=True,
)

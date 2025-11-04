import streamlit as st
import os
import base64
import test
import test3

# =========================================================
# Page Config
# =========================================================
st.set_page_config(page_title="TakaTaka Logistics Tools", layout="wide")

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
        "Strict Orders Uploader (Excel Order)"
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

# =========================================================
# Footer (optional)
# =========================================================
st.markdown(
    "<hr><center>© 2025 TakaTaka Solutions | Powered by Streamlit</center>",
    unsafe_allow_html=True,
)

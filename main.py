
import streamlit as st
import subprocess
import sys
import os
import base64

st.set_page_config(page_title="Logistics Route Launcher", page_icon="🚚", layout="centered")


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def set_background():
    try:
        background_image = get_base64_image("TakaTaka_thumbnail_solution.jpg")
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{background_image}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        pass


def show_logo_top_right(image_path, width=120):
    try:
        logo_base64 = get_base64_image(image_path)
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
    except Exception:
        pass


set_background()
show_logo_top_right("taka-taka-solutions-logo (1).jpg", width=120)
st.markdown("<br>", unsafe_allow_html=True)

st.title("🚦 Logistics Route Launcher")
st.markdown("### Choose which app to run below:")

app_choice = st.selectbox(
    "Select App",
    [
        "Optimized Route (Shortest Path)",
        "Excel Sequence (As Listed)"
    ]
)

app_files = {
    "Optimized Route (Shortest Path)": "test.py",
    "Excel Sequence (As Listed)": "test3.py",
}

selected_file = app_files[app_choice]

#st.info(f"Selected app: **{selected_file}**")

if st.button("▶️ Run Selected App"):
    st.success(f"Launching {selected_file}...")
    python_exe = sys.executable  # Use the same Python environment
    script_path = os.path.join(os.getcwd(), selected_file)

    if not os.path.exists(script_path):
        st.error(f"File not found: {script_path}")
    else:
        # Launch Streamlit app as a separate process
        subprocess.Popen(["streamlit", "run", script_path])
        st.stop()

st.markdown("---")
st.caption("Created by Mawa — unified launcher for Logistics route planning apps.")

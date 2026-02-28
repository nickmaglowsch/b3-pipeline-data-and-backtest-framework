"""
Convenience launcher for the B3 Data Pipeline Streamlit UI.

Usage:
    python run_ui.py

or directly:
    streamlit run ui/app.py
"""
import os

if __name__ == "__main__":
    os.execvp("streamlit", ["streamlit", "run", "ui/app.py"])

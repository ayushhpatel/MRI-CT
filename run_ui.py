"""
Launcher script for MRI→CT CycleGAN Streamlit UI

Run this script to start the web interface:
    python run_ui.py

Or run directly with Streamlit:
    streamlit run src/ui/app.py
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit UI application."""
    
    # Check if streamlit is installed
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ Streamlit installed successfully!")
    
    # Path to the UI application
    app_path = Path(__file__).parent / "src" / "ui" / "app.py"
    
    if not app_path.exists():
        print(f"❌ UI application not found at: {app_path}")
        return
    
    print("🚀 Starting MRI→CT CycleGAN UI...")
    print(f"📍 Application path: {app_path}")
    print("🌐 Opening browser at http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the server")
    
    # Launch streamlit
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", "8501",
        "--server.headless", "false"
    ])

if __name__ == "__main__":
    main()
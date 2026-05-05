# Yuji Divergent Fist – Shadow Clone

A Python + OpenCV/MediaPipe project that simulates Jujutsu Kaisen-inspired "Shadow Clone" and "Divergent Fist" effects.  
This module integrates gesture detection, cursed energy overlays, and debugging utilities.

---

## 🚀 Features
- **Hand/Fist Detection** – via `fist_detector.py`
- **Cursed Energy Simulation** – handled in `cursed_energy.py`
- **Configurable Parameters** – stored in `config.py`
- **Debugging Tools** – includes `debug_video.py`, `debug_frame.png`, `debug_mask.png`
- **Main Runner** – `main.py` orchestrates the pipeline

---

## 📂 Project Structure
shadowClone/
├── assets/               # Visual overlays and effect textures
├── pycache/          # Auto-generated Python cache (ignored)
├── config.py             # Configuration settings
├── cursed_energy.py      # Energy effect logic
├── fist_detector.py      # Gesture/fist detection
├── main.py               # Entry point
├── debug_video.py        # Debugging utilities
├── debug_frame.png       # Debug frame sample
├── debug_mask.png        # Debug mask sample
├── requirements.txt      # Dependencies


---

## ⚙️ Installation
```bash
# Clone repository
git clone https://github.com/Aryann-gd/yuji-divergent-fist.git
cd yuji-divergent-fist/shadowClone

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\\Scripts\\activate    # Windows

# Install dependencies
pip install -r requirements.txt

▶️ Usage
Configure parameters in config.py (e.g., thresholds, overlay paths).

Run the main script:python main.py

python main.py
Debugging:

Use debug_video.py for frame-by-frame inspection.

Check debug_frame.png and debug_mask.png for visual debugging

📜 Notes
Ensure webcam access is enabled.

Assets must remain in the assets/ folder for overlays to load correctly.

Tested on Python 3.9+ with OpenCV and MediaPipe.

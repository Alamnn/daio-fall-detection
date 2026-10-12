# Contributing to DAÏO

Thank you for your interest in contributing to **DAÏO (Intelligent Vision-Based Fall Detection System)**!

DAÏO is designed to be an edge-capable, high-specificity computer vision system that runs locally with physical kinematic constraints, protecting privacy while minimizing false alarms from routine Activities of Daily Living (ADLs).

---

## Code of Conduct
Please ensure all discussions, issue comments, and pull requests remain respectful, constructive, and focused on improving assistive AI safety.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/daio-fall-detection.git
   cd daio-fall-detection
   ```

2. **Create a Python 3.11+ virtual environment**:
   Using `uv` (recommended):
   ```bash
   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
   Or standard `python`:
   ```bash
   python3.11 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Generate synthetic benchmark clips**:
   ```bash
   python tools/generate_demo_videos.py
   ```

4. **Verify baseline tests**:
   ```bash
   python main.py --benchmark
   ```

---

## Pull Request Guidelines

- **Adhere to Physics Constraints**: Any changes to the state-space model or kinematic logic must preserve scale invariance and prevent false positives on ADLs (squats, prayer prostrations, rapid sitting).
- **Edge Efficiency**: Keep inference lightweight. Avoid introducing heavy neural networks that compromise real-time FPS on low-power devices like the Raspberry Pi 4.
- **Local Privacy**: DAÏO must remain 100% offline-first. Never stream raw video feeds to external services without explicit opt-in settings.
- **Formatting**: Format code according to PEP 8 standards. Include informative docstrings and type annotations.

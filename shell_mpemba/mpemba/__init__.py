"""Energy-shell predictions and validation for the impurity spin chain.

Import the module for the calculation you need; importing mpemba performs no
simulation and loads no numerical libraries. Paths are relative to this checkout.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache" / "spin_chain"
RESULTS_DIR = PROJECT_ROOT / "results_chaotic_wavefunction"
PROGRAMME_DIR = RESULTS_DIR / "article_programme"
PLOTS_DIR = PROJECT_ROOT / "paper_plots" / "article"

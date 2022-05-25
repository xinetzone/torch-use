import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.extend([str(ROOT)])

from myst_conf import *
from mynb_conf import *

# load extensions
extensions = ["myst_nb"]



# specify project details
project = "MyST-NB Quickstart"

# basic build settings
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
nitpicky = True

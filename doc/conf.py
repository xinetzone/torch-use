import sys
from pathlib import Path
import ablog

ROOT = Path(__file__).parent.resolve()
sys.path.extend([str(ROOT), str(ROOT/"src")])
from _conf import *

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates', ablog.get_html_templates_path()]

# load extensions
extensions = [
    "ablog",
    "myst_nb",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    # "sphinx_thebe",
    "sphinx_copybutton",
    "sphinx_comments",
    "sphinxcontrib.mermaid",
    "sphinx_design",
    "sphinx_inline_tabs",
    # "sphinx.ext.autodoc",
    # "sphinx.ext.autosummary",
    "sphinx_automodapi.automodapi",
    # "sphinx.ext.todo",
    # "sphinxcontrib.bibtex",
    # "sphinx_togglebutton",
    # "sphinx.ext.viewcode",
    # "sphinx.ext.doctest",
    # "sphinx_design",
    # "sphinx.ext.ifconfig",
    # "sphinxext.opengraph",
]

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# basic build settings
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = ["default.css"]
# -- 国际化输出 ----------------------------------------------------------------
language = 'zh_CN'
locale_dirs = ['locales/']  # path is example but recommended.
gettext_compact = False  # optional.
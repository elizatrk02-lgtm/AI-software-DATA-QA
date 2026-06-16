import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'AI Data QA Platform'
copyright = '2026, Startup Founder'
author = 'Startup Founder'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

html_theme = 'alabaster'

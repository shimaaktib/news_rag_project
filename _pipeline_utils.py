"""
Filenames like `06_retrieve_context.py` can't be imported with a normal
`import 06_retrieve_context` statement (Python module names can't start with
a digit). This tiny helper loads them by file path instead, so
streamlit_app.py and the later pipeline steps can reuse the earlier steps
without renaming any of the required files.
"""

import importlib.util
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_module(filename, module_name):
    """Load e.g. load_module('06_retrieve_context.py', 'retrieve_context')."""
    path = os.path.join(BASE_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

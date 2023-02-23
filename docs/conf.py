# Basic configuration
project = 'SofaCUDALinearSolver'
copyright = '2023, DEFROST Team, Inria & Cristal'
author = 'DEFROST Team'

# Extensions to use
extensions = [ "breathe", "exhale"]

# Configuration for the breathe extension
# Which directory to read the Doxygen output from
breathe_projects = {"SofaCUDALinearSolver":"xml"}
breathe_default_project = "SofaCUDALinearSolver"

# Configuration for the exhale extensions
exhale_args = {
    "containmentFolder": "./api",
    "doxygenStripFromPath": "../src",
    "rootFileName": "library_root.rst",
    "rootFileTitle": "Library API",
}

# Configuration for the theme
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/SofaDefrost/SofaCUDALinearSolver",
    "use_repository_button": True,
}

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'PyTomoATT'
copyright = '2023, Mijian Xu'
author = 'Mijian Xu'
release = '0.1.4'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.githubpages',
              "sphinx.ext.intersphinx",
              'sphinx.ext.autodoc',
              "sphinx_copybutton",
              "sphinx_design",
              "myst_nb",
              "sphinx.ext.mathjax",
              "sphinx.ext.todo",
              "sphinx.ext.viewcode",
            #   "sphinxcontrib.icon",
            #   "sphinx_gallery.gen_gallery",
            #   'sphinxcontrib.bibtex',
]


templates_path = ['_templates']
exclude_patterns = []


myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# sphinx_gallery_conf = {
#     # path to your examples scripts
#     "examples_dirs": [
#         "./examples/matplotlib",
#     ],
#     # path where to save gallery generated examples
#     "gallery_dirs": ['seismo/matplotlib'],
#     # Patter to search for example files
#     "filename_pattern": r"\.py",
#     # Remove the "Download all examples" button from the top level gallery
#     "download_all_examples": False,
#     # Sort gallery example by file name instead of number of lines (default)
#     # "within_subsection_order": ExampleTitleSortKey,
#     # directory where function granular galleries are stored
#     "backreferences_dir": "api/generated/backreferences",
#     # Modules for which function level galleries are created.  In
#     # this case sphinx_gallery and numpy in a tuple of strings.
#     # "doc_module": "seispy",
#     # Insert links to documentation of objects in the examples
#     "reference_url": {"pygmt": None},
#     # Removes configuration comments from scripts
#     "remove_config_comments": True,
# }
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_theme_options = {
    "sidebar_hide_name": False,
}
html_static_path = ['_static']

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = True
copybutton_remove_prompts = True


jupyter_execute_notebooks = "cache"
html_extra_path = []
html_title = project
html_css_files = ["custom.css"]
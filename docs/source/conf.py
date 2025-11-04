# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'kioptipack-blueprints'
copyright = '2025, WZL-IQS RWTH Aachen University'
author = 'Alexander Nasuta, Mats Gesenhues'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",

    "sphinx.ext.duration",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.autodoc",
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',

    "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_css_files = ['custom.css']

html_logo = '_static/logo-optipack.svg'
html_title = 'KIOptiPack Blueprints'

copybutton_prompt_text = ">>> "
copybutton_prompt_is_regexp = False

nbsphinx_execute = 'never'     # 'auto' oder 'always' falls Ausführung gewünscht
nbsphinx_timeout = 60
nbsphinx_allow_errors = True



from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util.docutils import SphinxDirective

class PrereqDirective(SphinxDirective):
    has_content = True
    optional_arguments = 1

    def run(self):
        title = self.arguments[0] if self.arguments else "Prerequisites"
        node = nodes.admonition()
        node['classes'] += ['caution']  # Nutze z. B. 'important'
        title_node = nodes.title(text=title)
        node += title_node
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


class TutorialDirective(SphinxDirective):
    has_content = True
    optional_arguments = 1

    def run(self):
        title = self.arguments[0] if self.arguments else "Tutorial"
        node = nodes.admonition()
        node['classes'] += ['seealso']  # Nutze z. B. 'important'
        title_node = nodes.title(text=title)
        node += title_node
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]

def setup(app):
    app.add_directive("prereq", PrereqDirective)
    app.add_directive("tutorial", TutorialDirective)


"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based
"""

#######################################
## Configuration and Helpers for PyDoit
#######################################

## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./cmds/")

from os import getcwd

## Custom reporter: Print PyDoit Text in Green
# This is helpful because some tasks write to sterr and pollute the output in
# the console. I don't want to mute this output, because this can sometimes
# cause issues when, for example, LaTeX hangs on an error and requires
# presses on the keyboard before continuing. However, I want to be able
# to easily see the task lines printed by PyDoit. I want them to stand out
# from among all the other lines printed to the console.
from doit.reporter import ConsoleReporter
from colorama import Fore, Style, init


class GreenReporter(ConsoleReporter):
    def write(self, stuff, **kwargs):
        self.outstream.write(Fore.GREEN + stuff + Style.RESET_ALL)


DOIT_CONFIG = {
    "reporter": GreenReporter,
    # other config here...
    # "cleanforget": True, # Doit will forget about tasks that have been cleaned.
}
init(autoreset=True)

## Helper for determining OS
import platform


def get_os():
    os_name = platform.system()
    if os_name == "Windows":
        return "windows"
    elif os_name == "Darwin":
        return "nix"
    elif os_name == "Linux":
        return "nix"
    else:
        return "unknown"


os_type = get_os()

##################################
## Begin rest of PyDoit tasks here
##################################
import config
from pathlib import Path
from doit.tools import run_once

OUTPUT_DIR = Path(config.OUTPUT_DIR)
DATA_DIR = Path(config.DATA_DIR)


## Helpers for handling Jupyter Notebook tasks
# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook):
    return f'jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace "./discussions/{notebook}.ipynb"'
def jupyter_to_html(notebook, output_dir=OUTPUT_DIR):
    return f'jupyter nbconvert --to html --output-dir={output_dir} "./discussions/{notebook}.ipynb"'
def jupyter_to_md(notebook, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f'jupytext --to markdown --output-dir={output_dir} "./discussions/{notebook}.ipynb"'
def jupyter_to_python(notebook, build_dir):
    """Convert a notebook to a python script"""
    return f'jupyter nbconvert --to python "./discussions/{notebook}.ipynb" --output "_{notebook}.py" --output-dir {build_dir}'
def jupyter_clear_output(notebook):
    return f'jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace "./discussions/{notebook}.ipynb"'
# fmt: on


def copy_notebook_to_folder(notebook_stem, origin_folder, destination_folder):
    origin_path = Path(origin_folder) / f"{notebook_stem}.ipynb"
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination_path = destination_folder / f"_{notebook_stem}.ipynb"
    if os_type == "nix":
        command = f'cp "{origin_path}" "{destination_path}"'
    else:
        command = f'copy  "{origin_path}" "{destination_path}"'
    return command


def task_setup():
    """ """
    file_dep = ["./cmds/config.py"]
    file_output = []
    targets = []

    return {
        "actions": [
            "python ./cmds/config.py",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": [],  # Don't clean these files by default. The ideas
        # is that a data pull might be expensive, so we don't want to
        # redo it unless we really mean it. So, when you run
        # doit clean, all other tasks will have their targets
        # cleaned and will thus be rerun the next time you call doit.
        # But this one wont.
        # Use doit forget --all to redo all tasks. Use doit clean
        # to clean and forget the cheaper tasks.
    }


##############################$
## Demo: Other misc. data pulls
##############################$
# def task_pull_fred():
#     """ """
#     file_dep = [
#         "./src/load_bloomberg.py",
#         "./src/load_CRSP_Compustat.py",
#         "./src/load_CRSP_stock.py",
#         "./src/load_fed_yield_curve.py",
#         ]
#     file_output = [
#         "bloomberg.parquet",
#         "CRSP_Compustat.parquet",
#         "CRSP_stock.parquet",
#         "fed_yield_curve.parquet",
#         ]
#     targets = [DATA_DIR / "pulled" / file for file in file_output]

#     return {
#         "actions": [
#             "ipython ./src/load_bloomberg.py",
#             "ipython ./src/load_CRSP_Compustat.py",
#             "ipython ./src/load_CRSP_stock.py",
#             "ipython ./src/load_fed_yield_curve.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": [],  # Don't clean these files by default.
#     }

# def task_summary_stats():
#     """ """
#     file_dep = ["./src/example_table.py"]
#     file_output = [
#         "example_table.tex",
#         "pandas_to_latex_simple_table1.tex",
#     ]
#     targets = [OUTPUT_DIR / file for file in file_output]

#     return {
#         "actions": [
#             "ipython ./src/example_table.py",
#             "ipython ./src/pandas_to_latex_demo.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#     }


# def task_example_plot():
#     """Example plots"""
#     file_dep = [Path("./src") / file for file in ["example_plot.py", "load_fred.py"]]
#     file_output = ["example_plot.png"]
#     targets = [OUTPUT_DIR / file for file in file_output]

#     return {
#         "actions": [
#             "ipython ./src/example_plot.py",
#         ],
#         "targets": targets,
#         "file_dep": file_dep,
#         "clean": True,
#     }


notebook_tasks = {
    "D.0. Introduction.ipynb": {
        "file_dep": [],
        "targets": [],
    },
    "D.1.X. Debt Markets.ipynb": {
        "file_dep": [],
        "targets": [],
    },
    "D.1. Treasury Debt.ipynb": {
        "file_dep": [],
        "targets": [],
    },
    "D.2. Money Markets.ipynb": {
        "file_dep": [],
        "targets": [],
    },
    "D.3. Inflation.ipynb": {
        "file_dep": [],
        "targets": [],
    },
    "D.4. Commodity Futures.ipynb": {
        "file_dep": [],
        "targets": [],
    },
    # "D.3. The Fed.ipynb": {
    #     "file_dep": [],
    #     "targets": [],
    # },
    # "D.5. Equity Indexes and ETFs.ipynb": {
    #     "file_dep": [],
    #     "targets": [],
    # },
    "D.6. Currency.ipynb": {
        "file_dep": [],
        "targets": [],
    },
}


def task_convert_notebooks_to_scripts():
    """Convert notebooks to script form to detect changes to source code rather
    than to the notebook's metadata.
    """
    build_dir = Path(OUTPUT_DIR)
    build_dir.mkdir(parents=True, exist_ok=True)

    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".ipynb")[0]
        yield {
            "name": notebook,
            "actions": [
                # jupyter_execute_notebook(notebook_name),
                # jupyter_to_html(notebook_name),
                # copy_notebook_to_folder(notebook_name, Path("./src"), "./docs/_notebook_build/"),
                jupyter_clear_output(notebook_name),
                jupyter_to_python(notebook_name, build_dir),
            ],
            "file_dep": [Path("./discussions") / notebook],
            "targets": [OUTPUT_DIR / f"_{notebook_name}.py"],
            "clean": True,
            "verbosity": 0,
        }

def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """

    for notebook in notebook_tasks.keys():
        notebook_name = notebook.split(".ipynb")[0]
        yield {
            "name": notebook,
            "actions": [
                jupyter_execute_notebook(notebook_name),
                jupyter_to_html(notebook_name),
                copy_notebook_to_folder(
                    notebook_name, Path("./discussions"), "./docs/_notebook_build/"
                ),
                jupyter_clear_output(notebook_name),
                # jupyter_to_python(notebook_name, build_dir),
            ],
            "file_dep": [
                OUTPUT_DIR / f"_{notebook_name}.py",
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
            # "verbosity": 1,
        }


# def task_compile_sphinx_docs():
#     """Compile Sphinx Docs"""
#     file_dep = [
#         "./docs/conf.py",
#         "./docs/index.rst",
#         "./docs/myst_markdown_demos.md",
#     ]
#     targets = [
#         "./docs/_build/html/index.html",
#         "./docs/_build/html/myst_markdown_demos.html",
#     ]

#     return {
#         "actions": ["sphinx-build -M html ./docs/ ./docs/_build"],
#         "targets": targets,
#         "file_dep": file_dep,
#         "task_dep": ["run_notebooks"],
#         "clean": True,
#     }


def task_compile_book():
    """Run jupyter-book build to compile the book."""

    file_dep = [
        "./docs/myst_markdown_demos.md",
        "./docs/_config.yml",
        "./docs/_toc.yml",
    ]

    targets = [
        "./docs/_build/html/index.html",
        "./docs/_build/html/myst_markdown_demos.html",
    ]

    return {
        "actions": [
            # "jupyter-book build -W ./docs",
            "jupyter-book build ./docs",
        ],
        "targets": targets,
        "file_dep": file_dep,
        "clean": True,
        "task_dep": ["run_notebooks"],
    }


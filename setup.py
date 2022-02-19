"""
Python setuptools config to package this repository.

Unfortunately we can't import any non-standard python modules, even the ones specified in build requirements because
it will break load_setup_py_data() in the conda recipe.  This is a known bug that should be fixed in the future.
"""

from setuptools import setup, find_namespace_packages
from pathlib import Path

setup(
    name="rube",
    version="0.5.0",  # automatically updated by deploy tool
    package_dir={"": "lib"},
    packages=find_namespace_packages(where="lib"),
    data_files=[(str(dir), [str(fn) for fn in Path(dir).glob("*") if fn.is_file()]) for dir in Path("share").glob("**/*") if dir.is_dir()],
    scripts=[str(fn) for fn in Path("scripts").glob("**/*") if fn.is_file()],
    include_package_data=True,
    zip_safe=False,  # implicit namespaces do not appear to import from zipped egg
)

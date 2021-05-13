# ortools

Tools and scripts for [OpenRocket](https://openrocket.info/) using Python and
[orhelper](https://pypi.org/project/orhelper/).


## Installation

1. Download or clone the project.
2. Create and activate some conda or pip environment with Python 3. E.g.:

   ```shell
   conda create -n ortools python=3
   conda activate ortools
   ```

3. If you are a developer, install the package in editable mode with pip. If not, leave out the `-e`
   flag for a normal installation. In editable mode, all changes to the source files are immediately
   reflected in your environment, so you don't have to reinstall/update the package after changes.

   ```shell
   cd ortools
   pip install -e .
   ```

   All required packages will be automatically installed with pip too. If you don't like that you
   can check [setup.cfg](setup.cfg) for the dependencies and install them beforehand with conda
   (except for orhelper which is only available via pip).

   ```shell
   conda install numpy matplotlib click configparser
   pip install orhelper
   ```

## Usage

So far only one command line tool is in development. It is called `diana` and does dispersion
analyses. Just execute

```shell
diana -h
```

to get help and usage information.

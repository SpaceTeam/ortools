# ortools

Tools and scripts for [OpenRocket](https://openrocket.info/) using Python and
[orhelper](https://pypi.org/project/orhelper/). 

As an example result, see the image from a landing scatter visualization produced by the 6DOF dispersion analysis tool `diana`
![Sample image for landing scatter visualization by diana](https://github.com/SpaceTeam/ortools/blob/master/docs/landing_scatter_sample.jpg)


## Prerequisites
You must use Java 8 (a really, really old version) for this to work, see [orhelper](https://pypi.org/project/orhelper/).
Additionally, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for installing `jpype1`.

Tested with the following version combinations:
* miniconda3-py39 with python3.9, openjdk 1.8.0_312-1, orhelper 0.1.2, Windows 10
* miniconda3-py39 with python3.9, oracle jdk1.8.0_202, orhelper 0.1.2, Windows 10

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
   conda install numpy matplotlib click configparser simplekml scipy pyproj
   pip install orhelper
   ```
   If jpype install fails, check if Microsoft C++ Build Tools are installed, and `JAVA_HOME` is set properly. 
   If used with python 3.10, build jpype manually with [this source](https://github.com/jpype-project/jpype/commit/bbdca907d053f1e04e4dcd414d4ebce8f9da6313),
   see the [following github-PR](https://github.com/kylebarron/pydelatin/pull/24).

4. `diana` needs to find `OpenRocket-15.03.jar`. Put it in the same folder where you run `diana`, or add the file (not the folder)
   to an environment variable called `CLASSPATH`, 
   see [the oracle documentation](https://docs.oracle.com/javase/tutorial/essential/environment/paths.html).
   Check with (example for Windows)
   ```shell
   echo %CLASSPATH%
   ```
   >$ PATHTTO\projects\OpenRocket-15.03.jar
   ```shell
   java -jar "%CLASSPATH%"
   ```
   and OR should be launched.


## Usage

So far only one command line tool is in development. It is called `diana` and does dispersion
analyses. Just execute
```shell
diana
```
in the root directory of `ortools` to run with the provided examples.
Run
```shell
diana -h
```
to get help and usage information. For further information, check the [GitHub Wiki](https://github.com/SpaceTeam/ortools/wiki).


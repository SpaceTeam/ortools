# ortools

Tools and scripts for [OpenRocket](https://openrocket.info/) using Python and
[orhelper](https://pypi.org/project/orhelper/).
## Prerequisites
You must use Java 8 (a really, really old version) for this to work.
See [orhelper](https://pypi.org/project/orhelper/), 
additionally install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) for installing `jpype1`.
Tested with miniconda3-py39 ith python3.9, openjdk 1.8.0_312-1, orhelper 0.1.2 on Windows 10.
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

4. `diana` needs to find `OpenRocket-15.03.jar`. Call `diana` from the same folder or add the file (not the folder)
   to an environment variable called `CLASSPATH`, 
   see [the oracle documentation](https://docs.oracle.com/javase/tutorial/essential/environment/paths.html).
   Check with 
   ```shell
   echo %CLASSPATH%
   $ PATHTTO\projects\OpenRocket-15.03.jar
   java -jar OpenRocket-15.03.jar
   ```


## Usage

So far only one command line tool is in development. It is called `diana` and does dispersion
analyses. Just execute

```shell
diana -h
```

to get help and usage information.


### Procedure

1. Define your rocket in the `ork` file with nominal parameters.
2. Create a `ini` file with your simulation settings, see the examples in [examples](examples)
3. Fill the section `General` in the `ini` file:
   - SimulationIndex: Index of simulation defined in the `ork` file, start counting from 0
   - NumberOfSimulations: Number of simulations, which will be performed by `diana`
   - OrkFile: the name of your `ork` file
4. Configure the deviations for every supported parameter in the `ini`-file. If not specified otherwise, 
	normal distribution is used with mean value from the original `ork` file and the standard deviation set in your `ini` file.
	Currently supported are:
   - Section `Aerodynamics`
      - FinCant: Cant angle of the fins, calculated for every fincan in the design (degree)
	  - ParachuteCd: Drag coefficient, calculated for every parachute in the design (unitless)
	  - Roughness: Surface roughness used for drag calculation (m). 
	      Only discrete roughness-categories are used by OpenRocket, see the [Source](https://github.com/openrocket/openrocket/blob/unstable/core/src/net/sf/openrocket/rocketcomponent/ExternalComponent.java#L23-L32)
   - Section `LaunchRail`
      - [TiltMean (degree)]: consider  that for the first simulation the values set in OR will be used
      - Tilt Standard Deviation (degree)
      - AzimuthMean (degree, 0 is north): consider  that for the first simulation the values set in OR will be used
	  - Azimuth Standard Deviation (degree)	   
   - Section `Staging`: 
	Useful for rockets with more than one stage. Define the staging event (burnout, launch, etc) and the nominal delay in the `ork` file. A uniform distribution with the interval
	`(NominalDelay + StageSeparationDelayDeltaNeg, NominalDelay + StageSeparationDelayDeltaPos)` is then applied without changing the staging event.
      - StageSeparationDelayDeltaNeg (in s)
      - StageSeparationDelayDeltaPos (in s)
   - Section `Propulsion`
      - ThrustFactor: Factor applied to thrust curve (unitless, factor of 1 is nominal)
      - NozzleCrossSection: Calculates thrust increase due to decreased ambient pressure. This is set via the nozzle diameter (set 0 to deactivate this feature).
5. Run `diana`


### Limitations

- OpenRocket does not solve stable if high wind speeds and/or large parameter deviations are set. 
  Try to reduce the simulation step time, or lower the parameter deviations.
- Check that the simulation option **flat earth** or **WGS84** is set.


### Wind Data

Wind data can be given in an additional text file, containing the following data in three columns separated by whitespace:

- Altitude (in m)
- Direction (in °)
- Wind speed (in m/s)

Add the path to this file to the `WindModel` section to the key `DataFile`

Where to get these data? Aviation aloft data, or wind predictions from, e.g., windy.com

- USA
   - copy data from
   https://www.aviationweather.gov/windtemp/data?level=low&fcst=06&region=sfo&layout=on&date=
   - or alternatively AFTERNOON WINDS ALOFT FORECAST
   https://forecast.weather.gov/product.php?site=NWS&product=SRG&issuedby=REV
   - or https://rucsoundings.noaa.gov/
   - or could be downloaded automatically via
   https://api.weather.gov/products/types/FD8/locations/US7 and
   https://api.weather.gov/products/types/FD1/locations/US1
- Poland: https://awiacja.imgw.pl/en/airmet-2/#
- worldwide: https://www.windy.com/39.984/-119.697?500h,39.592,-120.048,8,i:pressure

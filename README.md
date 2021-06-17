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
   conda install numpy matplotlib click configparser scipy pyproj
   pip install orhelper
   ```

4. `diana` needs to find `OpenRocket-15.03.jar`. Call `diana` from the same folder or add the file
   to an environment variable called `CLASSPATH`, see [the oracle documentation](https://docs.oracle.com/javase/tutorial/essential/environment/paths.html).
   Check with 
   ```shell
   echo %CLASSPATH%
   $ PATHTTO\projects\OpenRocket-15.03.jar
   ```


## Usage

So far only one command line tool is in development. It is called `diana` and does dispersion
analyses. Just execute

```shell
diana -h
```

to get help and usage information.


### Procedure

1. Define your project in the .ork file with nominal parameters.
2. At the moment, the very first simulation configuration is used. 
3. Except for the wind data, the following parameters can be used for dispersion analysis
   - Launch Rod Tilt (0 is vertical)
   - Launch Rod Azimuth (0 is north)
   - Thrust Factor (1 is nominal)
   - Thrust increase due to decreased ambient pressure. This is set via the nozzle diameter (set 0 to deactivate this feature).
4. Configure the standard deviation for every supported parameter in the `ini`-file.
5. Run `diana`


### Limitations

- Only single stage rockets are supported
- Check that the simulation option **flat earth** or **WGS84** is set.


### Wind Data

Wind data can be given in an additional text file, containing the following data in three columns separated by whitespace:

- Altitude (in m)
- Direction (in °)
- Wind speed (in m/s)

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

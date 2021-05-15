import ortools.utility as utility

import orhelper
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import click
import configparser
from scipy.interpolate import interp1d
import math

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--directory", "-d", type=click.Path(exists=True),
              default=".", show_default=True,
              help=("Directory in which the .ork and .ini files are located/"
                    "searched for."))
@click.option("--filename", "-f", type=click.Path(exists=True),
              help=("Name of the OpenRocket simulation (.ork) file. If not "
                    "given, the latest .ork file in `directory` is used."))
@click.option("--config", "-c", type=click.Path(exists=True),
              help=("Name of the config (.ini) file. If not given, the "
                    "latest .ini file in `directory` is used."))
@click.option("--output", "-o", type=click.Path(exists=False),
              help="Name of the file the output is saved to.")
@click.option("--show", "-s", is_flag=True, default=False,
              help="Show the results on screen.")
def diana(directory, filename, config, output, show):
    """Do a dispersion analysis of an OpenRocket simulation.

    A dispersion analysis runs multiple simulations with slightly
    varying parameters. The config file specifies which simulation
    parameters are varied and by how much as well as the total number of
    simulations run.

    Example usage:

        diana -d examples -o test.pdf -s
    """
    ork_file_path = filename or utility.find_latest_file(".ork", directory)
    config_file_path = config or utility.find_latest_file(".ini", directory)
    output_filename = output or "dispersion_analysis.pdf"
    results_are_shown = show
    print("directory      : {}".format(directory))
    print(".ork file      : {}".format(ork_file_path))
    print("config file    : {}".format(config_file_path))
    print("output file    : {}".format(output_filename))
    print("output is shown: {}".format(results_are_shown))
    config = configparser.ConfigParser()
    config.read(config_file_path)
    print("config sections: {}".format(config.sections()))
    print("")

    with orhelper.OpenRocketInstance() as instance:
        orh = orhelper.Helper(instance)
        sim = orh.load_doc(ork_file_path).getSimulation(0)

        elevation, azimuth = get_simulation_parameters(sim, config)

        _landing_points = []
        _launch_points = []
        n_simulations = int(config["General"]["NumberOfSimulations"])
        
        for i in range(n_simulations):
            print("Running simulation {:4} of {}".format(i+1, n_simulations))
            randomize_simulation_parameters(sim, config, elevation, azimuth)
            run_simulation(orh, sim, _launch_points, _landing_points)
            #print(_landing_points)

        print_stats(_launch_points, _landing_points)

    # Plot stuff, show and store it
    #create_plots(results, output_filename, results_are_shown)


def get_simulation_parameters(sim, config):
    """Setup the simulation parameters according to the given config."""
    options = sim.getOptions()
    elevation = math.degrees(options.getLaunchRodAngle())
    azimuth = math.degrees(options.getLaunchRodDirection())
    print("Initial launch rail elevation = {:6.2f}°".format(elevation))
    print("Initial launch rail azimuth   = {:6.2f}°".format(azimuth))
    return elevation, azimuth

def randomize_simulation_parameters(sim, config, elevation, azimuth):
    """Draw random samples for the simulation parameters."""
    
    dev_azimuth = float(config["LaunchRail"]["Azimuth"])
    dev_elevation = float(config["LaunchRail"]["Elevation"])
    #print("Standard deviation(elevation) = {:6.2f}°".format(dev_elevation))
    #print("Standard deviation(azimuth)   = {:6.2f}°".format(dev_azimuth))
    
    _rng = np.random.default_rng()
    options = sim.getOptions()
    options.setLaunchRodAngle(math.radians(
        _rng.normal(elevation, dev_elevation)))
    options.setLaunchIntoWind(False) # otherwise launch rod direction cannot be altered
    options.setLaunchRodDirection(math.radians(
        _rng.normal(azimuth, dev_azimuth)))

    elevation = math.degrees(options.getLaunchRodAngle())
    azimuth = math.degrees(options.getLaunchRodDirection())
    print("Used launch rail elevation = {:6.2f}°".format(elevation))
    print("Used launch rail azimuth   = {:6.2f}°".format(azimuth))

def run_simulation(orh, sim, _launch_points, _landing_points):
    """Run a single simulation and return the results."""
    wind_listener = WindListener()
    landing_point_listener = LandingPointListener(_launch_points, _landing_points)
    orh.run_simulation(sim, listeners=(wind_listener, landing_point_listener))
    
def create_plots(results, output_filename, results_are_shown=False):
    """Create, store and optionally show the plots of the results."""
    raise NotImplementedError
    
def print_stats(launch_points, landing_points):
    #print(launch_points)
    #print(landing_points)
    _distances = []
    _bearings = []
        
    launch_point = launch_points[0] # launch point is the same for every simulation
    for landing_point in landing_points:
        print(launch_point)
        print(landing_point)
        _distance, _bearing = compute_distance_and_bearing(launch_point, landing_point)
        _distances.append(_distance)
        _bearings.append(_bearing)
        
    print(
        "Rocket landing zone {:.1f}m ± {:.1f}m ".format(
            np.mean(_distances), np.std(_distances))
        + "bearing {:3.2f}° ± {:3.4f}° ".format(
            np.degrees(np.mean(_bearings)),
            np.degrees(np.std(_bearings)))
        + "from launch site. Based on {} simulations.".format(
            len(landing_points)))

class LandingPointListener(orhelper.AbstractSimulationListener):
    def __init__(self, launch_point, landing_point):
        self.launch_points = launch_point
        self.landing_points = landing_point

    def endSimulation(self, status, simulation_exception):
        landing_position = status.getRocketWorldPosition()
        
        conditions = status.getSimulationConditions()
        launchpos = conditions.getLaunchSite()

        geodetic_computation = conditions.getGeodeticComputation()

        if geodetic_computation != geodetic_computation.FLAT:
            raise ValueError("GeodeticComputationStrategy type not supported!")

        self.launch_points.append(launchpos)
        #print(launchpos)
        self.landing_points.append(landing_position)
        #print(landing_position)


def compute_distance_and_bearing(start, end):
    dx = ((end.getLongitudeDeg() - start.getLongitudeDeg())
          * METERS_PER_DEGREE_LONGITUDE_EQUATOR)
    dy = ((end.getLatitudeDeg() - start.getLatitudeDeg())
          * METERS_PER_DEGREE_LATITUDE)
    distance = math.sqrt(dx * dx + dy * dy)
    bearing = math.pi / 2. - math.atan(dy / dx)
    return distance, bearing


METERS_PER_DEGREE_LATITUDE = 111325
METERS_PER_DEGREE_LONGITUDE_EQUATOR = 111050


class WindListener(orhelper.AbstractSimulationListener):
    # Set the wind speed as a function of altitude
    
    def postWindModel(self, status, wind):

        # list of aloft data
        # see https://en.wikipedia.org/wiki/Winds_aloft
        # TODO: input from external file
        _wind_altitude_m = [0,914,1829,2743,3658,5486,7315,9144,10363,11887,13716,16154]
        _wind_direction_degree = [10, 0,0,310,330,340,260,250,240,250,260,260]
        _wind_speed_ms = [10,0,0,3,3,3,9,15,15,15,14,10]
        if len(_wind_altitude_m) != len(_wind_direction_degree) or len(_wind_altitude_m) != len(_wind_speed_ms):
            raise ValueError("Aloft data incorrect!")

        # interpolate at current altitude
        # print(wind)
        position = status.getRocketPosition()
        #print("Altitude ", position.z)
        # TODO: which fill_values shall be used above the aloft data? zero? last value? extrapolate?
        f_wind_speed_ms = interp1d(_wind_altitude_m, _wind_speed_ms, bounds_error = False,
                                   fill_value=(_wind_speed_ms[0], _wind_speed_ms[-1]))
        wind_speed_ms = f_wind_speed_ms(position.z)
        f_wind_direction_degree = interp1d(_wind_altitude_m, _wind_direction_degree, bounds_error = False,
                                           fill_value=(_wind_direction_degree[0], _wind_direction_degree[-1]))
        _wind_direction_degree = f_wind_direction_degree(position.z)
        # give the wind in NE coordinates
        # (where it is going, not where it is coming from)
        v_east = -math.sin(math.radians(_wind_direction_degree))*wind_speed_ms;
        v_north = -math.cos(math.radians(_wind_direction_degree))*wind_speed_ms;
        wind = wind.setX(v_east)
        wind = wind.setY(v_north)
        #print(wind)
        return wind


if __name__ == "__main__":
    diana()

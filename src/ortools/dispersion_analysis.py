import ortools.utility as utility

import orhelper
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import click
import configparser
import scipy.interpolate

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

        landing_points = []
        launch_points = []
        n_simulations = int(config["General"]["NumberOfSimulations"])

        for i in range(n_simulations):
            print("Running simulation {:4} of {}".format(i + 1, n_simulations))
            randomize_simulation_parameters(sim, config, elevation, azimuth)
            run_simulation(orh, sim, launch_points, landing_points)
            # print(_landing_points)

        print_stats(launch_points, landing_points)


def get_simulation_parameters(sim, config):
    """Return launch rail elevation and azimuth angle."""
    options = sim.getOptions()
    elevation = math.degrees(options.getLaunchRodAngle())
    azimuth = math.degrees(options.getLaunchRodDirection())
    print("Initial launch rail elevation = {:6.2f}°".format(elevation))
    print("Initial launch rail azimuth   = {:6.2f}°".format(azimuth))
    return elevation, azimuth


def randomize_simulation_parameters(sim, config, elevation, azimuth):
    """Draw random samples for the simulation parameters."""

    azimuth_dev = float(config["LaunchRail"]["Azimuth"])
    elevation_dev = float(config["LaunchRail"]["Elevation"])
    #print("Standard deviation(elevation) = {:6.2f}°".format(elevation_dev))
    #print("Standard deviation(azimuth)   = {:6.2f}°".format(azimuth_dev))

    rng = np.random.default_rng()
    options = sim.getOptions()
    options.setLaunchRodAngle(
        math.radians(rng.normal(elevation, elevation_dev)))
    # otherwise launch rod direction cannot be altered
    options.setLaunchIntoWind(False)
    options.setLaunchRodDirection(
        math.radians(rng.normal(azimuth, azimuth_dev)))

    elevation = math.degrees(options.getLaunchRodAngle())
    azimuth = math.degrees(options.getLaunchRodDirection())
    print("Used launch rail elevation = {:6.2f}°".format(elevation))
    print("Used launch rail azimuth   = {:6.2f}°".format(azimuth))


def run_simulation(orh, sim, launch_points, landing_points):
    """Run a single simulation and return the results."""
    wind_listener = WindListener()
    landing_point_listener = LandingPointListener(
        launch_points, landing_points)
    orh.run_simulation(sim, listeners=(wind_listener, landing_point_listener))


def create_plots(results, output_filename, results_are_shown=False):
    """Create, store and optionally show the plots of the results."""
    raise NotImplementedError


def print_stats(launch_points, landing_points):
    # print(launch_points)
    # print(landing_points)
    distances = []
    bearings = []

    # launch point is the same for every simulation
    launch_point = launch_points[0]
    for landing_point in landing_points:
        print(launch_point)
        print(landing_point)
        distance, bearing = compute_distance_and_bearing(
            launch_point, landing_point)
        distances.append(distance)
        bearings.append(bearing)

    print(
        "Rocket landing zone {:.1f}m ± {:.1f}m ".format(
            np.mean(distances), np.std(distances))
        + "bearing {:3.2f}° ± {:3.4f}° ".format(
            np.degrees(np.mean(bearings)),
            np.degrees(np.std(bearings)))
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
        # print(launchpos)
        self.landing_points.append(landing_position)
        # print(landing_position)


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
        altitudes_m = [0, 914, 1829, 2743, 3658, 5486, 7315, 9144, 10363,
                       11887, 13716, 16154]
        wind_directions_degree = [10, 0, 0, 310, 330, 340, 260, 250, 240, 250,
                                  260, 260]
        wind_speeds_mps = [10, 0, 0, 3, 3, 3, 9, 15, 15, 15, 14, 10]
        if (len(altitudes_m) != len(wind_directions_degree)
                or len(altitudes_m) != len(wind_speeds_mps)):
            raise ValueError(
                "Aloft data is incorrect! `altitudes_m`, "
                + "`wind_directions_degree` and `wind_speeds_mps` must be of "
                + "the same length.")

        # interpolate at current altitude
        # print(wind)
        position = status.getRocketPosition()
        #print("Altitude ", position.z)
        # TODO: which fill_values shall be used above the aloft data? zero?
        # last value? extrapolate?
        interpolate_wind_speed_mps = scipy.interpolate.interp1d(
            altitudes_m, wind_speeds_mps, bounds_error=False,
            fill_value=(wind_speeds_mps[0], wind_speeds_mps[-1]))
        wind_speed_mps = interpolate_wind_speed_mps(position.z)
        interpolate_wind_direction_degree = scipy.interpolate.interp1d(
            altitudes_m, wind_directions_degree, bounds_error=False,
            fill_value=(wind_directions_degree[0], wind_directions_degree[-1]))
        wind_direction_degree = interpolate_wind_direction_degree(position.z)
        # give the wind in NE coordinates
        # (where it is going, not where it is coming from)
        v_north, v_east = -utility.polar_to_cartesian(wind_speed_mps,
                                                      wind_direction_degree)
        wind = wind.setX(v_east)
        wind = wind.setY(v_north)
        print(wind)
        return wind


if __name__ == "__main__":
    diana()

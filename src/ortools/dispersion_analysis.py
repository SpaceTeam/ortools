import ortools.utility as utility

import orhelper
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import click
import configparser
import scipy.interpolate

import math
import collections
import dataclasses

import logging, sys

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
    config = configparser.ConfigParser(converters={'list': lambda x: [float(i.strip()) for i in x.split(',')]})    
    config.read(config_file_path)
    print("config sections: {}".format(config.sections()))
    print("")

    # setup of  logging on stderr.
    # use logging.WARNING, or logging.DEBUG if necessary   
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    with orhelper.OpenRocketInstance() as instance:
        orh = orhelper.Helper(instance)
        sim = orh.load_doc(ork_file_path).getSimulation(0)

        rng = np.random.default_rng()
        random_parameters = set_up_random_parameters(sim, config, rng)

        landing_points = []
        n_simulations = int(config["General"]["NumberOfSimulations"])

        for i in range(n_simulations):
            print("Running simulation {:4} of {}".format(i + 1, n_simulations))
            randomize_simulation(sim, random_parameters)
            landing_point, launch_point = run_simulation(orh, sim, config)
            landing_points.append(landing_point)

        print_stats(launch_point, landing_points)


def set_up_random_parameters(sim, config, rng):
    """Return a ``namedtuple`` containing all random parameters.

    The random parameters are actually lambdas that return a new random
    sample when called.
    """
    options = sim.getOptions()
    azimuth_mean = options.getLaunchRodDirection()
    azimuth_stddev = math.radians(float(config["LaunchRail"]["Azimuth"]))
    elevation_mean = options.getLaunchRodAngle()
    elevation_stddev = math.radians(float(config["LaunchRail"]["Elevation"]))

    print("Initial launch rail elevation = {:6.2f}°".format(
        math.degrees(elevation_mean)))
    print("Initial launch rail azimuth   = {:6.2f}°".format(
        math.degrees(azimuth_mean)))

    RandomParameters = collections.namedtuple("RandomParameters", [
        "elevation",
        "azimuth"])
    return RandomParameters(
        elevation=lambda: rng.normal(elevation_mean, elevation_stddev),
        azimuth=lambda: rng.normal(azimuth_mean, azimuth_stddev))


def randomize_simulation(sim, random_parameters):
    """Set simulation parameters to random samples."""
    options = sim.getOptions()
    options.setLaunchRodAngle(random_parameters.elevation())
    # otherwise launch rod direction cannot be altered
    options.setLaunchIntoWind(False)
    options.setLaunchRodDirection(random_parameters.azimuth())

    elevation = math.degrees(options.getLaunchRodAngle())
    azimuth = math.degrees(options.getLaunchRodDirection())
    print("Used launch rail elevation = {:6.2f}°".format(elevation))
    print("Used launch rail azimuth   = {:6.2f}°".format(azimuth))


def run_simulation(orh, sim, config):
    """Run a single simulation and return the results.

    :return:
        A 2-tuple containing the landing and launch position
    """
    wind_listener = WindListener(config)
    landing_point_listener = LandingPointListener()
    orh.run_simulation(sim, listeners=(landing_point_listener, wind_listener))
    return (landing_point_listener.landing_points[0],
            landing_point_listener.launch_points[0])


class LandingPointListener(orhelper.AbstractSimulationListener):
    def __init__(self):
        # FIXME: This is a weird workaround because I don't know how to
        # create the member variables of the correct type.
        self.landing_points = []
        self.launch_points = []

    def endSimulation(self, status, simulation_exception):
        self.landing_points.append(status.getRocketWorldPosition())
        conditions = status.getSimulationConditions()
        self.launch_points.append(conditions.getLaunchSite())
        geodetic_computation = conditions.getGeodeticComputation()

        if geodetic_computation != geodetic_computation.FLAT:
            raise ValueError("GeodeticComputationStrategy type not supported!")


class WindListener(orhelper.AbstractSimulationListener):
    """Set the wind speed as a function of altitude."""

    def __init__(self, config):
        # read wind level model data from file
        altitudes_m = config["WindModel"].getlist("Altitude")
        wind_directions_degree = config["WindModel"].getlist("WindDirection")
        wind_speeds_mps = config["WindModel"].getlist("WindSpeed")
        
        logging.debug('Input wind levels model data:')     
        logging.debug('Altitude (m) ')
        logging.debug(altitudes_m)
        logging.debug("Direction (°) ")
        logging.debug(wind_directions_degree)        
        logging.debug("Wind speed (m/s) ")
        logging.debug(wind_speeds_mps)
        
        wind_directions_rad = np.radians(wind_directions_degree)
        if (len(altitudes_m) != len(wind_directions_degree)
                or len(altitudes_m) != len(wind_speeds_mps)):
            raise ValueError(
                "Aloft data is incorrect! `altitudes_m`, "
                + "`wind_directions_degree` and `wind_speeds_mps` must be of "
                + "the same length.")

        # TODO: which fill_values shall be used above the aloft data? zero?
        # last value? extrapolate?
        # TODO: this i not safe if direction rotates over 180deg -> use x/y coordinates
        self.interpolate_wind_speed_mps = scipy.interpolate.interp1d(
            altitudes_m, wind_speeds_mps, bounds_error=False,
            fill_value=(wind_speeds_mps[0], wind_speeds_mps[-1]))
        self.interpolate_wind_direction_rad = scipy.interpolate.interp1d(
            altitudes_m, wind_directions_rad, bounds_error=False,
            fill_value=(wind_directions_rad[0], wind_directions_rad[-1]))

    def postWindModel(self, status, wind):
        position = status.getRocketPosition()
        wind_speed_mps = self.interpolate_wind_speed_mps(position.z)
        wind_direction_rad = self.interpolate_wind_direction_rad(position.z)
        # give the wind in NE coordinates 
        v_north, v_east = utility.polar_to_cartesian(wind_speed_mps,
                                                     wind_direction_rad)
        wind = wind.setY(v_north)
        wind = wind.setX(v_east)
        return wind


def create_plots(results, output_filename, results_are_shown=False):
    """Create, store and optionally show the plots of the results."""
    raise NotImplementedError


def print_stats(launch_point, landing_points):
    distances = []
    bearings = []

    logging.debug('Results: distances in cartesian coordinates')
    
    for landing_point in landing_points:
        distance, bearing = compute_distance_and_bearing(
            launch_point, landing_point)
        distances.append(distance)
        bearings.append(bearing)

    logging.debug('distances and bearings in polar coordinates')
    logging.debug(distances)
    logging.debug(bearings)

    print(
        "Rocket landing zone {:.1f}m ± {:.2f}m ".format(
            np.mean(distances), np.std(distances))
        + "bearing {:.1f}° ± {:.1f}° ".format(
            np.degrees(np.mean(bearings)), np.degrees(np.std(bearings)))
        + "from launch site. Based on {} simulations.".format(
            len(landing_points)))


def compute_distance_and_bearing(start, end):
    dx = ((end.getLongitudeDeg() - start.getLongitudeDeg())
          * METERS_PER_DEGREE_LONGITUDE_EQUATOR)
    dy = ((end.getLatitudeDeg() - start.getLatitudeDeg())
          * METERS_PER_DEGREE_LATITUDE)
    logging.debug('Longitude {:.1f}°, Latitude {:.1f}°'.format(end.getLongitudeDeg(), end.getLatitudeDeg()))
    logging.debug('dx {:.1f}m, dy {:.1f}m'.format(dx, dy))
    distance = math.sqrt(dx * dx + dy * dy)
    bearing = math.pi / 2. - math.atan2(dy, dx)
    if bearing > math.pi:
        bearing = bearing - 2*math.pi
    return distance, bearing


METERS_PER_DEGREE_LATITUDE = 111325
METERS_PER_DEGREE_LONGITUDE_EQUATOR = 111050


if __name__ == "__main__":
    diana()

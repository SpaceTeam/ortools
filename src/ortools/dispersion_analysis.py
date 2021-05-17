import ortools.utility as utility

import orhelper
from orhelper import FlightDataType, FlightEvent

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import click
import configparser
import scipy.interpolate
import pyproj

import os
import sys
import math
import collections
import logging


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
    # TODO: Define default values for all parameters of the .ini fileata file)
    config.read(config_file_path)
    make_paths_in_config_absolute(config, config_file_path)
    print("config sections: {}".format(config.sections()))
    print("")

    # Setup of logging on stderr
    # Use logging.WARNING, or logging.DEBUG if necessary
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    with orhelper.OpenRocketInstance() as instance:
        idx_simulation = int(config["General"]["SimulationIndex"])

        orh = orhelper.Helper(instance)
        doc = orh.load_doc(ork_file_path)
        simulation_count = doc.getSimulationCount()

        if idx_simulation < 0 or idx_simulation > (simulation_count - 1):
            raise ValueError("Wrong value of SimulationIndex!")
        sim = doc.getSimulation(idx_simulation)

        print("Load simulation number {} called {}.".format(
            idx_simulation, sim.getName()))

        rng = np.random.default_rng()
        random_parameters = set_up_random_parameters(sim, config, rng)

        landing_points = []
        apogee_points = []
        n_simulations = int(config["General"]["NumberOfSimulations"])

        for i in range(n_simulations):
            print("Running simulation {:4} of {}".format(i + 1, n_simulations))
            randomize_simulation(sim, random_parameters)
            landing_point, launch_point, geodetic_computation, apogee = run_simulation(
                orh, sim, config)
            landing_points.append(landing_point)
            apogee_points.append(apogee)

        print_stats(
            launch_point,
            landing_points,
            apogee_points,
            geodetic_computation)


def make_paths_in_config_absolute(config, config_file_path):
    """Turn all paths in the diana config file into absolute ones."""
    directory = os.path.dirname(os.path.abspath(config_file_path))
    config["WindModel"]["DataFile"] = os.path.join(
        directory, config["WindModel"]["DataFile"])


def set_up_random_parameters(sim, config, rng):
    """Return a ``namedtuple`` containing all random parameters.

    The random parameters are actually lambdas that return a new random
    sample when called.
    """
    options = sim.getOptions()
    azimuth_mean = options.getLaunchRodDirection()
    azimuth_stddev = math.radians(float(config["LaunchRail"]["Azimuth"]))
    tilt_mean = options.getLaunchRodAngle()
    tilt_stddev = math.radians(float(config["LaunchRail"]["Tilt"]))

    print("Initial launch rail tilt = {:6.2f}°".format(
        math.degrees(tilt_mean)))
    print("Initial launch rail azimuth   = {:6.2f}°".format(
        math.degrees(azimuth_mean)))

    RandomParameters = collections.namedtuple("RandomParameters", [
        "tilt",
        "azimuth"])
    return RandomParameters(
        tilt=lambda: rng.normal(tilt_mean, tilt_stddev),
        azimuth=lambda: rng.normal(azimuth_mean, azimuth_stddev))


def randomize_simulation(sim, random_parameters):
    """Set simulation parameters to random samples."""
    options = sim.getOptions()
    options.setLaunchRodAngle(random_parameters.tilt())
    # Otherwise launch rod direction cannot be altered
    options.setLaunchIntoWind(False)
    options.setLaunchRodDirection(random_parameters.azimuth())

    tilt = math.degrees(options.getLaunchRodAngle())
    azimuth = math.degrees(options.getLaunchRodDirection())
    print("Used launch rail tilt = {:6.2f}°".format(tilt))
    print("Used launch rail azimuth   = {:6.2f}°".format(azimuth))


def run_simulation(orh, sim, config):
    """Run a single simulation and return the results.

    :return:
        A 2-tuple containing the landing and launch position
    """
    wind_listener = WindListener(config["WindModel"]["DataFile"])
    launch_point_listener = LaunchPointListener()
    landing_point_listener = LandingPointListener()
    orh.run_simulation(
        sim,
        listeners=(
            launch_point_listener,
            landing_point_listener,
            wind_listener))

    # process results
    events = orh.get_events(sim)
    data = orh.get_timeseries(sim, [
        FlightDataType.TYPE_TIME,
        FlightDataType.TYPE_ALTITUDE,
        FlightDataType.TYPE_LONGITUDE,
        FlightDataType.TYPE_LATITUDE,
    ])
    def index_at(t): return (
        np.abs(data[FlightDataType.TYPE_TIME] - t)).argmin()

    for event, times in events.items():
        if event is FlightEvent.APOGEE:
            for time in times:
                apogee = orh.openrocket.util.WorldCoordinate(
                    math.degrees(
                        data[FlightDataType.TYPE_LATITUDE][index_at(time)]),
                    math.degrees(
                        data[FlightDataType.TYPE_LONGITUDE][index_at(time)]),
                    data[FlightDataType.TYPE_ALTITUDE][index_at(time)])
                logging.debug("Apogee at {:.1f}s: longitude {:.1f}°, latitude,{:.1f}°, altitude {:.1f}m".format(
                    time,
                    apogee.getLatitudeDeg(),
                    apogee.getLongitudeDeg(),
                    apogee.getAltitude()))
    return (landing_point_listener.landing_points[0],
            launch_point_listener.launch_point,
            launch_point_listener.geodetic_computation,
            apogee)


class LaunchPointListener(orhelper.AbstractSimulationListener):
    """Return the launch point at the startSimulation callback."""

    def __init__(self):
        # FIXME: This is a weird workaround because I don't know how to
        # create the member variables of the correct type.
        self.launch_point = None
        self.geodetic_computation = None

    def startSimulation(self, status):
        """Analyze the simulation conditions of openrocket.

        These are the launch point and if a supported
        geodetic model is set."""
        conditions = status.getSimulationConditions()
        self.launch_point = conditions.getLaunchSite()

        self.geodetic_computation = conditions.getGeodeticComputation()
        logging.debug(self.geodetic_computation)
        if (self.geodetic_computation != self.geodetic_computation.FLAT and
                self.geodetic_computation != self.geodetic_computation.WGS84):
            raise ValueError("GeodeticComputationStrategy type not supported!")


class LandingPointListener(orhelper.AbstractSimulationListener):
    """Return the landing point at the endSimulation callback."""

    def __init__(self):
        # FIXME: This is a weird workaround because I don't know how to
        # create the member variables of the correct type.
        self.landing_points = []
        self.launch_points = []

    def endSimulation(self, status, simulation_exception):
        """Return the landing position from openrocket."""
        self.landing_points.append(status.getRocketWorldPosition())


class WindListener(orhelper.AbstractSimulationListener):
    """Set the wind speed as a function of altitude."""

    def __init__(self, wind_model_file=""):
        """Read wind level model data from file.

        Save them as interpolation functions to be used in other callbacks
        of this class.
        """
        try:
            # Read wind level model data from file
            data = np.loadtxt(wind_model_file)
        except (IOError, FileNotFoundError):
            self._default_wind_model_is_used = True
            print("Warning: wind model file '{}' ".format(wind_model_file)
                  + "not found! Default wind model will be used.")
            return

        self._default_wind_model_is_used = False
        altitudes_m = data[:, 0]
        wind_directions_degree = data[:, 1]
        wind_speeds_mps = data[:, 2]

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
                + "`wind_directions_degree` and `wind_speeds_mps` must be "
                + "of the same length.")

        # TODO: Which fill_values shall be used above the aloft
        # data? zero? last value? extrapolate?
        # TODO: This is not safe if direction rotates over 180deg ->
        # use x/y coordinates
        self.interpolate_wind_speed_mps = scipy.interpolate.interp1d(
            altitudes_m, wind_speeds_mps, bounds_error=False,
            fill_value=(wind_speeds_mps[0], wind_speeds_mps[-1]))
        self.interpolate_wind_direction_rad = scipy.interpolate.interp1d(
            altitudes_m, wind_directions_rad, bounds_error=False,
            fill_value=(wind_directions_rad[0], wind_directions_rad[-1]))

    def postWindModel(self, status, wind):
        """Set the wind coordinates at every simulation step."""
        if self._default_wind_model_is_used:
            return wind
        else:
            position = status.getRocketPosition()
            wind_speed_mps = self.interpolate_wind_speed_mps(position.z)
            wind_direction_rad = self.interpolate_wind_direction_rad(
                position.z)
            # Give the wind in NE coordinates
            v_north, v_east = utility.polar_to_cartesian(wind_speed_mps,
                                                         wind_direction_rad)
            wind = wind.setY(v_north)
            wind = wind.setX(v_east)
            return wind


def create_plots(results, output_filename, results_are_shown=False):
    """Create, store and optionally show the plots of the results."""
    raise NotImplementedError


def print_stats(launch_point, landing_points,
                apogee_points, geodetic_computation):
    """Print statistics of all simulations."""
    distances = []
    bearings = []
    max_altitude = []

    logging.debug('Results: distances in cartesian coordinates')
    for landing_point in landing_points:

        if geodetic_computation == geodetic_computation.FLAT:
            distance, bearing = compute_distance_and_bearing_flat(
                launch_point, landing_point)
        elif geodetic_computation == geodetic_computation.WGS84:
            geodesic = pyproj.Geod(ellps='WGS84')
            fwd_azimuth, back_azimuth, distance = geodesic.inv(
                launch_point.getLongitudeDeg(),
                launch_point.getLatitudeDeg(),
                landing_point.getLongitudeDeg(),
                landing_point.getLatitudeDeg())
            bearing = np.radians(fwd_azimuth)

        distances.append(distance)
        bearings.append(bearing)

    for apogee in apogee_points:
        max_altitude.append(apogee.getAltitude())

    logging.debug('distances and bearings in polar coordinates')
    logging.debug(distances)
    logging.debug(bearings)

    print("Apogee: {:.1f}m ± {:.2f}m ".format(
        np.mean(max_altitude), np.std(max_altitude)))
    print(
        "Rocket landing zone {:.1f}m ± {:.2f}m ".format(
            np.mean(distances), np.std(distances))
        + "bearing {:.1f}° ± {:.1f}° ".format(
            np.degrees(np.mean(bearings)), np.degrees(np.std(bearings))))
    print("Based on {} simulations.".format(
        len(landing_points)))


def compute_distance_and_bearing_flat(start, end):
    """Return distance and bearing betweeen two points.

    valid for flat earth approximation only.

    Arguments:
    start, end --  two points of Coordinate class

    Return:
    distance -- in m
    bearing -- in degree
    """
    dx = ((end.getLongitudeDeg() - start.getLongitudeDeg())
          * METERS_PER_DEGREE_LONGITUDE_EQUATOR)
    dy = ((end.getLatitudeDeg() - start.getLatitudeDeg())
          * METERS_PER_DEGREE_LATITUDE)
    logging.debug("Longitude {:.1f}°, Latitude {:.1f}°".format(
        end.getLongitudeDeg(), end.getLatitudeDeg()))
    logging.debug("dx {:.1f}m, dy {:.1f}m".format(dx, dy))
    distance = math.sqrt(dx * dx + dy * dy)
    bearing = math.pi / 2. - math.atan2(dy, dx)
    if bearing > math.pi:
        bearing = bearing - 2 * math.pi
    return distance, bearing


METERS_PER_DEGREE_LATITUDE = 111325
METERS_PER_DEGREE_LONGITUDE_EQUATOR = 111050


if __name__ == "__main__":
    diana()

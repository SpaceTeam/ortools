import ortools.utility as utility

import orhelper
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import click
import configparser

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
              help="Show the output on screen.")
def cli(directory, filename, config, output, show):
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
    output_is_shown = show
    print("directory      : {}".format(directory))
    print(".ork file      : {}".format(ork_file_path))
    print("config file    : {}".format(config_file_path))
    print("output file    : {}".format(output_filename))
    print("output is shown: {}".format(output_is_shown))
    config = configparser.ConfigParser()
    config.read(config_file_path)
    print("config sections: {}".format(config.sections()))
    print("")

    # Add landing point class, and run simulations
    dispersion_analysis = DispersionAnalysis()

    n_simulations = int(config["General"]["NumberOfSimulations"])
    print("Start {} simulation(s).".format(n_simulations))
    dispersion_analysis.set_ork_file_path(ork_file_path)
    dispersion_analysis.set_launch_rail_deviation(
        float(config["LaunchRail"]["Azimuth"]),
        float(config["LaunchRail"]["Elevation"]))
    dispersion_analysis.run_simulations(n_simulations)

    # Statistics
    if output_is_shown:
        dispersion_analysis.print_stats()


class DispersionAnalysis():
    """A class running many simulations to do a dispersion analysis."""

    def __init__(self):
        self._distances = []
        self._bearings = []
        self._landing_points = []
        self._rng = np.random.default_rng()
        self._ork_file_path = None
        self._dev_azimuth = None
        self._dev_elevation = None
    
    def set_ork_file_path(self, ork_file_path):
        self._ork_file_path = ork_file_path

    def set_launch_rail_deviation(self, dev_azimuth, dev_elevation):
        self._dev_azimuth = dev_azimuth
        self._dev_elevation = dev_elevation
        print("Standard deviation(elevation) = {:6.2f}°".format(dev_elevation))
        print("Standard deviation(azimuth)   = {:6.2f}°".format(dev_azimuth))

    def run_simulations(self, n_simulations):
        with orhelper.OpenRocketInstance() as instance:

            # Load the document and get simulation
            orh = orhelper.Helper(instance)
            doc = orh.load_doc(self._ork_file_path)
            sim = doc.getSimulation(0)
            print("Loaded Simulation: '{}'".format(sim.getName()))

            options = sim.getOptions()
            rocket = options.getRocket()

            elevation = math.degrees(options.getLaunchRodAngle())
            azimuth = math.degrees(options.getLaunchRodDirection())
            print("Initial launch rail elevation = {:6.2f}°".format(elevation))
            print("Initial launch rail azimuth   = {:6.2f}°".format(azimuth))

            # Run simulations and append landing points
            for p in range(n_simulations):
                print("Running simulation {} of {}".format(p + 1,
                                                           n_simulations))

                options.setLaunchRodAngle(math.radians(
                    self._rng.normal(elevation, self._dev_elevation)))
                options.setLaunchRodDirection(math.radians(
                    self._rng.normal(azimuth, self._dev_azimuth)))

                wind_listener = WindListener()
                landing_point_listener = LandingPointListener(self._distances,
                                                              self._bearings)
                orh.run_simulation(
                    sim, listeners=(wind_listener, landing_point_listener))
                self._landing_points.append(landing_point_listener)

    def print_stats(self):
        print(
            "Rocket landing zone {:.1f}m ± {:.1f}m ".format(
                np.mean(self._distances), np.std(self._distances))
            + "bearing {:3.2f}° ± {:3.4f}° ".format(
                np.degrees(np.mean(self._bearings)),
                np.degrees(np.std(self._bearings)))
            + "from launch site. Based on {} simulations.".format(
                len(self._landing_points)))


class LandingPointListener(orhelper.AbstractSimulationListener):
    def __init__(self, distances, bearings):
        self.distances = distances
        self.bearings = bearings

    def endSimulation(self, status, simulation_exception):
        landing_position = status.getRocketWorldPosition()
        conditions = status.getSimulationConditions()
        launch_position = conditions.getLaunchSite()
        geodetic_computation = conditions.getGeodeticComputation()

        if geodetic_computation != geodetic_computation.FLAT:
            raise ValueError("GeodeticComputationStrategy type not supported!")

        distance, bearing = compute_distance_and_bearing(launch_position,
                                                         landing_position)
        self.distances.append(distance)
        self.bearings.append(bearing)


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
        # print(wind)
        #position = status.getRocketPosition()
        #print("Altitude ", position.z)
        wind = wind.setX(1)
        wind = wind.setY(1)
        # print(wind)
        return wind


if __name__ == "__main__":
    cli()

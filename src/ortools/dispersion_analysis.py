import ortools.utility as utility

import orhelper
import numpy as np
import math
from random import gauss
import matplotlib as mpl
from matplotlib import pyplot as plt
import click
import configparser

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


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

    # add landing point class, and add simulations
    points = LandingPoints()
    
    number_iterations = int(config["General"]["NumberOfIterations"])
    print('Start ', number_iterations, ' simulation(s).')    
    points.set_path(ork_file_path)
    points.set_launch_rail_deviation(float(config["LaunchRail"]["Azimuth"]), float(config["LaunchRail"]["Elevation"]))
    points.add_simulations(number_iterations)

    # statistics
    if output_is_shown:
        points.print_stats()
    
class LandingPoints(list):
    "A list of landing points with ability to run simulations and populate itself"

    def __init__(self):
        self.ranges = []
        self.bearings = []

    def set_path(self, path):
        self.ork_file_path = path

    def set_launch_rail_deviation(self, dev_azimuth, dev_elevation):
        self.dev_azimuth = dev_azimuth;
        self.dev_elevation = dev_elevation;
        print('Use deviation for azimut ', dev_azimuth, ' and elevation ', dev_elevation)

    def add_simulations(self, num):
        with orhelper.OpenRocketInstance() as instance:

            # Load the document and get simulation
            orh = orhelper.Helper(instance)
            doc = orh.load_doc(self.ork_file_path)
            sim = doc.getSimulation(0)
            print('Loaded Simulation: "', sim.getName(), '"')

            # Randomize various parameters
            opts = sim.getOptions()
            rocket = opts.getRocket()
            
            elevation = math.degrees(opts.getLaunchRodAngle())
            azimuth = math.degrees(opts.getLaunchRodDirection())
            print('Initial launch rail elevation ', elevation, ' and azimuth ', azimuth)

            # Run num simulations and add to self
            for p in range(num):
                print('Running simulation ', p+1, ' of ', num)

                opts.setLaunchRodAngle(math.radians(gauss(elevation, self.dev_elevation))) 
                opts.setLaunchRodDirection(math.radians(gauss(azimuth, self.dev_azimuth))) 

                windlistener = WindListen()  # simulation listener
                lp = LandingPoint(self.ranges, self.bearings)
                orh.run_simulation(sim, listeners=(windlistener, lp))
                self.append(lp)

    def print_stats(self):
        print(
            'Rocket landing zone %3.2f m +- %3.2f m bearing %3.2f deg +- %3.4f deg from launch site. Based on %i simulations.' % \
            (np.mean(self.ranges), np.std(self.ranges), np.degrees(np.mean(self.bearings)),
             np.degrees(np.std(self.bearings)), len(self)))
        
class LandingPoint(orhelper.AbstractSimulationListener):
    def __init__(self, ranges, bearings):
        self.ranges = ranges
        self.bearings = bearings

    def endSimulation(self, status, simulation_exception):
        worldpos = status.getRocketWorldPosition()
        conditions = status.getSimulationConditions()
        launchpos = conditions.getLaunchSite()
        geodetic_computation = conditions.getGeodeticComputation()

        if geodetic_computation != geodetic_computation.FLAT:
            raise Exception("GeodeticComputationStrategy type not supported")

        self.ranges.append(range_flat(launchpos, worldpos))
        self.bearings.append(bearing_flat(launchpos, worldpos))

        
class WindListen(orhelper.AbstractSimulationListener):
    # set the wind speed as a function of altitude
    def postWindModel(self, status, wind):
        #print(wind)
        #position = status.getRocketPosition()
        #print('Altitude ', position.z)
        wind = wind.setX(1)
        wind = wind.setY(1)
        #print(wind)
        return wind


METERS_PER_DEGREE_LATITUDE = 111325
METERS_PER_DEGREE_LONGITUDE_EQUATOR = 111050


def range_flat(start, end):
    dy = (end.getLatitudeDeg() - start.getLatitudeDeg()) * METERS_PER_DEGREE_LATITUDE
    dx = (end.getLongitudeDeg() - start.getLongitudeDeg()) * METERS_PER_DEGREE_LONGITUDE_EQUATOR
    return math.sqrt(dy * dy + dx * dx)


def bearing_flat(start, end):
    dy = (end.getLatitudeDeg() - start.getLatitudeDeg()) * METERS_PER_DEGREE_LATITUDE
    dx = (end.getLongitudeDeg() - start.getLongitudeDeg()) * METERS_PER_DEGREE_LONGITUDE_EQUATOR
    return math.pi / 2 - math.atan(dy / dx)

if __name__ == "__main__":
    cli()

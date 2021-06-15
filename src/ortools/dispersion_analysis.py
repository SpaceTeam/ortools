import ortools.utility as utility

import orhelper
import numpy as np
import matplotlib as mpl
import matplotlib.patches
import matplotlib.gridspec
import matplotlib.transforms
from matplotlib import pyplot as plt
import click
import configparser
import scipy.interpolate
import pyproj
import cycler

import os
import sys
import math
import time
import logging
import collections


plt.style.use("default")
mpl.rcParams["figure.figsize"] = ((1920 - 160) / 5 / 25.4,
                                  (1080 - 90) / 5 / 25.4)
mpl.rcParams["figure.dpi"] = 254 / 2
mpl.rcParams["axes.unicode_minus"] = True
mpl.rcParams["axes.grid"] = True
# Print friendly, colorblind safe colors for qualitative data
# Source: https://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=3
mpl.rcParams["axes.prop_cycle"] = cycler.cycler(
    color=("#7570b3", "#d95f02", "#1b9e77"), linestyle=("-", "--", ":"))
line_color_map = mpl.cm.gist_rainbow

PLOTS_ARE_TESTED = False

STANDARD_PRESSURE = 101325.0  # The standard air pressure (1.01325 bar)
METERS_PER_DEGREE_LATITUDE = 111325.0
METERS_PER_DEGREE_LONGITUDE_EQUATOR = 111050.0

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
@click.option("--plot-coordinate-type", "-ct",
              type=click.Choice(["flat", "wgs84"]),
              default="flat", show_default=True,
              help=("The type of coordinates used in the scatter plot of the "
                    "landing points."))
@click.option("--show", "-s", is_flag=True, default=False,
              help="Show the results on screen.")
def diana(directory, filename, config, output, plot_coordinate_type, show):
    """Do a dispersion analysis of an OpenRocket simulation.

    A dispersion analysis runs multiple simulations with slightly
    varying parameters. The config file specifies which simulation
    parameters are varied and by how much as well as the total number of
    simulations run.

    Example usage:

        diana -d examples -o test.pdf -s
    """
    # TODO: Maybe put .ork file path in config file
    t0 = time.time()
    ork_file_path = filename or utility.find_latest_file(".ork", directory)
    config_file_path = config or utility.find_latest_file(".ini", directory)
    timestamp = str(int(time.time()))
    output_filename = output or "dispersion_analysis_" + timestamp + ".pdf"
    results_are_shown = show
    print("directory   : {}".format(directory))
    print(".ork file   : {}".format(ork_file_path))
    print("config file : {}".format(config_file_path))
    print("output file : {}".format(output_filename))
    config = configparser.ConfigParser()
    # TODO: Define default values for all parameters of the .ini fileata file)
    config.read(config_file_path)
    make_paths_in_config_absolute(config, config_file_path)

    # Setup of logging on stderr
    # Use logging.WARNING, or logging.DEBUG if necessary
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    if PLOTS_ARE_TESTED:
        create_plots(
            [],
            output_filename,
            plot_coordinate_type,
            results_are_shown)
        return

    with orhelper.OpenRocketInstance() as instance:
        t1 = time.time()
        orh = orhelper.Helper(instance)
        i_simulation = int(config["General"]["SimulationIndex"])
        sim = get_simulation(orh, ork_file_path, i_simulation)
        rocket_components, original_parameters, random_parameters = \
            set_up_random_parameters(orh, sim, config)

        results = []
        n_simulations = int(config["General"]["NumberOfSimulations"])
        for i in range(n_simulations):
            print("-- Running simulation {:4} of {} --".format(
                i + 1, n_simulations))
            if i > 0:
                randomize_simulation(orh, sim, rocket_components,
                                     original_parameters, random_parameters)
            else:
                print("with nominal parameter set but wind-model applied")
            result = run_simulation(orh, sim, config, random_parameters)
            results.append(result)
        t2 = time.time()
        print_statistics(results)
        t3 = time.time()
        print("---")
        print("time for {} simulations = {:.1f}s".format(n_simulations,
                                                         t2 - t1))
        print("total execution time = {:.1f}s".format(t3 - t0))
        create_plots(results, output_filename, plot_coordinate_type,
                     results_are_shown)


def make_paths_in_config_absolute(config, config_file_path):
    """Turn all paths in the diana config file into absolute ones."""
    directory = os.path.dirname(os.path.abspath(config_file_path))
    config["WindModel"]["DataFile"] = os.path.join(
        directory, config["WindModel"]["DataFile"])


def get_simulation(open_rocket_helper, ork_file_path, i_simulation):
    """Return the simulation with the given index from the .ork file.

    :arg open_rocket_helper:
        Instance of ``orhelper.Helper()``

    :raise IndexError:
        If `i_simulation` is negative or >= the number of simulations in
        the given .ork file
    """
    doc = open_rocket_helper.load_doc(ork_file_path)
    n_simulations = doc.getSimulationCount()
    if i_simulation < 0 or i_simulation >= n_simulations:
        raise IndexError(
            "Simulation index is out of bounds!\n"
            + "i_simulation  = {}\n".format(i_simulation)
            + "n_simulations = {}\n".format(doc.getSimulationCount()))
    sim = doc.getSimulation(i_simulation)
    print("Load simulation number {} called {}.".format(
        i_simulation, sim.getName()))
    return sim


def set_up_random_parameters(orh, sim, config):
    """Return iterable components, original values, random parameters.

    The random parameters are actually lambdas that return a new random
    sample when called.
    """
    options = sim.getOptions()
    # this does not work, because at every save this setting is not kept..
    #azimuth_mean = options.getLaunchRodDirection()
    azimuth_mean = math.radians(float(config["LaunchRail"]["AzimuthMean"]))
    azimuth_stddev = math.radians(float(config["LaunchRail"]["Azimuth"]))
    tilt_mean = options.getLaunchRodAngle()
    tilt_stddev = math.radians(float(config["LaunchRail"]["Tilt"]))
    thrust_factor_stddev = float(config["Propulsion"]["ThrustFactor"])
    fincant_stddev = float(config["Aerodynamics"]["FinCant"])
    parachute_cd_stddev = float(config["Aerodynamics"]["ParachuteCd"])
    roughness_stddev = float(config["Aerodynamics"]["Roghness"])

    # get rocket data
    opts = sim.getOptions()
    rocket = opts.getRocket()

    # TODO: Move the components into its own function get_components()
    # or something. It makes no sense for this to be here.
    components_fin_sets = []
    components_parachutes = []
    components_external_components = []
    # TODO: Remove these original values. They should be set as the mean
    # of the normal distribution. If there are multiple fin_cants, etc.
    # just make that part of the RandomParameters a list.
    original_fin_cant = []
    original_parachute_cd = []
    original_roughness = []
    print("Rocket has {} stage(s).".format(rocket.getStageCount()))
    for component in orhelper.JIterator(rocket):
        logging.debug("Component {} of type {}".format(component.getName(),
                                                       type(component)))
        # fins can be
        #   orh.openrocket.rocketcomponent.FreeformFinSet
        #   orh.openrocket.rocketcomponent.TrapezoidFinSet
        #   orh.openrocket.rocketcomponent.EllipticalFinSet
        if (isinstance(component, orh.openrocket.rocketcomponent.FinSet)):
            # FIXME: Is it rad or °? In randomize_simulation() it is °.
            print("Finset({}) with ".format(component.getName())
                  + "cant angle {:6.2f}rad".format(component.getCantAngle()))
            components_fin_sets.append(component)
            original_fin_cant.append(component.getCantAngle())
        if isinstance(component, orh.openrocket.rocketcomponent.Parachute):
            print("Parachute with drag surface diameter "
                  + "{:6.2f}m and ".format(component.getDiameter())
                  + "CD of {:6.2f}".format(component.getCD()))
            components_parachutes.append(component)
            original_parachute_cd.append(component.getCD())
        if isinstance(component,
                      orh.openrocket.rocketcomponent.ExternalComponent):
            print("External component {} with finish {}".format(
                component, component.getFinish()))
            components_external_components.append(component)
            original_roughness.append(component.getFinish().getRoughnessSize())

    print("Initial launch rail tilt    = {:6.2f}°".format(
        math.degrees(tilt_mean)))
    print("Initial launch rail azimuth = {:6.2f}°".format(
        math.degrees(azimuth_mean)))

    RocketComponents = collections.namedtuple("RocketComponents", [
        "fin_sets",
        "parachutes",
        "external_components"])
    OriginalParameters = collections.namedtuple("OriginalParameters", [
        "fin_cant",
        "parachute_cd",
        "roughness"])
    rng = np.random.default_rng()
    RandomParameters = collections.namedtuple("RandomParameters", [
        "tilt",
        "azimuth",
        "thrust_factor",
        "fin_cant",
        "parachute_cd",
        "roughness"])
    return (
        RocketComponents(
            fin_sets=components_fin_sets,
            parachutes=components_parachutes,
            external_components=components_external_components),
        OriginalParameters(
            fin_cant=original_fin_cant,
            parachute_cd=original_parachute_cd,
            roughness=original_roughness),
        RandomParameters(
            tilt=lambda: rng.normal(tilt_mean, tilt_stddev),
            azimuth=lambda: rng.normal(azimuth_mean, azimuth_stddev),
            thrust_factor=lambda: rng.normal(1, thrust_factor_stddev),
            fin_cant=lambda: rng.normal(0, fincant_stddev),
            parachute_cd=lambda: rng.normal(0, parachute_cd_stddev),
            roughness=lambda: rng.normal(0, roughness_stddev)))


def randomize_simulation(open_rocket_helper, sim, rocket_components,
                         original_parameters, random_parameters):
    """Set simulation parameters to random samples."""
    logging.info("Randomize variables...")
    options = sim.getOptions()
    options.setLaunchRodAngle(random_parameters.tilt())
    # Otherwise launch rod direction cannot be altered
    options.setLaunchIntoWind(False)
    options.setLaunchRodDirection(random_parameters.azimuth())
    tilt = math.degrees(options.getLaunchRodAngle())
    azimuth = math.degrees(options.getLaunchRodDirection())
    logging.info("Launch rail tilt    = {:6.2f}°".format(tilt))
    logging.info("Launch rail azimuth = {:6.2f}°".format(azimuth))

    # There can be more than one finset -> add unbiased
    # normaldistributed value
    ct = 0
    logging.info("Finset: ")
    for fins in rocket_components.fin_sets:
        fins.setCantAngle(original_parameters.fin_cant[ct]
                          + random_parameters.fin_cant())
        # FIXME: Is it rad or °? In set_up_random_parameters() it is rad.
        logging.info("{} with cant angle {:6.2f}°".format(fins.getName(),
                                                          fins.getCantAngle()))
        ct += 1

    # There can be more than one parachute -> add unbiased
    # normaldistributed value
    ct = 0
    logging.info("Parachutes: ")
    for parachute in rocket_components.parachutes:
        parachute.setCD(max(original_parameters.parachute_cd[ct]
                            + random_parameters.parachute_cd(), 0.))
        logging.info(parachute.getName(),
                     "with CD {:6.2f}".format(
            parachute.getCD()))
        ct += 1

    # TODO: How can one change the finish roughness with arbitrary
    # values? the Finish(string, double) constructor is private:
    # https://github.com/openrocket/openrocket/blob/unstable/core/src/net/sf/openrocket/rocketcomponent/ExternalComponent.java#L38-L41
    # http://tutorials.jenkov.com/java/enums.html#enum-fields
    # Workaround with randomized variable put into bins and using
    # predefined enums
    # //// Rough
    # ROUGH("ExternalComponent.Rough", 500e-6),
    # //// Unfinished
    # UNFINISHED("ExternalComponent.Unfinished", 150e-6),
    # //// Regular paint
    # NORMAL("ExternalComponent.Regularpaint", 60e-6),
    # //// Smooth paint
    # SMOOTH("ExternalComponent.Smoothpaint", 20e-6),
    # //// Polished
    # POLISHED("ExternalComponent.Polished", 2e-6);
    roughness_values = np.array([500e-6, 150e-6, 60e-6, 20e-6, 2e-6])
    # Calculate bin edges, average between adjacent roughness_values
    roughness_bins = (roughness_values[1:] + roughness_values[:-1]) / 2.
    logging.debug("bins {}".format(roughness_bins))
    ct = 0
    logging.info("External components: ")
    for ext_comp in rocket_components.external_components:
        roughness_random = original_parameters.roughness[ct] + \
            random_parameters.roughness()
        roughness_in_bin = np.digitize(roughness_random, roughness_bins)
        logging.debug(
            "roughness {} is in bin {}, i.e. {}".format(
                roughness_random,
                roughness_in_bin,
                roughness_values[roughness_in_bin]))
        if roughness_in_bin == 0:
            ext_comp.setFinish(ext_comp.Finish.ROUGH)
        elif roughness_in_bin == 1:
            ext_comp.setFinish(ext_comp.Finish.UNFINISHED)
        elif roughness_in_bin == 2:
            ext_comp.setFinish(ext_comp.Finish.NORMAL)
        elif roughness_in_bin == 3:
            ext_comp.setFinish(ext_comp.Finish.SMOOTH)
        elif roughness_in_bin == 4:
            ext_comp.setFinish(ext_comp.Finish.POLISHED)
        logging.info(ext_comp,
                     " with finish ",
                     ext_comp.getFinish())
        ct += 1


def run_simulation(orh, sim, config, random_parameters):
    """Run a single simulation and return the results.

    :return:
        A tuple containing (landing_point, launch_position,
        geodetic_computation, apogee, trajectory)
    """
    wind_listener = WindListener(config["WindModel"]["DataFile"])
    launch_point_listener = LaunchPointListener()
    landing_point_listener = LandingPointListener()
    motor_listener = MotorListener(
        random_parameters.thrust_factor(),
        float(config["Propulsion"]["NozzleCrossSection"]))

    orh.run_simulation(
        sim, listeners=(launch_point_listener,
                        landing_point_listener,
                        wind_listener,
                        motor_listener))
    apogee = get_apogee(orh, sim)
    trajectory = get_trajectory(orh, sim)
    # TODO: Return results in a nicer way, using a dictionary or
    # namedtuple for example. This makes handling afterwards easier
    # since we don't have to know which result is at which index
    return (landing_point_listener.landing_points_world[0],
            launch_point_listener.launch_point,
            launch_point_listener.geodetic_computation,
            apogee,
            trajectory,
            landing_point_listener.landing_points_cartesian[0],
            )


class LaunchPointListener(orhelper.AbstractSimulationListener):
    """Return information at the ``startSimulation`` callback."""

    def __init__(self):
        self.launch_point = None
        self.geodetic_computation = None

    def startSimulation(self, status):
        """Analyze the simulation conditions of OpenRocket.

        These are the launch point and if a supported geodetic model is
        set.

        :raise ValueError:
            If the geodetic computation is not flat or WGS84
        """
        conditions = status.getSimulationConditions()
        self.launch_point = conditions.getLaunchSite()

        self.geodetic_computation = conditions.getGeodeticComputation()
        logging.debug(self.geodetic_computation)
        computation_is_supported = (
            self.geodetic_computation == self.geodetic_computation.FLAT
            or self.geodetic_computation == self.geodetic_computation.WGS84)
        if not computation_is_supported:
            raise ValueError("GeodeticComputationStrategy type not supported!")


class LandingPointListener(orhelper.AbstractSimulationListener):
    """Return the landing point at the ``endSimulation`` callback."""

    def __init__(self):
        # FIXME: This is a weird workaround because I don't know how to
        # create the member variable of the correct type.
        self.landing_points_world = []
        self.landing_points_cartesian = []

    def endSimulation(self, status, simulation_exception):
        """Return the landing position from OpenRocket and check exceptions

        (from OR code) simulation_exception:
        the exception that caused ending the simulation, or <code>null</code> if
        ending normally.
        """
        self.landing_points_world.append(status.getRocketWorldPosition())
        self.landing_points_cartesian.append(status.getRocketPosition())
        # TODO: do something senseful if exception is thrown
        if simulation_exception is not None:
            logging.warning(
                "Simulation threw the exception ",
                (simulation_exception.args[0]))


class MotorListener(orhelper.AbstractSimulationListener):
    """Override the thrust of the motor."""

    def __init__(self, thrust_factor, nozzle_cross_section_mm2):
        self.thrust_factor = thrust_factor
        logging.info("Used thrust factor = {:6.2f}".format(thrust_factor))
        self.nozzle_cross_section = nozzle_cross_section_mm2 * 1e-6
        logging.info("Nozzle cross section = {:6.2g}mm^2".format(
            nozzle_cross_section_mm2))
        self.pressure = STANDARD_PRESSURE

    def postAtmosphericModel(self, status, atmospheric_conditions):
        """Get the ambient pressure from the atmospheric model."""
        self.pressure = atmospheric_conditions.getPressure()

    def postSimpleThrustCalculation(self, status, thrust):
        """Return the adapted thrust."""
        # add thrust_increse if motor is burning and apply factor
        thrust_increase = (
            STANDARD_PRESSURE - self.pressure) * self.nozzle_cross_section
        if thrust >= thrust_increase:
            logging.debug("Thrust increase due to decreased ambient pressure "
                          + "= {:6.2f}N".format(thrust_increase))
            return self.thrust_factor * thrust + thrust_increase
        else:
            return self.thrust_factor * thrust


class WindListener(orhelper.AbstractSimulationListener):
    """Set the wind speed as a function of altitude."""

    def __init__(self, wind_model_file=""):
        """Read wind level model data from file.

        Save them as interpolation functions to be used in other
        callbacks of this class.

        :raise ValueError:
            If the arrays loaded from the `wind_model_file` are of
            unequal length
        """
        try:
            # Read wind level model data from file
            data = np.loadtxt(wind_model_file)
        except (IOError, FileNotFoundError):
            self._default_wind_model_is_used = True
            logging.warning("Warning: wind model file '{}' ".format(wind_model_file)
                            + "not found! Default wind model will be used.")
            return

        self._default_wind_model_is_used = False
        altitudes_m = data[:, 0]
        wind_directions_degree = data[:, 1]
        wind_directions_rad = np.radians(wind_directions_degree)
        wind_speeds_mps = data[:, 2]
        wind_speeds_north_mps = wind_speeds_mps * np.cos(wind_directions_rad)
        wind_speeds_east_mps = wind_speeds_mps * np.sin(wind_directions_rad)

        if (len(altitudes_m) != len(wind_directions_degree)
                or len(altitudes_m) != len(wind_speeds_mps)):
            raise ValueError(
                "Aloft data is incorrect! `altitudes_m`, "
                + "`wind_directions_degree` and `wind_speeds_mps` must be "
                + "of the same length.")

        logging.debug("Input wind levels model data:")
        logging.debug("Altitude (m) ")
        logging.debug(altitudes_m)
        logging.debug("Direction (°) ")
        logging.debug(wind_directions_degree)
        logging.debug("Wind speed (m/s) ")
        logging.debug(wind_speeds_mps)

        # assume that the intermediate values of a rotation of the wind by 180° is zero 
        # instead of the same magnitude and 90° rotaion -> 
        # use x/y coordinates for interpolation
        # TODO: Which fill_values shall be used above the wind
        # data? zero? last value? extrapolate?
        self.interpolate_wind_speed_north_mps = scipy.interpolate.interp1d(
            altitudes_m, wind_speeds_north_mps, bounds_error=False,
            fill_value=(wind_speeds_north_mps[0], wind_speeds_north_mps[-1]))
        self.interpolate_wind_speed_east_mps = scipy.interpolate.interp1d(
            altitudes_m, wind_speeds_east_mps, bounds_error=False,
            fill_value=(wind_speeds_east_mps[0], wind_speeds_east_mps[-1]))

    def preWindModel(self, status):
        """Set the wind coordinates at every simulation step."""
        if self._default_wind_model_is_used:
            return None

        position = status.getRocketPosition()
        wind_speed_north_mps = self.interpolate_wind_speed_north_mps(position.z)
        wind_speed_east_mps = self.interpolate_wind_speed_east_mps(position.z)
        wind_speed_mps = math.sqrt(wind_speed_north_mps*wind_speed_north_mps
                            + wind_speed_east_mps*wind_speed_east_mps)
        wind_direction_rad = math.atan2(wind_speed_east_mps, wind_speed_north_mps)

        conditions = status.getSimulationConditions()
        wind_model = conditions.getWindModel()
        wind_model.setDirection(wind_direction_rad)
        wind_model.setAverage(wind_speed_mps)
        conditions.setWindModel(wind_model)

    def postWindModel(self, status, wind):
        logging.debug("Wind: {}".format(wind))


# TODO: Maybe we can get all the interesting simulation results in a
# similar way. This way run_simulation() no longer needs to return the
# results. Instead something like get_simulation_results() could be
# implemented. This would be a nice separation of
# concerns/responsibilities.
def get_apogee(open_rocket_helper, simulation):
    """Return the apogee of the simulation as ``WorldCoordinate``."""
    FlightDataType = orhelper.FlightDataType
    data = open_rocket_helper.get_timeseries(simulation, [
        FlightDataType.TYPE_TIME,
        FlightDataType.TYPE_ALTITUDE,
        FlightDataType.TYPE_LONGITUDE,
        FlightDataType.TYPE_LATITUDE])
    t = np.array(data[FlightDataType.TYPE_TIME])
    altitude = np.array(data[FlightDataType.TYPE_ALTITUDE])
    longitude = np.array(data[FlightDataType.TYPE_LONGITUDE])
    latitude = np.array(data[FlightDataType.TYPE_LATITUDE])

    events = open_rocket_helper.get_events(simulation)
    t_apogee = events[orhelper.FlightEvent.APOGEE][0]
    apogee = open_rocket_helper.openrocket.util.WorldCoordinate(
        math.degrees(latitude[t == t_apogee]),
        math.degrees(longitude[t == t_apogee]),
        altitude[t == t_apogee])
    logging.debug(
        "Apogee at {:.1f}s: ".format(t_apogee)
        + "longitude {:.1f}°, ".format(apogee.getLatitudeDeg())
        + "latitude,{:.1f}°, ".format(apogee.getLongitudeDeg())
        + "altitude {:.1f}m".format(apogee.getAltitude()))
    return apogee


# TODO: Return x, y or lat, lon depending on `coordinate_type`
# TODO: I guess we should also directly use these x, y values for things
# like apogee, landing points, launch point, etc.
def get_trajectory(open_rocket_helper, simulation):
    """Return the x, y and altitude values of the rocket.

    :return:
        List of 3 arrays containing the values for x, y and altitude at
        each simulation step
    """
    FlightDataType = orhelper.FlightDataType
    data = open_rocket_helper.get_timeseries(simulation, [
        FlightDataType.TYPE_POSITION_X,
        FlightDataType.TYPE_POSITION_Y,
        FlightDataType.TYPE_ALTITUDE])
    x = np.array(data[FlightDataType.TYPE_POSITION_X])
    y = np.array(data[FlightDataType.TYPE_POSITION_Y])
    altitude = np.array(data[FlightDataType.TYPE_ALTITUDE])
    return [x, y, altitude]


def print_statistics(results):
    """Print statistics of all simulation results."""
    landing_points = [r[0] for r in results]
    launch_point = results[0][1]
    geodetic_computation = results[0][2]
    apogees = [r[3] for r in results]

    logging.debug("Results: distances in cartesian coordinates")
    distances = []
    bearings = []
    for landing_point in landing_points:
        if geodetic_computation == geodetic_computation.FLAT:
            distance, bearing = compute_distance_and_bearing_flat(
                launch_point, landing_point)
        elif geodetic_computation == geodetic_computation.WGS84:
            geodesic = pyproj.Geod(ellps="WGS84")
            fwd_azimuth, back_azimuth, distance = geodesic.inv(
                launch_point.getLongitudeDeg(),
                launch_point.getLatitudeDeg(),
                landing_point.getLongitudeDeg(),
                landing_point.getLatitudeDeg())
            bearing = np.radians(fwd_azimuth)

        distances.append(distance)
        bearings.append(bearing)

    max_altitude = []
    for apogee in apogees:
        max_altitude.append(apogee.getAltitude())

    logging.debug("distances and bearings in polar coordinates")
    logging.debug(distances)
    logging.debug(bearings)
    
    #TODO how can one calculate the 2pi-safe statistics of the bearing
    print("---")
    print("Apogee: {:.1f}m ± {:.2f}m ".format(
        np.mean(max_altitude), np.std(max_altitude)))
    print(
        "Rocket landing zone {:.1f}m ± {:.2f}m ".format(
            np.mean(distances), np.std(distances))
        + "bearing {:.1f}° ± {:.1f}° ".format(
            np.degrees(np.mean(bearings)), np.degrees(np.std(bearings))))
    print("Based on {} simulations.".format(
        len(landing_points)))


def compute_distance_and_bearing_flat(from_, to):
    """Return distance and bearing betweeen two points.

    Valid for flat earth approximation only.

    :arg WorldCoordinate from_:
        First coordinate
    :arg WorldCoordinate to:
        Second coordinate

    :return:
        A tuple containing (distance in m, bearing in °)
    """
    # uses the world coordinates used by OR if 
    # geodetic_computation=FLAT is set
    dx = ((to.getLongitudeDeg() - from_.getLongitudeDeg())
          * METERS_PER_DEGREE_LONGITUDE_EQUATOR)
    dy = ((to.getLatitudeDeg() - from_.getLatitudeDeg())
          * METERS_PER_DEGREE_LATITUDE)
    logging.debug("Longitude {:.1f}°, Latitude {:.1f}°".format(
        to.getLongitudeDeg(), to.getLatitudeDeg()))
    logging.debug("dx {:.1f}m, dy {:.1f}m".format(dx, dy))
    distance = math.sqrt(dx * dx + dy * dy)
    bearing = math.pi / 2. - math.atan2(dy, dx)

    return distance, bearing


# TODO: Try to refactor this ugly plotting function
def create_plots(results, output_filename, plot_coordinate_type="flat",
                 results_are_shown=False):
    """Create, store and optionally show the plots of the results.

    :raise ValueError:
        If Coordinate type is not "flat" or "wgs84".
    """
    def to_array_world(coordinate):
        return np.array([coordinate.getLatitudeDeg(),
                        coordinate.getLongitudeDeg(),
                        coordinate.getAltitude()])

    def to_array_cartesian(coordinate):
        return np.array([coordinate.x,
                        coordinate.y,
                        coordinate.z])

    if PLOTS_ARE_TESTED:
        # Test Data
        rng = np.random.default_rng()
        n_simulations = 1000
        lat = rng.normal(55, 2, n_simulations)
        lon = rng.normal(20, 1, n_simulations)
        landing_points_world = np.array([lat, lon]).T
        landing_points_cartesian = np.array([lat, lon]).T
        alt = rng.normal(15346, 17, n_simulations)
        apogees = np.zeros((n_simulations, 3))
        apogees[:, 2] = alt
        geodetic_computation = 'flat'
    else:
        landing_points_world = np.array(
            [to_array_world(r[0]) for r in results])
        landing_points_cartesian = np.array(
            [to_array_cartesian(r[5]) for r in results])
        launch_point = to_array_world(results[0][1])
        geodetic_computation = results[0][2]
        apogees = np.array([to_array_world(r[3]) for r in results])
        trajectories = [r[4] for r in results]
        geodetic_computation = results[0][2]

    fig = plt.figure(constrained_layout=True)
    # TODO: Increase pad on the right
    spec = mpl.gridspec.GridSpec(nrows=2, ncols=2, figure=fig,
                                 width_ratios=[1.5, 1],
                                 height_ratios=[3.5, 1])
    ax_trajectories = fig.add_subplot(spec[:, 0], projection='3d')
    ax_landing_points = fig.add_subplot(spec[0, 1])
    ax_apogees = fig.add_subplot(spec[1, 1])

    # Scatter plot of landing coordinates
    ax_lps = ax_landing_points
    ax_lps.set_title("Landing Points")
    if plot_coordinate_type == "flat":
        # OR simulates with cartesian coordinates -> take them directly
        x = landing_points_cartesian[:, 0]
        y = landing_points_cartesian[:, 1]
        x0 = 0
        y0 = 0
        ax_lps.set_xlabel(r"$\Delta x$ in m")
        ax_lps.set_ylabel(r"$\Delta y$ in m")
    elif plot_coordinate_type == "wgs84":
        # use world coordinates with OR's implementation of WGS84
        if geodetic_computation == geodetic_computation.FLAT:
            raise ValueError(
                "Wrong geodetic_computation set in OR for plot coordinate type {}, change to WGS84!".format(plot_coordinate_type))
        x = landing_points_world[:, 1]
        y = landing_points_world[:, 0]
        x0 = launch_point[1]
        y0 = launch_point[0]
        ax_lps.set_xlabel("Longitude in °")
        ax_lps.set_ylabel("Latitude in °")
    else:
        raise ValueError(
            "Coordinate type {} is not supported for plotting! ".format(
                plot_coordinate_type)
            + "Valid values are 'flat' and 'wgs84'.")
    ax_lps.plot(x0, y0, "bx", markersize=5, zorder=0, label="Launch")
    ax_lps.plot(x, y, "r.", markersize=3, zorder=1, label="Landing")
    # FIXME: For some reason, this does not change the x limits so
    # equalizing xlim and ylim with the 3d plot later on does not work.
    ax_lps.axis("equal")
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = mpl.rcParams["axes.prop_cycle"].by_key()["linestyle"]
    confidence_ellipse(x, y, ax_lps, n_std=1, label=r"$1\sigma$",
                       edgecolor=colors[2], ls=linestyles[0])
    confidence_ellipse(x, y, ax_lps, n_std=2, label=r"$2\sigma$",
                       edgecolor=colors[1], ls=linestyles[1])
    confidence_ellipse(x, y, ax_lps, n_std=3, label=r"$3\sigma$",
                       edgecolor=colors[0], ls=linestyles[2])
    ax_lps.legend()
    ax_lps.ticklabel_format(useOffset=False, style="plain")

    # Histogram of apogee altitudes
    ax_apogees.set_title("Apogees")
    n_simulations = apogees.shape[0]
    n_bins = int(round(1 + 3.322 * math.log(n_simulations), 0))
    ax_apogees.hist(apogees[:, 2], bins=n_bins, orientation="vertical",
                    fc=colors[2], ec="k")
    ax_apogees.set_xlabel("Altitude in m")
    ax_apogees.set_ylabel("Number of Simulations")

    # Plot the trajectories
    ax_trajectories.set_title("Trajectories")
    colors = line_color_map(np.linspace(0, 1, len(trajectories)))
    if plot_coordinate_type == "flat":
        for trajectory, color in zip(trajectories, colors):
            x, y, alt = trajectory
            ax_trajectories.plot(
                xs=x, ys=y, zs=alt, color=color, linestyle="-")
        ax_trajectories.ticklabel_format(useOffset=False, style="plain")
        # Set x and y limits equal to that of the landing points plot. Does
        # not work because .axis("equal") is strange.
        xlim = ax_lps.get_xlim()
        ylim = ax_lps.get_ylim()
        logging.debug("xlim", xlim)
        logging.debug("ylim", ylim)
        ax_trajectories.set_xlim(xlim)
        ax_trajectories.set_ylim(ylim)
        ax_trajectories.set_xlabel("x in m")
        ax_trajectories.set_ylabel("y in m")
    elif plot_coordinate_type == "wgs84":
        # TODO create data also in WGS84, depending on plot_coordinate_type
        for trajectory, color in zip(trajectories, colors):
            x, y, alt = trajectory
            ax_trajectories.plot(
                xs=x, ys=y, zs=alt, color=color, linestyle="-")
        ax_trajectories.ticklabel_format(useOffset=False, style="plain")
        ax_trajectories.set_xlabel("x in m")
        ax_trajectories.set_ylabel("y in m")

    else:
        raise ValueError(
            "Coordinate type {} is not supported for plotting! ".format(
                plot_coordinate_type)
            + "Valid values are 'flat' and 'wgs84'.")
    ax_trajectories.set_zlabel("altitude in m")

    # Save and show the figure
    plt.suptitle("Dispersion Analysis of {} Simulations".format(n_simulations))
    plt.savefig(output_filename)
    if results_are_shown:
        plt.show()


# TODO: Convert docstring style
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = mpl.patches.Ellipse(
        (0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
        facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (mpl.transforms.Affine2D().
              rotate_deg(45).
              scale(scale_x, scale_y).
              translate(mean_x, mean_y))

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def get_object_methods(obj):
    """Return object methods for debugging/development"""
    object_methods = [method_name for method_name in dir(obj)
                      if callable(getattr(obj, method_name))]
    print(object_methods)


if __name__ == "__main__":
    diana()

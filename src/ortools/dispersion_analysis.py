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
import simplekml

import os
import sys
import csv
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
WINDMODEL_TEST = False
EXCEPTION_FOR_MISSING_EVENTS = False

STANDARD_PRESSURE = 101325.0  # The standard air pressure (1.01325 bar)
METERS_PER_DEGREE_LATITUDE = 111325.0
METERS_PER_DEGREE_LONGITUDE_EQUATOR = 111050.0

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option("--directory", "-d", type=click.Path(exists=True),
              default="examples/", show_default=True,
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
              help="Show the plots.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed results on screen.")
def diana(directory, filename, config, output,
          plot_coordinate_type, show, verbose):
    """Do a dispersion analysis of an OpenRocket simulation.

    A dispersion analysis runs multiple simulations with slightly
    varying parameters. The config file specifies which simulation
    parameters are varied and by how much as well as the total number of
    simulations run.

    Example usage:

        diana -d examples -o test.pdf -s
    """
    t0 = time.time()
    config_file_path = config or utility.find_latest_file(".ini", directory)
    config = configparser.ConfigParser()
    # TODO: Define default values for all parameters of the .ini fileata file)
    config.read(config_file_path)
    make_paths_in_config_absolute(config, config_file_path)

    timestamp = str(int(time.time()))
    output_filename = output or "dispersion_analysis_" + timestamp
    results_are_shown = show
    ork_file_path = config["General"]["OrkFile"]
    print("directory   : {}".format(directory))
    print(".ork file   : {}".format(ork_file_path))
    print("config file : {}".format(config_file_path))
    print("output file : {}".format(output_filename))

    # Setup of logging on stderr
    if verbose:
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    else:
        # Use logging.WARNING, or logging.DEBUG if necessary
        logging.basicConfig(stream=sys.stderr, level=logging.WARNING)

    if WINDMODEL_TEST:
        plot_wind_model(config["WindModel"]["DataFile"])
        return

    if PLOTS_ARE_TESTED:
        create_plots(
            [],
            output_filename,
            general_parameters,
            plot_coordinate_type,
            results_are_shown)
        return

    with orhelper.OpenRocketInstance() as instance:
        t1 = time.time()
        orh = orhelper.Helper(instance)
        i_simulation = int(config["General"]["SimulationIndex"])
        sim = get_simulation(orh, ork_file_path, i_simulation)

        # get global simulation parameters
        options = sim.getOptions()
        geodetic_computation = options.getGeodeticComputation()
        #:raise ValueError:
        #    If the geodetic computation is not flat or WGS84
        logging.info(
            "Geodetic computation {} found.".format(geodetic_computation))
        computation_is_supported = (
            geodetic_computation == geodetic_computation.FLAT
            or geodetic_computation == geodetic_computation.WGS84)
        if not computation_is_supported:
            raise ValueError("GeodeticComputationStrategy type not supported!")

        launch_point = orh.openrocket.util.WorldCoordinate(
            options.getLaunchLatitude(),
            options.getLaunchLongitude(),
            options.getLaunchAltitude())
        logging.info("Launch Point {} found.".format(launch_point))

        GeneralParameters = collections.namedtuple("GeneralParameters", [
            "launch_point",
            "geodetic_computation"])
        general_parameters = GeneralParameters(
            launch_point=launch_point,
            geodetic_computation=geodetic_computation)

        # get random parameters
        rocket_components, random_parameters = \
            set_up_random_parameters(orh, sim, config)

        results = []
        parametersets = []
        n_simulations = int(config["General"]["NumberOfSimulations"])
        for i in range(n_simulations):
            print("-- Running simulation {:4} of {} --".format(
                i + 1, n_simulations))
            if i > 0:
                randomize_simulation(orh, sim, rocket_components,
                                     random_parameters)
                parameterset = get_simulation_parameters(orh,
                                                         sim, rocket_components,
                                                         random_parameters, True)
            else:
                print("with nominal parameter set but wind-model applied")
                parameterset = get_simulation_parameters(orh,
                                                         sim, rocket_components,
                                                         random_parameters, False)

            result = run_simulation(orh, sim, config, parameterset)

            results.append(result)
            parametersets.append(parameterset)

        # the following functions rely on orhelper
        # -> run them before JVM is shut down
        t2 = time.time()
        print_statistics(results, general_parameters)
        export_results_csv(
            results,
            parametersets,
            general_parameters,
            output_filename)
        export_results_kml(
            results,
            parametersets,
            general_parameters,
            output_filename)
        t3 = time.time()
        print("---")
        print("time for {} simulations = {:.1f}s".format(n_simulations,
                                                         t2 - t1))
        print("total execution time = {:.1f}s".format(t3 - t0))
        create_plots(results, output_filename, general_parameters,
                     plot_coordinate_type, results_are_shown)


def make_paths_in_config_absolute(config, config_file_path):
    """Turn all paths in the diana config file into absolute ones."""
    directory = os.path.dirname(os.path.abspath(config_file_path))
    config["WindModel"]["DataFile"] = os.path.join(
        directory, config["WindModel"]["DataFile"])
    config["General"]["OrkFile"] = os.path.join(
        directory, config["General"]["OrkFile"])


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
    # this does not work, because at every save this setting is not kept.
    #azimuth_mean = options.getLaunchRodDirection()
    # Workaround: set setLaunchIntoWind(False) and read azimuth from ini file
    options.setLaunchIntoWind(False)
    options.setLaunchRodDirection(
        math.radians(float(config["LaunchRail"]["AzimuthMean"])))

    azimuth_intowind = options.getLaunchIntoWind()
    azimuth_mean = options.getLaunchRodDirection()

    azimuth_stddev = math.radians(float(config["LaunchRail"]["Azimuth"]))
    if config.has_option("LaunchRail", "TiltMean"):
        #override OR settings
        tilt_mean = math.radians(float(config["LaunchRail"]["TiltMean"]))        
        options.setLaunchRodAngle(tilt_mean)
    else:
        tilt_mean = options.getLaunchRodAngle()
    tilt_stddev = math.radians(float(config["LaunchRail"]["Tilt"]))
    thrust_factor_stddev = float(config["Propulsion"]["ThrustFactor"])
    fincant_stddev = math.radians(float(config["Aerodynamics"]["FinCant"]))
    parachute_cd_stddev = float(config["Aerodynamics"]["ParachuteCd"])
    roughness_stddev = float(config["Aerodynamics"]["Roughness"])

    mcid = options.getMotorConfigurationID()
    rocket = options.getRocket()
    num_stages = rocket.getChildCount()
    stages = []
    stage_separation_delay_max = []
    stage_separation_delay_min = []
    for stage_nr in range(1, num_stages):
        stages.append(rocket.getChild(stage_nr))
        separationEventConfiguration = rocket.getChild(
            stage_nr).getStageSeparationConfiguration().get(mcid)
        separationDelay = separationEventConfiguration.getSeparationDelay()
        if config.has_section("Staging") and config.has_option(
                "Staging", "StageSeparationDelayDeltaNeg") and config.has_option("Staging", "StageSeparationDelayDeltaPos"):
            # TODO set motor burnout of booster stage as lower minimum, and
            # motor ignition of upper stage as maximum
            stage_separation_delay_min.append(separationDelay + float(
                config["Staging"]["StageSeparationDelayDeltaNeg"]))
            stage_separation_delay_max.append(separationDelay + float(
                config["Staging"]["StageSeparationDelayDeltaPos"]))
        else:
            stage_separation_delay_max.append(separationDelay)
            stage_separation_delay_min.append(separationDelay)

    # get simulation data
    opts = sim.getOptions()

    # get rocket data
    rocket = opts.getRocket()

    # TODO: Move the components into its own function get_components()
    # or something. It makes no sense for this to be here.
    components_fin_sets = []
    components_parachutes = []
    components_external_components = []

    fin_cant_means = []
    parachute_cd_means = []
    component_roughness_means = []
    print("Rocket has {} stage(s).".format(rocket.getStageCount()))
    for component in orhelper.JIterator(rocket):
        logging.debug("Component {} of type {}".format(component.getName(),
                                                       type(component)))
        # fins can be
        #   orh.openrocket.rocketcomponent.FreeformFinSet
        #   orh.openrocket.rocketcomponent.TrapezoidFinSet
        #   orh.openrocket.rocketcomponent.EllipticalFinSet
        if (isinstance(component, orh.openrocket.rocketcomponent.FinSet)):
            logging.info("Finset({}) with ".format(component.getName())
                         + "cant angle {:6.2f}°".format(math.degrees(component.getCantAngle())))
            components_fin_sets.append(component)
            fin_cant_means.append(component.getCantAngle())
        if isinstance(component, orh.openrocket.rocketcomponent.Parachute):
            logging.info("Parachute with drag surface diameter "
                         + "{:6.2f}m and ".format(component.getDiameter())
                         + "CD of {:6.2f}".format(component.getCD()))
            components_parachutes.append(component)
            parachute_cd_means.append(component.getCD())
        if isinstance(component,
                      orh.openrocket.rocketcomponent.ExternalComponent):
            logging.info("External component {} with finish {}".format(
                component, component.getFinish()))
            components_external_components.append(component)
            component_roughness_means.append(
                component.getFinish().getRoughnessSize())

    logging.info("Initial launch rail tilt    = {:6.2f}°".format(
        math.degrees(tilt_mean)))
    logging.info("Initial launch rail azimuth = {:6.2f}°".format(
        math.degrees(azimuth_mean)))

    RocketComponents = collections.namedtuple("RocketComponents", [
        "fin_sets",
        "parachutes",
        "external_components",
        "stages"])
    rng = np.random.default_rng()
    RandomParameters = collections.namedtuple("RandomParameters", [
        "tilt",
        "azimuth",
        "thrust_factor",
        "stage_separation",
        "fin_cants",
        "parachutes_cd",
        "roughnesses"])
    return (
        RocketComponents(
            fin_sets=components_fin_sets,
            parachutes=components_parachutes,
            external_components=components_external_components,
            stages=stages),
        RandomParameters(
            tilt=lambda: rng.normal(tilt_mean, tilt_stddev),
            azimuth=lambda: rng.normal(azimuth_mean, azimuth_stddev),
            thrust_factor=lambda: rng.normal(1, thrust_factor_stddev),
            stage_separation=lambda: [rng.uniform(min,
                                                  max) for (min, max) in zip(
                stage_separation_delay_min,
                stage_separation_delay_max)],
            fin_cants=lambda: [rng.normal(mean, fincant_stddev)
                               for mean in fin_cant_means],
            parachutes_cd=lambda: [
                rng.normal(
                    mean,
                    parachute_cd_stddev) for mean in parachute_cd_means],
            roughnesses=lambda: [rng.normal(mean, roughness_stddev) for mean in component_roughness_means]))


def randomize_simulation(open_rocket_helper, sim, rocket_components,
                         random_parameters):
    """Set simulation parameters to random samples.
    return the global parameter set"""
    logging.info("Randomize variables...")
    options = sim.getOptions()
    options.setLaunchRodAngle(random_parameters.tilt())
    # Otherwise launch rod direction cannot be altered
    options.setLaunchIntoWind(False)
    options.setLaunchRodDirection(random_parameters.azimuth())

    # set stage sepration
    mcid = options.getMotorConfigurationID()
    rocket = options.getRocket()
    num_stages = rocket.getChildCount()
    for (stage, stage_separation_delay) in zip(
            rocket_components.stages, random_parameters.stage_separation()):
        separationEventConfiguration = stage.getStageSeparationConfiguration().get(mcid)
        logging.info(
            "Set separation delay of stage {}".format(stage))
        separationEventConfiguration.setSeparationDelay(
            stage_separation_delay)

    # There can be more than one finset -> add unbiased
    # normaldistributed value
    logging.info("Finset: ")
    for fins, fin_cant in zip(rocket_components.fin_sets,
                              random_parameters.fin_cants()):
        fins.setCantAngle(fin_cant)
        logging.info("{} with cant angle {:6.2f}°".format(fins.getName(),
                                                          math.degrees(fins.getCantAngle())))
    # There can be more than one parachute -> add unbiased
    # normaldistributed value
    logging.info("Parachutes: ")
    for parachute, parachute_cd in zip(
            rocket_components.parachutes, random_parameters.parachutes_cd()):
        parachute.setCD(max([parachute_cd, 0.]))
        logging.info("{} with CD {:6.2f}".format(
            parachute.getName(),
            parachute.getCD()))

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
    logging.info("External components: ")
    for ext_comp, roughness_random in zip(
            rocket_components.external_components, random_parameters.roughnesses()):
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
        logging.info("{} with finish {}".format(ext_comp,
                     ext_comp.getFinish()))


def get_simulation_parameters(open_rocket_helper, sim, rocket_components,
                              random_parameters, randomize=True):
    """Collect all global simulation parameters for export"""
    logging.info("Used global parameters...")
    options = sim.getOptions()
    tilt = math.degrees(options.getLaunchRodAngle())
    azimuth = math.degrees(options.getLaunchRodDirection())
    if randomize:
        thrust_factor = random_parameters.thrust_factor()
    else:
        thrust_factor = 1.

    logging.info("Launch rail tilt    = {:6.2f}°".format(tilt))
    logging.info("Launch rail azimuth = {:6.2f}°".format(azimuth))
    logging.info("Thrust factor = {:6.2f}".format(thrust_factor))

    # stage sepration
    mcid = options.getMotorConfigurationID()
    rocket = options.getRocket()
    separationDelays = []
    for stage in rocket_components.stages:
        separationEventConfiguration = stage.getStageSeparationConfiguration().get(mcid)
        separationDelays.append(
            separationEventConfiguration.getSeparationDelay())
        logging.info("Separation delay of stage {} = {:6.2f}s".format(
            stage, separationDelays[-1]))

    fin_cants = []
    for fins in rocket_components.fin_sets:
        fin_cants.append(math.degrees(fins.getCantAngle()))

    # There can be more than one parachute -> add unbiased
    # normaldistributed value
    parachute_cds = []
    for parachute in rocket_components.parachutes:
            parachute_cds.append(parachute.getCD())

    Parameters = collections.namedtuple("Parameters", [
        "tilt",
        "azimuth",
        "thrust_factor",
        "separation_delays",
        "fin_cants",
        "parachute_cds"])
    return Parameters(
        tilt = tilt,
        azimuth = azimuth,
        thrust_factor = thrust_factor,
        separation_delays = separationDelays,
        fin_cants = fin_cants,
        parachute_cds = parachute_cds)


def run_simulation(orh, sim, config, parameterset, branch_number=0):
    """Run a single simulation and return the results.

    :return:
        A tuple containing (landing_point_world, launch_position,
        apogee, trajectory, landing_point_cartesian
    """
    wind_listener = WindListener(config["WindModel"]["DataFile"], "linear")
    motor_listener = MotorListener(
        parameterset.thrust_factor,
        float(config["Propulsion"]["NozzleCrossSection"]))

    orh.run_simulation(
        sim, listeners=(wind_listener,
                        motor_listener))

    # see if there were any warnings
    simulated_warnings = sim.getSimulatedWarnings()
    if simulated_warnings is not None:
        for warning in simulated_warnings:
            # yes, we know that we use listeners
            if not warning.equals(
                    orh.openrocket.aerodynamics.Warning.LISTENERS_AFFECTED):
                logging.info(warning)

    # was there any exception thrown? only main branch is considered
    simulation_exception_raised = False
    events = sim.getSimulatedData().getBranch(0).getEvents()
    for ev in events:
        if ev.getType() == ev.Type.EXCEPTION:
            simulation_exception_raised = True

    # extract trajectory in any case, but other results only if simulation
    # did not throw any exception
    trajectory = get_trajectory(orh, sim, branch_number)
    if not simulation_exception_raised:
        apogee = get_apogee(orh, sim, branch_number)
        landing_world, landing_cartesian = get_landing_site(
            orh, sim, branch_number)
        theta_ignition, altitude_ignition = get_ignition_tilt(
            orh, sim, branch_number)
    else:
        logging.warning(
            "Simulation threw an exception")
        apogee = []
        landing_world = []
        landing_cartesian = []
        theta_ignition = []
        altitude_ignition = []

    Results = collections.namedtuple("Results", [
        "landing_point_world",
        "apogee",
        "trajectory",
        "landing_point_cartesian",
        "theta_ignition",
        "altitude_ignition"])
    r = Results(landing_point_world=landing_world,
                apogee=apogee,
                trajectory=trajectory,
                landing_point_cartesian=landing_cartesian,
                theta_ignition=theta_ignition,
                altitude_ignition=altitude_ignition
                )
    return r


class MotorListener(orhelper.AbstractSimulationListener):
    """Override the thrust of the motor."""

    def __init__(self, thrust_factor, nozzle_cross_section_mm2):
        self.thrust_factor = thrust_factor
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


def plot_wind_model(wind_model_file):
    """Plot wind model file with different simulation methods"""
    try:
        # Read wind level model data from file
        data = np.loadtxt(wind_model_file)
    except (IOError, FileNotFoundError):
        logging.warning("Warning: wind model file '{}' ".format(wind_model_file)
                        + "not found! Default wind model will be used.")
        return

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

    logging.info("Input wind levels model data:")
    logging.info("Altitude (m) ")
    logging.info(altitudes_m)
    logging.info("Direction (°) ")
    logging.info(wind_directions_degree)
    logging.info("Wind speed (m/s) ")
    logging.info(wind_speeds_mps)

    fig, axs = plt.subplots(2)
    alt_rng = np.arange(-1, 20e3, 1e2)

    # spline via NE
    tck_north = scipy.interpolate.splrep(
        altitudes_m, wind_speeds_north_mps, s=0)
    tck_east = scipy.interpolate.splrep(altitudes_m, wind_speeds_east_mps, s=0)

    def interpolate_wind_speed_north_mps(alt): return scipy.interpolate.splev(
        alt, tck_north, der=0, ext=3)
    def interpolate_wind_speed_east_mps(alt): return scipy.interpolate.splev(
        alt, tck_east, der=0, ext=3)
    wind_north_plt_spline = interpolate_wind_speed_north_mps(alt_rng)
    wind_east_plt_spline = interpolate_wind_speed_east_mps(alt_rng)
    wind_speed_mps_ne_plt_spline = np.sqrt(wind_north_plt_spline * wind_north_plt_spline
                                           + wind_east_plt_spline * wind_east_plt_spline)
    wind_direction_rad_ne_plt_spline = np.unwrap(np.arctan2(
        wind_east_plt_spline, wind_north_plt_spline))

    # linear via NE
    interpolate_wind_speed_north_mps = scipy.interpolate.interp1d(
        altitudes_m, wind_speeds_north_mps, bounds_error=False,
        fill_value=(wind_speeds_north_mps[0], wind_speeds_north_mps[-1]))
    interpolate_wind_speed_east_mps = scipy.interpolate.interp1d(
        altitudes_m, wind_speeds_east_mps, bounds_error=False,
        fill_value=(wind_speeds_east_mps[0], wind_speeds_east_mps[-1]))
    wind_north_plt_linear = interpolate_wind_speed_north_mps(alt_rng)
    wind_east_plt_linear = interpolate_wind_speed_east_mps(alt_rng)
    wind_speed_mps_ne_plt_linear = np.sqrt(wind_north_plt_linear * wind_north_plt_linear
                                           + wind_east_plt_linear * wind_east_plt_linear)
    wind_direction_rad_ne_plt_linear = np.unwrap(np.arctan2(
        wind_east_plt_linear, wind_north_plt_linear))

    # spline, wind speed + dir direct
    tck_speed = scipy.interpolate.splrep(altitudes_m, wind_speeds_mps, s=0)
    wind_speed_plt_spline = scipy.interpolate.splev(
        alt_rng, tck_speed, der=0, ext=3)
    tck_dir = scipy.interpolate.splrep(altitudes_m, wind_directions_rad, s=0)
    wind_dir_plt_spline = np.unwrap(scipy.interpolate.splev(
        alt_rng, tck_dir, der=0, ext=3))

    # pchip, wind speed + dir direct
    wind_speed_plt_pchip_fct = scipy.interpolate.PchipInterpolator(
        altitudes_m, wind_speeds_mps, extrapolate=True)
    wind_speed_plt_pchip = wind_speed_plt_pchip_fct(alt_rng)
    wind_dir_plt_pchip_fct = scipy.interpolate.PchipInterpolator(
        altitudes_m, wind_directions_rad, extrapolate=True)
    wind_dir_plt_pchip = wind_dir_plt_pchip_fct(alt_rng)

    # pchip, north east
    wind_north_plt_pchip_fct = scipy.interpolate.PchipInterpolator(
        altitudes_m, wind_speeds_north_mps, extrapolate=True)
    wind_north_plt_pchip = wind_north_plt_pchip_fct(alt_rng)
    wind_east_plt_pchip_fct = scipy.interpolate.PchipInterpolator(
        altitudes_m, wind_speeds_east_mps, extrapolate=True)
    wind_east_plt_pchip = wind_east_plt_pchip_fct(alt_rng)
    wind_speed_mps_ne_plt_pchip = np.sqrt(wind_north_plt_pchip * wind_north_plt_pchip
                                          + wind_east_plt_pchip * wind_east_plt_pchip)
    wind_direction_rad_ne_plt_pchip = np.unwrap(np.arctan2(
        wind_east_plt_pchip, wind_north_plt_pchip))

    axs[0].plot(altitudes_m, wind_speeds_mps, 'o', label="model")
    axs[0].plot(
        alt_rng,
        wind_speed_mps_ne_plt_linear,
        color="r",
        label="linear via NE")
    axs[0].plot(
        alt_rng,
        wind_speed_mps_ne_plt_spline,
        color="g",
        label="spline via NE")
    axs[0].plot(
        alt_rng,
        wind_speed_mps_ne_plt_pchip,
        color="b",
        label="pchip via NE")
    axs[0].plot(
        alt_rng,
        wind_speed_plt_spline,
        color="g",
        label="spline direct")
    axs[0].plot(alt_rng, wind_speed_plt_pchip, color="b", label="pchip direct")
    axs[0].set_ylabel("speed / ms")
    axs[0].legend()
    axs[1].plot(
        altitudes_m,
        np.degrees(wind_directions_rad),
        'o',
        label="model")
    axs[1].plot(
        alt_rng,
        np.degrees(
            wind_direction_rad_ne_plt_linear +
            2 *
            np.pi),
        color="r",
        label="linear via NE")
    axs[1].plot(
        alt_rng,
        np.degrees(
            wind_direction_rad_ne_plt_spline +
            2 *
            np.pi),
        color="g",
        label="spline via NE")
    axs[1].plot(
        alt_rng,
        np.degrees(
            wind_direction_rad_ne_plt_pchip +
            2 *
            np.pi),
        color="b",
        label="pchip via NE")
    axs[1].plot(
        alt_rng,
        np.degrees(wind_dir_plt_spline),
        color="g",
        label="spline direct")
    axs[1].plot(
        alt_rng,
        np.degrees(wind_dir_plt_pchip),
        color="b",
        label="pchip direct")
    axs[1].set_ylabel("direction / deg")
    axs[1].legend()

    plt.show()


class WindListener(orhelper.AbstractSimulationListener):
    """Set the wind speed as a function of altitude."""

    def __init__(self, wind_model_file="", interpolation_method="linear"):
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
        if interpolation_method == "linear":
            self.interpolate_wind_speed_north_mps = scipy.interpolate.interp1d(
                altitudes_m, wind_speeds_north_mps, bounds_error=False,
                fill_value=(wind_speeds_north_mps[0], wind_speeds_north_mps[-1]))
            self.interpolate_wind_speed_east_mps = scipy.interpolate.interp1d(
                altitudes_m, wind_speeds_east_mps, bounds_error=False,
                fill_value=(wind_speeds_east_mps[0], wind_speeds_east_mps[-1]))
        elif interpolation_method == "spline":
            tck_north = scipy.interpolate.splrep(
                altitudes_m, wind_speeds_north_mps, s=0)
            tck_east = scipy.interpolate.splrep(
                altitudes_m, wind_speeds_east_mps, s=0)
            self.interpolate_wind_speed_north_mps = lambda alt: scipy.interpolate.splev(
                alt, tck_north, der=0, ext=3)
            self.interpolate_wind_speed_east_mps = lambda alt: scipy.interpolate.splev(
                alt, tck_east, der=0, ext=3)
        elif interpolation_method == "pchip":
            print("pchip")
            self.interpolate_wind_speed_north_mps = scipy.interpolate.PchipInterpolator(
                altitudes_m, wind_speeds_north_mps, extrapolate=True)
            self.interpolate_wind_speed_east_mps = scipy.interpolate.PchipInterpolator(
                altitudes_m, wind_speeds_east_mps, extrapolate=True)
        else:
            raise ValueError(
                "Wrong interpolation method. Available are ´linear´, ´spline´, ´pchip´")

        # pchip does not have an option to constrain to min/max values
        self.constrain_altitude = lambda alt: max([altitudes_m[1],
                                                   min([alt, altitudes_m[-1]])])

    def preWindModel(self, status):
        """Set the wind coordinates at every simulation step."""
        if self._default_wind_model_is_used:
            return None

        position = status.getRocketPosition()
        wind_speed_north_mps = self.interpolate_wind_speed_north_mps(
            self.constrain_altitude(position.z))
        wind_speed_east_mps = self.interpolate_wind_speed_east_mps(
            self.constrain_altitude(position.z))
        logging.debug("Wind: alt {}m, N {}m/s, E {}m/s".format(position.z,
                                                               wind_speed_north_mps, wind_speed_east_mps))
        wind_speed_mps = math.sqrt(wind_speed_north_mps * wind_speed_north_mps
                                   + wind_speed_east_mps * wind_speed_east_mps)
        wind_direction_rad = math.atan2(
            wind_speed_east_mps, wind_speed_north_mps)

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
def get_apogee(open_rocket_helper, simulation, branch_number=0):
    """Return the apogee of the simulation as ``WorldCoordinate``."""
    FlightDataType = orhelper.FlightDataType
    data = open_rocket_helper.get_timeseries(simulation, [
        FlightDataType.TYPE_TIME,
        FlightDataType.TYPE_ALTITUDE,
        FlightDataType.TYPE_LONGITUDE,
        FlightDataType.TYPE_LATITUDE], branch_number)
    t = np.array(data[FlightDataType.TYPE_TIME])
    altitude = np.array(data[FlightDataType.TYPE_ALTITUDE])
    longitude = np.array(data[FlightDataType.TYPE_LONGITUDE])
    latitude = np.array(data[FlightDataType.TYPE_LATITUDE])
    def index_at(t): return (
        np.abs(data[FlightDataType.TYPE_TIME] - t)).argmin()

    # TODO how can I determine the stage triggering the flight event?
    events = open_rocket_helper.get_events(simulation)
    try:
        ct_apogee = len(events[orhelper.FlightEvent.APOGEE])
        logging.info('# apogee events found: {}'.format(ct_apogee))
        t_apogee = events[orhelper.FlightEvent.APOGEE][0]
    except BaseException:
        if EXCEPTION_FOR_MISSING_EVENTS:
            logging.warning(
                'no apogee event found, search maximum within trajectory')
            t_apogee = t[np.argmax(altitude)]
        else:
            logging.warning('no apogee event found, skip')
            t_apogee = []

    if t_apogee:
        apogee = open_rocket_helper.openrocket.util.WorldCoordinate(
            math.degrees(latitude[index_at(t_apogee)]),
            math.degrees(longitude[index_at(t_apogee)]),
            altitude[index_at(t_apogee)])
        logging.info(
            "Apogee at {:.1f}s: ".format(t_apogee)
            + "longitude {:.1f}°, ".format(apogee.getLatitudeDeg())
            + "latitude,{:.1f}°, ".format(apogee.getLongitudeDeg())
            + "altitude {:.1f}m".format(apogee.getAltitude()))
    else:
        apogee = []

    return apogee


def get_landing_site(open_rocket_helper, simulation, branch_number=0):
    """Return the landing site of the simulation as ``WorldCoordinate``."""
    FlightDataType = orhelper.FlightDataType
    data = open_rocket_helper.get_timeseries(simulation, [
        FlightDataType.TYPE_TIME,
        FlightDataType.TYPE_ALTITUDE,
        FlightDataType.TYPE_LONGITUDE,
        FlightDataType.TYPE_LATITUDE,
        FlightDataType.TYPE_POSITION_X,
        FlightDataType.TYPE_POSITION_Y], branch_number)
    t = np.array(data[FlightDataType.TYPE_TIME])
    altitude = np.array(data[FlightDataType.TYPE_ALTITUDE])
    longitude = np.array(data[FlightDataType.TYPE_LONGITUDE])
    latitude = np.array(data[FlightDataType.TYPE_LATITUDE])
    position_x = np.array(data[FlightDataType.TYPE_POSITION_X])
    position_y = np.array(data[FlightDataType.TYPE_POSITION_Y])
    def index_at(t): return (
        np.abs(data[FlightDataType.TYPE_TIME] - t)).argmin()

    # TODO how can I determine the stage triggering the flight event?
    events = open_rocket_helper.get_events(simulation)
    try:
        ct_landings = len(events[orhelper.FlightEvent.GROUND_HIT])
        logging.info('# ground hit events found: {}'.format(ct_landings))
        t_landing = events[orhelper.FlightEvent.GROUND_HIT][0]
    except BaseException:
        if EXCEPTION_FOR_MISSING_EVENTS:
            logging.warning('no landing found, use last time instant')
            t_landing = t[-1]
        else:
            logging.warning('no landing found, skip')
            t_landing = []

    if t_landing:
        landing_world = open_rocket_helper.openrocket.util.WorldCoordinate(
            math.degrees(latitude[index_at(t_landing)]),
            math.degrees(longitude[index_at(t_landing)]),
            altitude[index_at(t_landing)])
        landing_cartesian = open_rocket_helper.openrocket.util.Coordinate(
            position_x[index_at(t_landing)],
            position_y[index_at(t_landing)],
            altitude[index_at(t_landing)])
        logging.info(
            "Landing at {:.1f}s: ".format(t_landing)
            + "longitude {:.1f}°, ".format(landing_world.getLatitudeDeg())
            + "latitude,{:.1f}°, ".format(landing_world.getLongitudeDeg())
            + "altitude {:.1f}m, ".format(landing_world.getAltitude())
            + "pos_x {:.1f}m, ".format(landing_cartesian.x)
            + "pos_y {:.1f}m".format(landing_cartesian.y))
    else:
        landing_world = []
        landing_cartesian = []

    return landing_world, landing_cartesian


def get_ignition_tilt(open_rocket_helper, simulation, branch_number=0):
    """Return the tilt angle at ignition."""
    FlightDataType = orhelper.FlightDataType
    data = open_rocket_helper.get_timeseries(simulation, [
        FlightDataType.TYPE_TIME,
        FlightDataType.TYPE_ALTITUDE,
        FlightDataType.TYPE_ORIENTATION_THETA], branch_number)
    t = np.array(data[FlightDataType.TYPE_TIME])
    theta = np.array(data[FlightDataType.TYPE_ORIENTATION_THETA])
    altitude = np.array(data[FlightDataType.TYPE_ALTITUDE])
    def index_at(t): return (
        np.abs(data[FlightDataType.TYPE_TIME] - t)).argmin()

    events = open_rocket_helper.get_events(simulation)
    try:
        ct_ignitions = len(events[orhelper.FlightEvent.IGNITION])
        logging.info('# ignition events found: {}'.format(ct_ignitions))

        if ct_ignitions > 1:
            # normally we are interested in the latest ignition only
            t_ignition = events[orhelper.FlightEvent.IGNITION][ct_ignitions - 1]
            logging.info("Ignition at {:.1f}s: ".format(t_ignition))
            theta_ignition = math.degrees(theta[index_at(t_ignition)])
            altitude_ignition = altitude[index_at(t_ignition)]
            logging.info("theta {:.1f}°, ".format(theta_ignition)
                         + "altitude {:.1f}m, ".format(altitude_ignition))
        else:
            t_ignition = []
            theta_ignition = []
            phi_ignition = []
            altitude_ignition = []
    except BaseException:
        logging.warning('no ignition found')
        theta_ignition = []
        altitude_ignition = []

    return theta_ignition, altitude_ignition

# TODO: Return x, y or lat, lon depending on `coordinate_type`
# TODO: I guess we should also directly use these x, y values for things
# like apogee, landing points, launch point, etc.


def get_trajectory(open_rocket_helper, simulation, branch_number=0):
    """Return the x, y and altitude values of the rocket.

    :return:
        List of 3 arrays containing the values for x, y and altitude at
        each simulation step
    """
    FlightDataType = orhelper.FlightDataType
    data = open_rocket_helper.get_timeseries(simulation, [
        FlightDataType.TYPE_POSITION_X,
        FlightDataType.TYPE_POSITION_Y,
        FlightDataType.TYPE_ALTITUDE], branch_number)
    x = np.array(data[FlightDataType.TYPE_POSITION_X])
    y = np.array(data[FlightDataType.TYPE_POSITION_Y])
    altitude = np.array(data[FlightDataType.TYPE_ALTITUDE])
    return [x, y, altitude]


def print_statistics(results, general_parameters):
    """Print statistics of all simulation results."""
    landing_points = [r.landing_point_world for r in results]
    launch_point = general_parameters.launch_point
    geodetic_computation = general_parameters.geodetic_computation
    apogees = [r.apogee for r in results]
    ignitions_theta = [r.theta_ignition for r in results if r.theta_ignition]
    ignitions_altitude = [
        r.altitude_ignition for r in results if r.altitude_ignition]

    logging.debug("Results: distances in cartesian coordinates")
    distances = []
    bearings = []
    for landing_point in landing_points:
        if landing_point:
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
        if apogee:
            max_altitude.append(apogee.getAltitude())

    logging.debug("distances and bearings in polar coordinates")
    logging.debug(distances)
    logging.debug(bearings)

    # TODO how can one calculate the 2pi-safe statistics of the bearing
    print("---")
    print("Apogee: {:.1f}m ± {:.2f}m ".format(
        np.mean(max_altitude), np.std(max_altitude)))
    print(
        "Rocket landing zone {:.1f}m ± {:.2f}m ".format(
            np.mean(distances), np.std(distances))
        + "bearing {:.1f}° ± {:.1f}° ".format(
            np.degrees(np.mean(bearings)), np.degrees(np.std(bearings))))
    if ignitions_theta:
        print("Ignition at altitude: {:.1f}m ± {:.2f}m ".format(
            np.mean(ignitions_altitude), np.std(ignitions_altitude)))
        print(" at tilt angle: {:.1f}° ± {:.2f}° ".format(
            np.mean(ignitions_theta), np.std(ignitions_theta)))

    print("Based on {} valid simulation(s) of {}.".format(
        len(distances), len(landing_points)))


def export_results_csv(results, parametersets,
                   general_parameters, output_filename):
    """Create csv with all simulation results and its global parameterset."""
    with open(output_filename + ".csv", 'w', newline='') as csvfile:
        resultwriter = csv.writer(csvfile, delimiter=',')
        resultwriter.writerow(["launch tilt / deg", " launch azimuth / deg",
                               " thrust_factor / 1", " stage separation delay / s",
                               " fin cant / deg", " parachute CD / 1",
                               " landing lat / deg", " landing lon / deg",
                               " landing x / m", " landing y /m", " apogee / m",
                               " ignition theta / deg", " ignition altitude / m"])
        for r, p in zip(results, parametersets):
            if r.landing_point_world:
                # valid solution
                resultwriter.writerow([
                    "%.2f" % p.tilt, 
                    "%.2f" % p.azimuth,
                    "%.2f" % p.thrust_factor,
                    p.separation_delays,
                    p.fin_cants,
                    p.parachute_cds,
                    "%.6f" % r.landing_point_world.getLatitudeDeg(),
                    "%.6f" % r.landing_point_world.getLongitudeDeg(),
                    "%.2f" % r.landing_point_cartesian.x,
                    "%.2f" % r.landing_point_cartesian.y,
                    "%.2f" % r.apogee.getAltitude(),
                    "%.2f" % r.theta_ignition,
                    "%.2f" % r.altitude_ignition])
            elif r.apogee:
                resultwriter.writerow([
                    p.tilt,
                    p.azimuth,
                    p.thrust_factor,
                    p.separation_delays,
                    p.fin_cants,
                    p.parachute_cds,
                    0, 0, 0, 0,
                    r.apogee.getAltitude(), 0, 0])
            else:
                resultwriter.writerow([
                    p.tilt,
                    p.azimuth,
                    p.thrust_factor,
                    p.separation_delays,
                    p.fin_cants,
                    p.parachute_cds,
                    0, 0, 0, 0, 0, 0, 0])

def export_results_kml(results, parametersets,
                   general_parameters, output_filename):
    """Create kml with all landing positions. """
    kml = simplekml.Kml()
    style = simplekml.Style()
    style.labelstyle.color = simplekml.Color.yellow  # color the text
    style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'
    
    pnt = kml.newpoint(name="Launch")
    pnt.coords = [(
        general_parameters.launch_point.getLongitudeDeg(),
        general_parameters.launch_point.getLatitudeDeg())]

    for r in results:
        if r.landing_point_world:
            pnt = kml.newpoint()
            pnt.coords = [(
                r.landing_point_world.getLongitudeDeg(),
                r.landing_point_world.getLatitudeDeg())]
            pnt.style = style
    kml.save(output_filename + "_landingscatter.kml")


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
def create_plots(results, output_filename, general_parameters,
                 plot_coordinate_type="flat", results_are_shown=False):
    """Create, store and optionally show the plots of the results.

    :raise ValueError:
        If Coordinate type is not "flat" or "wgs84".
    """
    def to_array_world(coordinate):
        if coordinate:
            return np.array([coordinate.getLatitudeDeg(),
                            coordinate.getLongitudeDeg(),
                            coordinate.getAltitude()])

    def to_array_cartesian(coordinate):
        if coordinate:
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
            [to_array_world(r.landing_point_world) for r in results if r.landing_point_world])
        landing_points_cartesian = np.array(
            [to_array_cartesian(r.landing_point_cartesian) for r in results if r.landing_point_cartesian])
        launch_point = to_array_world(general_parameters.launch_point)
        geodetic_computation = general_parameters.geodetic_computation
        apogees = np.array([to_array_world(r.apogee)
                           for r in results if r.apogee])
        trajectories = [r.trajectory for r in results]
        ignitions_theta = [r.theta_ignition for r in results]
        ignitions_altitude = [r.altitude_ignition for r in results]

    n_simulations = apogees.shape[0]
    if n_simulations < 1:
        logging.warning("No landing points found")
        return

    fig = plt.figure(constrained_layout=True)
    # TODO: Increase pad on the right
    spec = mpl.gridspec.GridSpec(nrows=1, ncols=2, figure=fig,
                                 width_ratios=[1.5, 1],)
    ax_trajectories = fig.add_subplot(spec[0, 0], projection='3d')
    ax_landing_points = fig.add_subplot(spec[0, 1])

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

    if len(x) > 2:
        confidence_ellipse(x, y, ax_lps, n_std=1, label=r"$1\sigma$",
                           edgecolor=colors[2], ls=linestyles[0])
        confidence_ellipse(x, y, ax_lps, n_std=2, label=r"$2\sigma$",
                           edgecolor=colors[1], ls=linestyles[1])
        confidence_ellipse(x, y, ax_lps, n_std=3, label=r"$3\sigma$",
                           edgecolor=colors[0], ls=linestyles[2])

    ax_lps.legend()
    ax_lps.ticklabel_format(useOffset=False, style="plain")

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
        # TODO trajectory might be outside of landing point scatter plot
        # ax_trajectories.set_xlim(xlim)
        # ax_trajectories.set_ylim(ylim)
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
    plt.savefig(output_filename + "_diana.pdf")
    if results_are_shown:
        plt.show()

    figHist = plt.figure(constrained_layout=True)
    spec = mpl.gridspec.GridSpec(nrows=1, ncols=3, figure=figHist,
                                 width_ratios=[1, 1, 1])
    ax_apogees = figHist.add_subplot(spec[0, 0])
    ax_ignition_tilt = figHist.add_subplot(spec[0, 1])
    ax_ignition_altitude = figHist.add_subplot(spec[0, 2])

    # Histogram of apogee altitudes
    ax_apogees.set_title("Apogees")
    n_bins = int(round(1 + 3.322 * math.log(n_simulations), 0))
    ax_apogees.hist(apogees[:, 2], bins=n_bins, orientation="vertical", ec="k")
    ax_apogees.set_xlabel("Altitude in m")
    ax_apogees.set_ylabel("Number of Simulations")

    # Histogram of tilt at ignition
    ax_ignition_tilt.set_title("Last Ignition Event")
    n_bins = int(round(1 + 3.322 * math.log(n_simulations), 0))
    ax_ignition_tilt.hist(
        ignitions_theta,
        bins=n_bins,
        orientation="vertical",
        ec="k")
    ax_ignition_tilt.set_xlabel("Tilt in °")
    ax_ignition_tilt.set_ylabel("Number of Simulations")
    # Histogram of altitude at ignition
    ax_ignition_altitude.set_title("Last Ignition Event")
    n_bins = int(round(1 + 3.322 * math.log(n_simulations), 0))
    ax_ignition_altitude.hist(
        ignitions_altitude,
        bins=n_bins,
        orientation="vertical",
        ec="k")
    ax_ignition_altitude.set_xlabel("Altitude in m")
    ax_ignition_altitude.set_ylabel("Number of Simulations")

    # Save and show the figure
    plt.suptitle("Statistics of {} Simulations".format(n_simulations))
    plt.savefig(output_filename + "_stats.pdf")
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

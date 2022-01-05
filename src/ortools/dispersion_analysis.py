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
import pyproj
import cycler
import simplekml
import os
import sys
import csv
import math
import scipy.interpolate
import scipy.stats
from scipy.spatial.transform import Rotation as R
import time
import logging
import collections

from shapely.geometry import Point, Polygon

DIANA_RELEASE = "1.2.0"

# -- plot options
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
line_color_map_jet = mpl.cm.jet

# -- global settings
STANDARD_PRESSURE = 101325.0  # The standard air pressure (1.01325 bar)
METERS_PER_DEGREE_LATITUDE = 111325.0
METERS_PER_DEGREE_LONGITUDE_EQUATOR = 111050.0
# wind model interpolation, can be ´linear´, ´spline´, ´pchip´
WIND_MODEL_INTERPOLATION = "linear"

# debug options
PLOTS_ARE_TESTED = False
WINDMODEL_TEST = False
EXCEPTION_FOR_MISSING_EVENTS = False


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
@click.option("--wind", "-w", is_flag=True, default=False, show_default=True,
              help="Visualize the wind model.")
@click.option("--show", "-s", is_flag=True, default=False, show_default=True,
              help="Show the plots.")
@click.option("--remove-outliers", "-r", is_flag=True, default=False, show_default=True,
              help="Skip results with landing points being outside of the three-sigma region. Confidence ellipses will be re-calculated after skipping.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed results on screen.")
def diana(directory, filename, config, output,
          plot_coordinate_type, wind, show, verbose, remove_outliers):
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
    # TODO: Define default values for all parameters of the .ini file)
    config.read(config_file_path)
    make_paths_in_config_absolute(config, config_file_path)

    timestamp = time.strftime("%y%m%d%H%M%S", time.localtime(t0))
    output_filename = output or "dispersion_analysis_" + timestamp
    results_are_shown = show
    ork_file_path = config["General"]["OrkFile"]
    print(f"This is diana v{DIANA_RELEASE}")
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

    if WINDMODEL_TEST or wind:
        if (config.has_section("WindModel")
            and config.has_option(
                "WindModel", "DataFile")):
            plot_wind_model(
                config["WindModel"]["DataFile"],
                WIND_MODEL_INTERPOLATION)
        else:
            raise ValueError("No wind model is provided")
        return

    if PLOTS_ARE_TESTED:
        create_plots(
            [],
            output_filename,
            general_parameters,
            plot_coordinate_type,
            results_are_shown,
            True)
        return

    with orhelper.OpenRocketInstance() as instance:
        t1 = time.time()
        orh = orhelper.Helper(instance)
        i_simulation = int(config["General"]["SimulationIndex"])
        sim = get_simulation(orh, ork_file_path, i_simulation)

        # get global simulation parameters
        options = sim.getOptions()
        geodetic_computation = options.getGeodeticComputation()
        # Put warning if the geodetic computation is not WGS84
        logging.info(
            "Geodetic computation {} found.".format(geodetic_computation))
        if not geodetic_computation == geodetic_computation.WGS84:
            logging.warning("Geodetic computation {} is not recommended for landing scatter plots. Use WGS84 for highest accuracy.".format(
                geodetic_computation))

        launch_point = orh.openrocket.util.WorldCoordinate(
            options.getLaunchLatitude(),
            options.getLaunchLongitude(),
            options.getLaunchAltitude())
        logging.info("Launch Point {} found.".format(launch_point))

        rocket = options.getRocket()
        num_stages = rocket.getChildCount()
        print("Rocket has {} stage(s):".format(num_stages))
        stage_names = [r.getName() for r in rocket.getChildren()]
        logging.info(stage_names)

        GeneralParameters = collections.namedtuple("GeneralParameters", [
            "launch_point",
            "geodetic_computation",
            "num_stages",
            "stage_names"])
        general_parameters = GeneralParameters(
            launch_point=launch_point,
            geodetic_computation=geodetic_computation,
            num_stages=num_stages,
            stage_names=stage_names)

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
                parameterset = get_simulation_parameters(
                    orh, sim, rocket_components, random_parameters, True)
            else:
                print(
                    "with nominal parameters. if set, wind-model and pressure-increased thrust is applied")
                parameterset = get_simulation_parameters(
                    orh, sim, rocket_components, random_parameters, False)

            result = run_simulation(
                orh,
                sim,
                config,
                parameterset,
                general_parameters,
                f"{output_filename}_thrust_increase_{i}.csv")

            results.append(result)
            parametersets.append(parameterset)

        if remove_outliers:
            # calculate if landing point is within three sigma region
            lp_is_inside_3s = calc_lp_inside_3s(results, general_parameters)
        else:
            # set all true
            lp_is_inside_3s = [[True] * len(results)
                               for i in range(num_stages)]

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
            lp_is_inside_3s,
            output_filename)
        t3 = time.time()
        print("---")
        print("time for {} simulations = {:.1f}s".format(n_simulations,
                                                         t2 - t1))
        print("total execution time = {:.1f}s".format(t3 - t0))
        create_plots(
            results,
            output_filename,
            general_parameters,
            lp_is_inside_3s,
            plot_coordinate_type,
            results_are_shown)


def make_paths_in_config_absolute(config, config_file_path):
    """Turn all paths in the diana config file into absolute ones."""
    directory = os.path.dirname(os.path.abspath(config_file_path))
    if (config.has_section("WindModel")
        and config.has_option(
            "WindModel", "DataFile")):
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

    guidance_stddev = math.radians(float(config["LaunchRail"]["Guidance"]))
    azimuth_stddev = math.radians(float(config["LaunchRail"]["Azimuth"]))
    if config.has_option("LaunchRail", "TiltMean"):
        # override OR settings
        tilt_mean = math.radians(float(config["LaunchRail"]["TiltMean"]))
        options.setLaunchRodAngle(tilt_mean)
    else:
        tilt_mean = options.getLaunchRodAngle()
    tilt_stddev = math.radians(float(config["LaunchRail"]["Tilt"]))
    thrust_factor_stddev = float(config["Propulsion"]["ThrustFactor"])
    # truncated normal distribution for thrust factor
    if config.has_option("Propulsion", "ThrustFactorMin"):
        thrust_factor_min = float(config["Propulsion"]["ThrustFactorMin"])
    else:
        thrust_factor_min = -np.inf
    if config.has_option("Propulsion", "ThrustFactorMax"):
        thrust_factor_max = float(config["Propulsion"]["ThrustFactorMax"])
    else:
        thrust_factor_max = np.inf
    thrust_factor_a, thrust_factor_b = (
        thrust_factor_min - 1.) / thrust_factor_stddev, (thrust_factor_max - 1.) / thrust_factor_stddev

    fincant_stddev = math.radians(float(config["Aerodynamics"]["FinCant"]))
    parachute_cd_stddev = float(config["Aerodynamics"]["ParachuteCd"])
    roughness_stddev = float(config["Aerodynamics"]["Roughness"])

    Caxial_factor_stddev = float(config["Aerodynamics"]["CaxialFactor"])
    Cside_factor_stddev = float(config["Aerodynamics"]["CsideFactor"])
    CN_factor_stddev = float(config["Aerodynamics"]["CNFactor"])

    mcid = options.getMotorConfigurationID()
    rocket = options.getRocket()
    num_stages = rocket.getChildCount()
    stages = []
    stage_separation_delays_max = []
    stage_separation_delays_min = []
    for stage_nr in range(1, num_stages):
        stages.append(rocket.getChild(stage_nr))
        separationEventConfiguration = rocket.getChild(
            stage_nr).getStageSeparationConfiguration().get(mcid)
        separationDelay = separationEventConfiguration.getSeparationDelay()
        if (config.has_section("Staging")
            and config.has_option(
                "Staging", "StageSeparationDelayDeltaNeg")
                and config.has_option("Staging", "StageSeparationDelayDeltaPos")):
            # TODO set motor burnout of booster stage as lower minimum, and
            # motor ignition of upper stage as maximum
            stage_separation_delays_min.append(separationDelay + float(
                config["Staging"]["StageSeparationDelayDeltaNeg"]))
            stage_separation_delays_max.append(separationDelay + float(
                config["Staging"]["StageSeparationDelayDeltaPos"]))
        else:
            stage_separation_delays_max.append(separationDelay)
            stage_separation_delays_min.append(separationDelay)

    # MassCalculation
    refLen = rocket.getReferenceType().getReferenceLength(sim.getConfiguration())
    if (config.has_section("MassCalculation")):
        if(config.has_option(
                "MassCalculation", "CG")):
            CG_stddev = float(config["MassCalculation"]["CG"])
        else:
            CG_stddev = 0.
        if(config.has_option(
                "MassCalculation", "Mass")):
            mass_factor_stddev = float(config["MassCalculation"]["Mass"])
        else:
            mass_factor_stddev = 0.
    else:
        CG_stddev = 0.
        mass_factor_stddev = 0.

    # TODO: Move the components into its own function get_components()
    # or something. It makes no sense for this to be here.
    components_fin_sets = []
    components_parachutes = []
    components_external_components = []
    components_motors = []

    fin_cant_means = []
    parachute_cd_means = []
    component_roughness_means = []
    ignition_delays_max = []
    ignition_delays_min = []

    for component in orhelper.JIterator(rocket):
        logging.debug("Component {} of type {}".format(component.getName(),
                                                       type(component)))
        # fins can be
        #   orh.openrocket.rocketcomponent.FreeformFinSet
        #   orh.openrocket.rocketcomponent.TrapezoidFinSet
        #   orh.openrocket.rocketcomponent.EllipticalFinSet
        if (isinstance(component, orh.openrocket.rocketcomponent.FinSet)):
            logging.info(
                "Finset({}) with ".format(
                    component.getName()) +
                "cant angle {:6.2f}°".format(
                    math.degrees(
                        component.getCantAngle())))
            components_fin_sets.append(component)
            fin_cant_means.append(component.getCantAngle())

        # parachutes
        if isinstance(component, orh.openrocket.rocketcomponent.Parachute):
            logging.info("Parachute with drag surface diameter "
                         + "{:6.2f}m and ".format(component.getDiameter())
                         + "CD of {:6.2f}".format(component.getCD()))
            components_parachutes.append(component)
            parachute_cd_means.append(component.getCD())

        # components with external surfaces -> drag
        if isinstance(component,
                      orh.openrocket.rocketcomponent.ExternalComponent):
            logging.info("External component {} with finish {}".format(
                component, component.getFinish()))
            components_external_components.append(component)
            component_roughness_means.append(
                component.getFinish().getRoughnessSize())

        # motormounts with motors
        if (isinstance(component, orh.openrocket.rocketcomponent.MotorMount)):
            if component.isMotorMount():
                components_motors.append(component)
                ignition_delay = component.getIgnitionConfiguration().get(mcid).getIgnitionDelay()
                logging.info(
                    "Motor mount in stage {}".format(
                        component.getStageNumber()) +
                    " with delay {}s".format(ignition_delay))

                if (
                    config.has_section("Propulsion") and config.has_option(
                        "Propulsion",
                        "IgnitionDelayDeltaNeg") and config.has_option(
                        "Propulsion",
                        "IgnitionDelayDeltaPos") and component.getStageNumber() < (
                        num_stages -
                        1)):  # do not change first stage ignition
                    ignition_delays_min.append(ignition_delay + float(
                        config["Propulsion"]["IgnitionDelayDeltaNeg"]))
                    ignition_delays_max.append(ignition_delay + float(
                        config["Propulsion"]["IgnitionDelayDeltaPos"]))
                else:
                    ignition_delays_min.append(ignition_delay)
                    ignition_delays_max.append(ignition_delay)

    logging.info("Initial launch rail tilt    = {:6.2f}°".format(
        math.degrees(tilt_mean)))
    logging.info("Initial launch rail azimuth = {:6.2f}°".format(
        math.degrees(azimuth_mean)))

    RocketComponents = collections.namedtuple("RocketComponents", [
        "fin_sets",
        "parachutes",
        "external_components",
        "stages",
        "motors"])
    rng = np.random.default_rng()
    RandomParameters = collections.namedtuple("RandomParameters", [
        "tilt",
        "azimuth",
        "thrust_factor",
        "stage_separation_delays",
        "fin_cants",
        "parachutes_cd",
        "roughnesses",
        "ignition_delays",
        "CG_shift",
        "mass_factor",
        "Caxial_factor",
        "CN_factor",
        "Cside_factor",
        "guidance"])
    return (
        RocketComponents(
            fin_sets=components_fin_sets,
            parachutes=components_parachutes,
            external_components=components_external_components,
            stages=stages,
            motors=components_motors),
        RandomParameters(
            tilt=lambda: rng.normal(tilt_mean, tilt_stddev),
            azimuth=lambda: rng.normal(azimuth_mean, azimuth_stddev),
            thrust_factor=lambda: scipy.stats.truncnorm.rvs(thrust_factor_a, thrust_factor_b, 1., thrust_factor_stddev),
            stage_separation_delays=lambda: [rng.uniform(min,
                                                         max) for (min, max) in zip(
                stage_separation_delays_min,
                stage_separation_delays_max)],
            fin_cants=lambda: [rng.normal(mean, fincant_stddev)
                               for mean in fin_cant_means],
            parachutes_cd=lambda: [
                rng.normal(
                    mean,
                    parachute_cd_stddev) for mean in parachute_cd_means],
            roughnesses=lambda: [rng.normal(mean, roughness_stddev) for mean in component_roughness_means],
            ignition_delays=lambda: [rng.uniform(min,
                                                 max) for (min, max) in zip(
                ignition_delays_min,
                ignition_delays_max)],
            CG_shift=lambda: refLen * rng.normal(0., CG_stddev),
            mass_factor=lambda: rng.normal(1, mass_factor_stddev),
            Caxial_factor=lambda: rng.normal(1, Caxial_factor_stddev),
            CN_factor=lambda: rng.normal(1, CN_factor_stddev),
            Cside_factor=lambda: rng.normal(1, Cside_factor_stddev),
            guidance=lambda: rng.normal(0, guidance_stddev)))


def randomize_simulation(open_rocket_helper, sim, rocket_components,
                         random_parameters):
    """Set simulation parameters to random samples.
    return the global parameter set"""
    logging.info("Randomize variables...")
    options = sim.getOptions()

    # --- calculate launch rod angles
    tilt = random_parameters.tilt()
    azimuth = random_parameters.azimuth()
    dx = random_parameters.guidance()
    dy = random_parameters.guidance()

    # calculate direction of launch rod in xyz space, with lenght=1
    v = [0, 0, 1]
    Rx = R.from_euler('x', -tilt)
    Rz = R.from_euler('z', -azimuth)
    pts = Rz.apply(Rx.apply(v))
    # superimpose pointing direction (azimuth+tilt) with improper guidance
    # within rail
    Rx = R.from_euler('x', dx)
    Ry = R.from_euler('y', dy)
    pts_plane = Ry.apply(Rx.apply(pts))
    # calc tilt and azimuth again
    tilt_result = np.arctan2(
        np.sqrt(
            pts_plane[0] *
            pts_plane[0] +
            pts_plane[1] *
            pts_plane[1]),
        pts_plane[2])
    azimuth_result = np.arctan2(pts_plane[0], pts_plane[1])

    options.setLaunchRodAngle(tilt_result)
    # Otherwise launch rod direction cannot be altered
    options.setLaunchIntoWind(False)
    options.setLaunchRodDirection(azimuth_result)

    mcid = options.getMotorConfigurationID()
    rocket = options.getRocket()
    num_stages = rocket.getChildCount()
    # set stage sepration
    for (
            stage,
            stage_separation_delay) in zip(
            rocket_components.stages,
            random_parameters.stage_separation_delays()):
        separationEventConfiguration = stage.getStageSeparationConfiguration().get(mcid)
        logging.info(
            "Set separation delay of stage {}".format(stage))
        separationEventConfiguration.setSeparationDelay(
            stage_separation_delay)

    # set motor ignition
    for (motor, ignition_delay) in zip(
            rocket_components.motors, random_parameters.ignition_delays()):
        ignitionConfiguration = motor.getIgnitionConfiguration().get(mcid)
        logging.info(
            "Set ignition delay of stage {}".format(motor.getStage()))
        ignitionConfiguration.setIgnitionDelay(
            ignition_delay)

    # There can be more than one finset -> add unbiased
    # normaldistributed value
    logging.info("Finset: ")
    for fins, fin_cant in zip(rocket_components.fin_sets,
                              random_parameters.fin_cants()):
        fins.setCantAngle(fin_cant)
        logging.info(
            "{} with cant angle {:6.2f}°".format(
                fins.getName(), math.degrees(
                    fins.getCantAngle())))
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
        mass_factor = random_parameters.mass_factor()
        CG_shift = random_parameters.CG_shift()
        Caxial_factor = random_parameters.Caxial_factor()
        CN_factor = random_parameters.CN_factor()
        Cside_factor = random_parameters.Cside_factor()
    else:
        thrust_factor = 1.
        mass_factor = 1.
        CG_shift = 0.
        Caxial_factor = 1.
        CN_factor = 1.
        Cside_factor = 1.

    logging.info("Launch rail tilt    = {:6.2f}°".format(tilt))
    logging.info("Launch rail azimuth = {:6.2f}°".format(azimuth))
    logging.info("Thrust factor = {:6.2f}".format(thrust_factor))
    logging.info("Mass factor = {:6.2f}".format(mass_factor))
    logging.info("CG shift = {:6.2f}m".format(CG_shift))
    logging.info("Caxial factor = {:6.2f}".format(Caxial_factor))
    logging.info("CN factor = {:6.2f}".format(CN_factor))
    logging.info("Cside factor = {:6.2f}".format(Cside_factor))

    mcid = options.getMotorConfigurationID()
    rocket = options.getRocket()

    # stage sepration
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

    # motor ignition
    ignitionDelays = []
    for motor in rocket_components.motors:
        ignitionDelays.append(
            motor.getIgnitionConfiguration().get(mcid).getIgnitionDelay())
        logging.info("Ignition delay of stage {} = {:6.2f}s".format(
            motor.getStage(), ignitionDelays[-1]))

    Parameters = collections.namedtuple("Parameters", [
        "tilt",
        "azimuth",
        "thrust_factor",
        "separation_delays",
        "fin_cants",
        "parachute_cds",
        "ignition_delays",
        "mass_factor",
        "CG_shift",
        "Caxial_factor",
        "CN_factor",
        "Cside_factor"])
    return Parameters(
        tilt=tilt,
        azimuth=azimuth,
        thrust_factor=thrust_factor,
        separation_delays=separationDelays,
        fin_cants=fin_cants,
        parachute_cds=parachute_cds,
        ignition_delays=ignitionDelays,
        mass_factor=mass_factor,
        CG_shift=CG_shift,
        Caxial_factor=Caxial_factor,
        CN_factor=CN_factor,
        Cside_factor=Cside_factor)


def run_simulation(
        orh,
        sim,
        config,
        parameterset,
        general_parameters,
        fname_motor):
    """Run a single simulation and return the results.

    :return:
        A tuple containing (landing_point_world, launch_position,
        apogee, trajectory, landing_point_cartesian
    """
    mass_calculation_listener = MassCalculationListener(
        parameterset.mass_factor, parameterset.CG_shift)
    aerodynamic_forces_listener = AerodynamicForcesListener(
        parameterset.Caxial_factor, parameterset.CN_factor, parameterset.Cside_factor)
    motor_listener = MotorListener(
        parameterset.thrust_factor,
        float(config["Propulsion"]["NozzleCrossSection"]),
        fname_motor)

    if (config.has_section("WindModel")
        and config.has_option(
            "WindModel", "DataFile")):
        wind_listener = WindListener(
            config["WindModel"]["DataFile"],
            WIND_MODEL_INTERPOLATION)
        orh.run_simulation(
            sim, listeners=(wind_listener,
                            mass_calculation_listener,
                            aerodynamic_forces_listener,
                            motor_listener))
    else:
        orh.run_simulation(
            sim, listeners=(mass_calculation_listener,
                            aerodynamic_forces_listener,
                            motor_listener))

    # see if there were any warnings
    simulated_warnings = sim.getSimulatedWarnings()
    if simulated_warnings is not None:
        for warning in simulated_warnings:
            # yes, we know that we use listeners
            if not warning.equals(
                    orh.openrocket.aerodynamics.Warning.LISTENERS_AFFECTED):
                logging.info(warning)

    # process results
    launch_point = general_parameters.launch_point
    geodetic_computation = general_parameters.geodetic_computation
    branch_ct = sim.getSimulatedData().getBranchCount()
    num_stages = general_parameters.num_stages
    logging.debug(
        f'Branch count {branch_ct}, vs stage count {general_parameters.num_stages}')

    # prepare lists for all stages' results
    Results = collections.namedtuple("Results", [
        "landing_point_world",
        "apogee",
        "trajectory",
        "landing_point_cartesian",
        "theta_ignition",
        "altitude_ignition",
        "distance",
        "bearing"], defaults=(None,) * 8)

    simulation_exception_raised = False
    r = []

    if branch_ct == num_stages:
        for branch_nr in range(branch_ct):
            # was there any exception thrown? -> skip results fo this stage
            events = sim.getSimulatedData().getBranch(branch_nr).getEvents()
            for ev in events:
                # how do I get the source of the exception?
                # OR code: currentStatus.getWarnings()
                if ev.getType() == ev.Type.EXCEPTION:
                    simulation_exception_raised = True
                    break

            # extract results only if simulation
            # did not throw any exception
            if not simulation_exception_raised and branch_nr < branch_ct:
                # for branch_nr in
                # range(sim.getSimulatedData().getBranchCount()):
                apogee = get_apogee(orh, sim, branch_nr)
                landing_world, landing_cartesian = get_landing_site(
                    orh, sim, branch_nr)
                # normally we are interested in the latest ignition only ->
                # extract from last stage only
                theta_ignition, altitude_ignition = get_ignition_tilt(
                    orh, sim)
                trajectory = get_trajectory(orh, sim, branch_nr)

                # calculate distance in m and bearing in radians
                if geodetic_computation == geodetic_computation.FLAT:
                    distance, bearing = compute_distance_and_bearing_flat(
                        launch_point, landing_world)
                else:
                    geodesic = pyproj.Geod(ellps="WGS84")
                    fwd_azimuth, back_azimuth, distance = geodesic.inv(
                        launch_point.getLongitudeDeg(),
                        launch_point.getLatitudeDeg(),
                        landing_world.getLongitudeDeg(),
                        landing_world.getLatitudeDeg())
                    bearing = np.radians(fwd_azimuth)

                if bearing < 0.:
                    bearing = bearing + 2 * math.pi

                r.append(Results(landing_point_world=landing_world,
                                 apogee=apogee,
                                 trajectory=trajectory,
                                 landing_point_cartesian=landing_cartesian,
                                 theta_ignition=theta_ignition,
                                 altitude_ignition=altitude_ignition,
                                 distance=distance,
                                 bearing=bearing
                                 ))
            else:
                logging.warning(
                    f"Simulation of stage {branch_nr} threw an exception")
                r.append(Results())

    else:
        logging.warning(
            "Not the same number of simulation branches than stages")
        r = Results()

    return r


class MotorListener(orhelper.AbstractSimulationListener):
    """Override the thrust of the motor."""

    def __init__(self, thrust_factor, nozzle_cross_section_mm2, fname):
        self.thrust_factor = thrust_factor
        self.nozzle_cross_section = nozzle_cross_section_mm2 * 1e-6
        logging.info("Nozzle cross section = {:6.2g}mm^2".format(
            nozzle_cross_section_mm2))
        self.pressure = STANDARD_PRESSURE
        self.fname = fname

        self.thrust = []
        self.pressures = []
        self.thrust_new = []
        self.thrust_increase = []

    def __del__(self):
        # create data file for thrust values if this feature is active
        if self.nozzle_cross_section > 0.:
            try:
                with open(self.fname, 'w', newline='') as csvfile:
                    resultwriter = csv.writer(csvfile, delimiter=',')
                    resultwriter.writerow([" STANDARD_PRESSURE / Pa",
                                           " pressure / Pa",
                                           " thrust_factor / 1",
                                           " thrust orig / N",
                                           " thrust new / N",
                                           " thrust increase / N",
                                           " thrust increase / %"])

                    for thrust, pressure, thrust_new, thrust_increase in zip(
                            self.thrust, self.pressures, self.thrust_new, self.thrust_increase):
                        if thrust > 0:
                            thrust_incr_percent = (
                                thrust_increase) / thrust * 100
                        else:
                            thrust_incr_percent = 0

                        resultwriter.writerow([
                            "%.2f" % STANDARD_PRESSURE,
                            "%.2f" % pressure,
                            "%.2f" % self.thrust_factor,
                            "%.2f" % thrust,
                            "%.2f" % thrust_new,
                            "%.2f" % thrust_increase,
                            "%.2f" % thrust_incr_percent])

            except BaseException:
                logging.warning('Could not write thrust file')

    def postAtmosphericModel(self, status, atmospheric_conditions):
        """Get the ambient pressure from the atmospheric model."""
        self.pressure = atmospheric_conditions.getPressure()

    def postSimpleThrustCalculation(self, status, thrust):
        """Return the adapted thrust."""

        thrust_increase = (
            STANDARD_PRESSURE - self.pressure) * self.nozzle_cross_section
        if thrust >= thrust_increase:
            # apply thrust_increase if motor is burning
            logging.debug("Thrust increase due to decreased ambient pressure "
                          + "= {:6.2f}N".format(thrust_increase))
        else:
            # no increase after motor burnout
            thrust_increase = 0

        # add thrust increase (if there is any) and apply thrust variation
        # factor
        thrust_new = self.thrust_factor * thrust + thrust_increase

        # save for thrust file
        self.thrust.append(thrust)
        self.thrust_new.append(thrust_new)
        self.thrust_increase.append(thrust_increase)
        self.pressures.append(self.pressure)

        return thrust_new


class MassCalculationListener(orhelper.AbstractSimulationListener):
    """Override the mass parameters of the rocket."""

    def __init__(self, mass_factor, CG_shift):
        # do something
        self.mass_factor = mass_factor
        self.CG_shift = CG_shift

    def postMassCalculation(self, status, mass_data):
        """"""
        # return RigidBody-object mass_data, will overwrite old mass_data
        # TODO: how to change mass_data?
        return mass_data


class AerodynamicForcesListener(orhelper.AbstractSimulationListener):
    """Override the aerodynamic parameters of the rocket."""

    def __init__(self, Caxial_factor, CN_factor, Cside_factor):
        # do something
        self.Caxial_factor = Caxial_factor
        self.Cside_factor = Cside_factor
        self.CN_factor = CN_factor

    def postAerodynamicCalculation(self, status, aerodynamic_forces):
        """"""
        # return AerodynamicForces-object aerodynamic_forces, will overwrite
        # old aerodynamic_forces
        aerodynamic_forces.setCaxial(
            aerodynamic_forces.getCaxial() *
            self.Caxial_factor)
        aerodynamic_forces.setCside(
            aerodynamic_forces.getCside() *
            self.Cside_factor)
        aerodynamic_forces.setCN(aerodynamic_forces.getCN() * self.CN_factor)
        return aerodynamic_forces


def plot_wind_model(wind_model_file, interpolation_method="linear"):
    """Plot wind model file at given interpolation methods"""
    try:
        # Read wind level model data from file
        data = np.loadtxt(wind_model_file)
    except (IOError, FileNotFoundError):
        logging.warning(
            "Warning: wind model file '{}' ".format(wind_model_file) +
            "not found! Default wind model will be used.")
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

    # pchip does not have an option to constrain to min/max values
    def constrain_altitude_array(alt):
        altitude_constrained = alt.copy()  # we need to create a copy
        altitude_constrained[altitude_constrained >
                             altitudes_m[-1]] = altitudes_m[-1]
        altitude_constrained[altitude_constrained <
                             altitudes_m[0]] = altitudes_m[0]
        return altitude_constrained

    fig, axs = plt.subplots(2)
    altitude_range = np.arange(-1, np.amax([20e3, np.amax(altitudes_m)]), 1e2)

    axs[0].plot(altitudes_m, wind_speeds_mps, 'o', label="model")
    axs[1].plot(
        altitudes_m,
        np.degrees(wind_directions_rad),
        'o',
        label="model")

    if interpolation_method == "linear":
        # linear via NE
        interpolate_wind_speed_north_mps = scipy.interpolate.interp1d(
            altitudes_m, wind_speeds_north_mps, bounds_error=False,
            fill_value=(wind_speeds_north_mps[0], wind_speeds_north_mps[-1]))
        interpolate_wind_speed_east_mps = scipy.interpolate.interp1d(
            altitudes_m, wind_speeds_east_mps, bounds_error=False,
            fill_value=(wind_speeds_east_mps[0], wind_speeds_east_mps[-1]))
        wind_north_plt_linear = interpolate_wind_speed_north_mps(
            altitude_range)
        wind_east_plt_linear = interpolate_wind_speed_east_mps(altitude_range)
        wind_speed_mps_ne_plt_linear = np.sqrt(
            wind_north_plt_linear *
            wind_north_plt_linear +
            wind_east_plt_linear *
            wind_east_plt_linear)
        wind_direction_rad_ne_plt_linear = np.unwrap(np.arctan2(
            wind_east_plt_linear, wind_north_plt_linear))

        axs[0].plot(
            altitude_range,
            wind_speed_mps_ne_plt_linear,
            color="r",
            label="linear via NE")
        axs[1].plot(
            altitude_range,
            np.degrees(
                wind_direction_rad_ne_plt_linear +
                2 *
                np.pi),
            color="r",
            label="linear via NE")
    elif interpolation_method == "spline":
        # spline via NE
        tck_north = scipy.interpolate.splrep(
            altitudes_m, wind_speeds_north_mps, s=0)
        tck_east = scipy.interpolate.splrep(
            altitudes_m, wind_speeds_east_mps, s=0)

        def interpolate_wind_speed_north_mps(
            alt): return scipy.interpolate.splev(alt, tck_north, der=0, ext=3)
        def interpolate_wind_speed_east_mps(
            alt): return scipy.interpolate.splev(alt, tck_east, der=0, ext=3)
        wind_north_plt_spline = interpolate_wind_speed_north_mps(
            altitude_range)
        wind_east_plt_spline = interpolate_wind_speed_east_mps(altitude_range)
        wind_speed_mps_ne_plt_spline = np.sqrt(
            wind_north_plt_spline *
            wind_north_plt_spline +
            wind_east_plt_spline *
            wind_east_plt_spline)
        wind_direction_rad_ne_plt_spline = np.unwrap(np.arctan2(
            wind_east_plt_spline, wind_north_plt_spline))
        axs[0].plot(
            altitude_range,
            wind_speed_mps_ne_plt_spline,
            color="g",
            label="spline via NE")
        axs[1].plot(
            altitude_range,
            np.degrees(
                wind_direction_rad_ne_plt_spline +
                2 *
                np.pi),
            color="g",
            label="spline via NE")
    elif interpolation_method == "pchip":
        # pchip, north east
        wind_north_plt_pchip_fct = scipy.interpolate.PchipInterpolator(
            altitudes_m, wind_speeds_north_mps, extrapolate=True)
        wind_north_plt_pchip = wind_north_plt_pchip_fct(
            constrain_altitude_array(altitude_range))
        wind_east_plt_pchip_fct = scipy.interpolate.PchipInterpolator(
            altitudes_m, wind_speeds_east_mps, extrapolate=True)
        wind_east_plt_pchip = wind_east_plt_pchip_fct(
            constrain_altitude_array(altitude_range))
        wind_speed_mps_ne_plt_pchip = np.sqrt(
            wind_north_plt_pchip *
            wind_north_plt_pchip +
            wind_east_plt_pchip *
            wind_east_plt_pchip)
        wind_direction_rad_ne_plt_pchip = np.unwrap(np.arctan2(
            wind_east_plt_pchip, wind_north_plt_pchip))
        axs[0].plot(
            altitude_range,
            wind_speed_mps_ne_plt_pchip,
            color="b",
            label="pchip via NE")
        axs[1].plot(
            altitude_range,
            np.degrees(
                wind_direction_rad_ne_plt_pchip +
                2 *
                np.pi),
            color="b",
            label="pchip via NE")
    elif interpolation_method == "spline_direct":
        # spline, wind speed + dir direct
        tck_speed = scipy.interpolate.splrep(altitudes_m, wind_speeds_mps, s=0)
        wind_speed_plt_spline = scipy.interpolate.splev(
            altitude_range, tck_speed, der=0, ext=3)
        tck_dir = scipy.interpolate.splrep(
            altitudes_m, wind_directions_rad, s=0)
        wind_dir_plt_spline = np.unwrap(scipy.interpolate.splev(
            altitude_range, tck_dir, der=0, ext=3))
        axs[0].plot(
            altitude_range,
            wind_speed_plt_spline,
            color="g",
            label="spline direct")
        axs[1].plot(
            altitude_range,
            np.degrees(wind_dir_plt_spline),
            color="g",
            label="spline direct")
    elif interpolation_method == "pchip_direct":
        # pchip, wind speed + dir direct
        wind_speed_plt_pchip_fct = scipy.interpolate.PchipInterpolator(
            altitudes_m, wind_speeds_mps, extrapolate=True)
        wind_speed_plt_pchip = wind_speed_plt_pchip_fct(
            constrain_altitude_array(altitude_range))
        wind_dir_plt_pchip_fct = scipy.interpolate.PchipInterpolator(
            altitudes_m, wind_directions_rad, extrapolate=True)
        wind_dir_plt_pchip = wind_dir_plt_pchip_fct(
            constrain_altitude_array(altitude_range))
        axs[0].plot(
            altitude_range,
            wind_speed_plt_pchip,
            color="b",
            label="pchip direct")
        axs[1].plot(
            altitude_range,
            np.degrees(wind_dir_plt_pchip),
            color="b",
            label="pchip direct")
    else:
        raise ValueError(
            "Wrong interpolation method. Available are ´linear´, ´spline´, ´pchip´")

    axs[0].set_ylabel("speed / ms")
    axs[0].legend()
    axs[1].set_ylabel("direction / deg")
    axs[1].legend()
    axs[1].set_xlabel("altitude / m")

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
            logging.warning(
                "Warning: wind model file '{}' ".format(wind_model_file) +
                "not found! Default wind model will be used.")
            return
        except (ValueError):
            self._default_wind_model_is_used = True
            logging.warning(
                "Aloft data is incorrect! `altitudes_m`, "
                + "`wind_direction_std_degree`, "
                + "`wind_directions_degree` and `wind_speeds_mps` must be "
                + "of the same length.")
            return

        try:
            # Convert data
            altitudes_m = data[:, 0]
            wind_directions_degree = data[:, 1]
            wind_speeds_mps = data[:, 2]
            wind_direction_std_degree = data[:, 3]
        except BaseException:
            self._default_wind_model_is_used = True
            logging.warning(
                "Warning: wind model file '{}' ".format(wind_model_file) +
                "is malformated.")
            return

        # npyio catches this, but just to be sure
        if (len(altitudes_m) != len(wind_directions_degree)
                or len(altitudes_m) != len(wind_speeds_mps)
                or len(altitudes_m) != len(wind_direction_std_degree)):
            self._default_wind_model_is_used = True
            logging.warning(
                "Aloft data is incorrect! `altitudes_m`, "
                + "`wind_directions_degree` and `wind_speeds_mps` must be "
                + "of the same length.")
            return

        self._default_wind_model_is_used = False
        wind_directions_rad = np.radians(wind_directions_degree)
        wind_direction_std_rad = np.radians(wind_direction_std_degree)

        logging.debug("Input wind levels model data:")
        logging.debug("Altitude (m) ")
        logging.debug(altitudes_m)
        logging.debug("Direction (°) ")
        logging.debug(wind_directions_degree)
        logging.debug("Wind speed (m/s) ")
        logging.debug(wind_speeds_mps)
        logging.debug("direction std deviation (°) ")
        logging.debug(wind_direction_std_degree)

        # randomize direction. wind speed is "randomized" by turbulence model
        # of OR
        rng = np.random.default_rng()
        wind_directions_randomized_rad = [
            rng.normal(
                direction, direction_std) for direction, direction_std in zip(
                wind_directions_rad, wind_direction_std_rad)]
        wind_speeds_north_mps = wind_speeds_mps * \
            np.cos(wind_directions_randomized_rad)
        wind_speeds_east_mps = wind_speeds_mps * \
            np.sin(wind_directions_randomized_rad)

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
        self.constrain_altitude = lambda alt: max(
            [altitudes_m[0], min([alt, altitudes_m[-1]])])

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
        conditions = status.getSimulationConditions()
        wind_model = conditions.getWindModel()
        logging.debug(
            "Intensity {:.1e}, stddev {:.1e}, average {:.1e}".format(
                wind_model.getTurbulenceIntensity(),
                wind_model.getStandardDeviation(),
                wind_model.getAverage()))


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

    t_apogee = t[np.argmax(altitude)]

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


def get_ignition_tilt(open_rocket_helper, simulation):
    """Return the tilt angle at ignition."""
    # normally we are interested in the latest ignition only
    FlightDataType = orhelper.FlightDataType
    data = open_rocket_helper.get_timeseries(simulation, [
        FlightDataType.TYPE_TIME,
        FlightDataType.TYPE_ALTITUDE,
        FlightDataType.TYPE_ORIENTATION_THETA], 0)
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

    launch_point = general_parameters.launch_point
    geodetic_computation = general_parameters.geodetic_computation

    for stage_nr in range(general_parameters.num_stages):
        landing_points = [
            r[stage_nr].landing_point_world for r in results if r[stage_nr] and r[stage_nr].landing_point_world]
        max_altitude = [r[stage_nr].apogee.getAltitude() for r in results if r[stage_nr]
                        and r[stage_nr].apogee]
        ignitions_theta = [
            r[stage_nr].theta_ignition for r in results if r[stage_nr] and r[stage_nr].theta_ignition]
        ignitions_altitude = [
            r[stage_nr].altitude_ignition for r in results if r[stage_nr] and r[stage_nr].altitude_ignition]
        distances = [
            r[stage_nr].distance for r in results if r[stage_nr] and r[stage_nr].distance]
        bearings = [
            r[stage_nr].bearing for r in results if r[stage_nr] and r[stage_nr].bearing]

        logging.debug("Results: distances in cartesian coordinates")
        logging.debug("distance and bearing in polar coordinates")
        logging.debug(distances)
        logging.debug(np.degrees(bearings))

        print(
            f"--- Stage number {stage_nr}: {general_parameters.stage_names[stage_nr]} ---")
        print("Apogee: {:.1f}m ± {:.2f}m ".format(
            np.mean(max_altitude), np.std(max_altitude)))
        print(
            "Rocket landing zone {:.1f}m ± {:.2f}m ".format(
                np.mean(distances),
                np.std(distances)) +
            "bearing {:.1f}° ± {:.1f}° ".format(
                np.degrees(
                    scipy.stats.circmean(bearings)),
                np.degrees(
                    scipy.stats.circstd(bearings))))
        if ignitions_theta:
            print("Ignition at altitude: {:.1f}m ± {:.2f}m ".format(
                np.mean(ignitions_altitude), np.std(ignitions_altitude)))
            print(" at tilt angle: {:.1f}° ± {:.2f}° ".format(
                np.mean(ignitions_theta), np.std(ignitions_theta)))

        print("Based on {} valid simulation(s).".format(
            len(distances)))


def export_results_csv(results, parametersets,
                       general_parameters, output_filename):
    """Create csv with all simulation results and its global parameterset."""
    # TODO split output of parameters to relevant stage
    for stage_nr in range(general_parameters.num_stages):
        if general_parameters.num_stages > 1:
            fname = f"{output_filename}_{stage_nr}.csv"
        else:
            fname = f"{output_filename}.csv"

        with open(fname, 'w', newline='') as csvfile:
            resultwriter = csv.writer(csvfile, delimiter=',')
            resultwriter.writerow(["launch tilt / deg",
                                   " launch azimuth / deg",
                                   " thrust_factor / 1",
                                   " stage separation delay / s",
                                   " ignition delay / s",
                                   " fin cant / deg",
                                   " parachute CD / 1",
                                   " Caxial factor / 1",
                                   " CN factor / 1",
                                   " Cside factor / 1",
                                   " landing lat / deg",
                                   " landing lon / deg",
                                   " landing x / m",
                                   " landing y / m",
                                   " landing distance / m",
                                   " landing bearing / deg",
                                   " apogee / m",
                                   " ignition theta / deg",
                                   " ignition altitude / m"])
            for rs, p in zip(results, parametersets):
                r = rs[stage_nr]

                if r and r.landing_point_world:
                    # valid solution
                    if r.theta_ignition:
                        # multi stage
                        resultwriter.writerow([
                            "%.2f" % p.tilt,
                            "%.2f" % p.azimuth,
                            "%.2f" % p.thrust_factor,
                            p.separation_delays,
                            p.ignition_delays,
                            p.fin_cants,
                            p.parachute_cds,
                            "%.2f" % p.Caxial_factor,
                            "%.2f" % p.CN_factor,
                            "%.2f" % p.Cside_factor,
                            "%.6f" % r.landing_point_world.getLatitudeDeg(),
                            "%.6f" % r.landing_point_world.getLongitudeDeg(),
                            "%.2f" % r.landing_point_cartesian.x,
                            "%.2f" % r.landing_point_cartesian.y,
                            "%.2f" % r.distance,
                            "%.2f" % np.degrees(r.bearing),
                            "%.2f" % r.apogee.getAltitude(),
                            "%.2f" % r.theta_ignition,
                            "%.2f" % r.altitude_ignition])
                    else:
                        # single stage
                        resultwriter.writerow([
                            "%.2f" % p.tilt,
                            "%.2f" % p.azimuth,
                            "%.2f" % p.thrust_factor,
                            p.separation_delays,
                            p.ignition_delays,
                            p.fin_cants,
                            p.parachute_cds,
                            "%.2f" % p.Caxial_factor,
                            "%.2f" % p.CN_factor,
                            "%.2f" % p.Cside_factor,
                            "%.6f" % r.landing_point_world.getLatitudeDeg(),
                            "%.6f" % r.landing_point_world.getLongitudeDeg(),
                            "%.2f" % r.landing_point_cartesian.x,
                            "%.2f" % r.landing_point_cartesian.y,
                            "%.2f" % r.distance,
                            "%.2f" % np.degrees(r.bearing),
                            "%.2f" % r.apogee.getAltitude(),
                            "%.2f" % 0,
                            "%.2f" % 0])
                elif r and r.apogee:
                    resultwriter.writerow([
                        "%.2f" % p.tilt,
                        "%.2f" % p.azimuth,
                        "%.2f" % p.thrust_factor,
                        p.separation_delays,
                        p.ignition_delays,
                        p.fin_cants,
                        p.parachute_cds,
                        "%.2f" % p.Caxial_factor,
                        "%.2f" % p.CN_factor,
                        "%.2f" % p.Cside_factor,
                        0, 0, 0, 0, 0, 0,
                        r.apogee.getAltitude(), 0, 0])
                else:
                    resultwriter.writerow([
                        "%.2f" % p.tilt,
                        "%.2f" % p.azimuth,
                        "%.2f" % p.thrust_factor,
                        p.separation_delays,
                        p.ignition_delays,
                        p.fin_cants,
                        p.parachute_cds,
                        "%.2f" % p.Caxial_factor,
                        "%.2f" % p.CN_factor,
                        "%.2f" % p.Cside_factor,
                        0, 0, 0, 0, 0, 0, 0, 0, 0])


def export_results_kml(results, parametersets,
                       general_parameters, lp_is_inside_3s, output_filename):
    """Create kml with all landing positions. """

    for stage_nr in range(general_parameters.num_stages):

        kml = simplekml.Kml()
        style = simplekml.Style()
        style.labelstyle.color = simplekml.Color.yellow  # color the text
        style.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png'

        # add mark for launch site
        pnt = kml.newpoint(name="Launch")
        pnt.coords = [(
            general_parameters.launch_point.getLongitudeDeg(),
            general_parameters.launch_point.getLatitudeDeg())]

        # add landing points
        valid_results = [results[i][stage_nr] for i in range(len(
            results)) if results[i][stage_nr] and results[i][stage_nr].landing_point_world and lp_is_inside_3s[stage_nr][i]]
        for r in valid_results:
            if r.landing_point_world:
                pnt = kml.newpoint()
                pnt.coords = [(
                    r.landing_point_world.getLongitudeDeg(),
                    r.landing_point_world.getLatitudeDeg())]
                pnt.style = style

        # add confidence ellipses
        def to_array_world(coordinate):
            if coordinate:
                return np.array([coordinate.getLatitudeDeg(),
                                coordinate.getLongitudeDeg(),
                                coordinate.getAltitude()])
        landing_points_world = np.array([to_array_world(r.landing_point_world)
                                        for r in valid_results])
        x = landing_points_world[:, 1]
        y = landing_points_world[:, 0]
        if len(x) > 2:
            ellipse1 = calc_confidence_ellipse(x, y, n_std=1)
            pol = kml.newpolygon(name='1 sigma confidence ellipse')
            pol.outerboundaryis = list(map(tuple, ellipse1))
            pol.style.linestyle.color = simplekml.Color.green
            pol.style.linestyle.width = 5
            pol.style.polystyle.color = simplekml.Color.changealphaint(
                100, simplekml.Color.green)

            ellipse3 = calc_confidence_ellipse(x, y, n_std=3)
            pol = kml.newpolygon(name='3 sigma confidence ellipse')
            pol.outerboundaryis = list(map(tuple, ellipse3))
            pol.style.linestyle.color = simplekml.Color.red
            pol.style.linestyle.width = 5
            pol.style.polystyle.color = simplekml.Color.changealphaint(
                100, simplekml.Color.red)

        if general_parameters.num_stages > 1:
            fname = f"{output_filename}_{stage_nr}_landingscatter.kml"
        else:
            fname = f"{output_filename}_landingscatter.kml"

        # save results
        kml.save(fname)


def compute_distance_and_bearing_flat(from_, to):
    """Return distance and bearing betweeen two points.

    Valid for flat earth approximation only.

    :arg WorldCoordinate from_:
        First coordinate
    :arg WorldCoordinate to:
        Second coordinate

    :return:
        A tuple containing (distance in m, bearing in rad)
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


def calc_lp_inside_3s(results, general_parameters):
    """Iterate over results and set flag if landing point is within 3s confidence ellipse
    the namedtuple ´results´ is immutable -> return new variable
    """

    lp_is_inside_3s = []
    for stage_nr in range(general_parameters.num_stages):
        landing_points_world = np.array([to_array_world(r[stage_nr].landing_point_world)
                                        for r in results if r[stage_nr] and r[stage_nr].landing_point_world])
        x = landing_points_world[:, 1]
        y = landing_points_world[:, 0]

        if len(x) <= 2:
            # we need more points to calc ellipse -> set true
            lp_is_inside_3s.append([True] * len(results))
        else:
            # check if points are in or out
            poly_ellipse_3s = Polygon(calc_confidence_ellipse(x, y, n_std=3))
            lp_is_inside_3s_stage = []
            for r in results:
                if r[stage_nr] and r[stage_nr].landing_point_world:
                    landing_point_world = to_array_world(
                        r[stage_nr].landing_point_world)
                    px = landing_point_world[1]
                    py = landing_point_world[0]
                    p = Point(px, py)
                    logging.debug(
                        f"p {p} is within 3s: {p.within(poly_ellipse_3s)}")
                    p_lp_is_inside_3s = p.within(poly_ellipse_3s)
                else:
                    p_lp_is_inside_3s = False
                lp_is_inside_3s_stage.append(p_lp_is_inside_3s)
            lp_is_inside_3s.append(lp_is_inside_3s_stage)

    return lp_is_inside_3s

# TODO: Try to refactor this ugly plotting function


def create_plots(
        results,
        output_filename_in,
        general_parameters,
        lp_is_inside_3s,
        plot_coordinate_type="flat",
        results_are_shown=False):
    """Create, store and optionally show the plots of the results.

    :raise ValueError:
        If Coordinate type is not "flat" or "wgs84".
    """
    nominal_trajectories = []
    ellipses_1s = []
    ellipses_3s = []
    num_stages = []

    for stage_nr in range(general_parameters.num_stages):
        num_stages.append(stage_nr)

        if general_parameters.num_stages > 1:
            output_filename = f"{output_filename_in}_{stage_nr}"
        else:
            output_filename = output_filename_in

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
            distances = rng.normal(1000, 1, n_simulations)
            bearings = rng.normal(1000, 1, n_simulations)
            ignitions_theta = rng.normal(1000, 1, n_simulations)
            ignitions_altitude = rng.normal(1000, 1, n_simulations)
            x = rng.normal(1000, 1, n_simulations)
            y = rng.normal(1000, 1, n_simulations)
            alt = rng.normal(1000, 1, n_simulations)
            trajectories = [x, y, alt]
        else:
            launch_point = to_array_world(general_parameters.launch_point)
            geodetic_computation = general_parameters.geodetic_computation

            # append valid results to lists
            valid_results = [results[i][stage_nr] for i in range(
                len(results)) if results[i][stage_nr] and lp_is_inside_3s[stage_nr][i]]

            landing_points_world = np.array([to_array_world(
                r.landing_point_world) for r in valid_results if r.landing_point_world])
            landing_points_cartesian = np.array([to_array_cartesian(
                r.landing_point_cartesian) for r in valid_results if r.landing_point_cartesian])
            apogees = np.array([to_array_world(r.apogee)
                               for r in valid_results if r.apogee])
            trajectories = [
                r.trajectory for r in valid_results]
            ignitions_theta = [
                r.theta_ignition for r in valid_results if r.theta_ignition]
            ignitions_altitude = [
                r.altitude_ignition for r in valid_results if r.altitude_ignition]
            distances = [
                r.distance for r in valid_results if r.distance]
            bearings = [
                r.bearing for r in valid_results if r.bearing]

        n_simulations = apogees.shape[0]
        if n_simulations < 1:
            logging.warning(f"No landing points found for stage {stage_nr}")
            continue

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
            # use world coordinates with OR's geodetic_computation
            # implementation
            x = landing_points_world[:, 1]
            y = landing_points_world[:, 0]
            x0 = launch_point.getLongitudeDeg()
            y0 = launch_point.getLatitudeDeg()
            ax_lps.set_xlabel("Longitude in °")
            ax_lps.set_ylabel("Latitude in °")
        else:
            raise ValueError(
                "Coordinate type {} is not supported for plotting! ".format(
                    plot_coordinate_type)
                + "Valid values are 'flat' and 'wgs84'.")
        ax_lps.plot(x0, y0, "bx", markersize=5, zorder=0, label="Launch")
        ax_lps.plot(x, y, "r.", markersize=3, zorder=1, label="Landing")

        colors_lps = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        linestyles = mpl.rcParams["axes.prop_cycle"].by_key()["linestyle"]

        if len(x) > 2:
            # use matplotlib, because ellipses are rendered more smoothly
            plot_confidence_ellipse(x, y, ax_lps, n_std=1, label=r"$1\sigma$",
                                    edgecolor=colors_lps[2], ls=linestyles[0])
            plot_confidence_ellipse(x, y, ax_lps, n_std=3, label=r"$3\sigma$",
                                    edgecolor=colors_lps[0], ls=linestyles[2])

        ax_lps.legend()
        ax_lps.ticklabel_format(useOffset=False, style="plain")

        # save nominal trajectory
        nominal_trajectories.append(trajectories[0])
        # Plot the trajectories
        ax_trajectories.set_title("Trajectories")
        colors = line_color_map(np.linspace(0, 1, len(trajectories)))
        if plot_coordinate_type == "flat":
            for trajectory, color in zip(trajectories, colors):
                if trajectory:
                    # valid trajectory
                    traj_x, traj_y, alt = trajectory
                    ax_trajectories.plot(
                        xs=traj_x, ys=traj_y, zs=alt, color=color, linestyle="-")
            # plot confidence ellipse, x+y are still the landing plots
            if len(x) > 2:
                zmin, zmax = ax_trajectories.get_zlim()
                ellipse_1s = calc_confidence_ellipse(x, y, n_std=1)
                ax_trajectories.plot(
                    ellipse_1s[:, 0], ellipse_1s[:, 1], zmin + 0 * ellipse_1s[:, 0], label=r"$1\sigma$",
                    color=colors_lps[2], ls=linestyles[0])
                ellipse_3s = calc_confidence_ellipse(x, y, n_std=3)
                ax_trajectories.plot(
                    ellipse_3s[:, 0], ellipse_3s[:, 1], zmin + 0 * ellipse_3s[:, 0], label=r"$3\sigma$",
                    color=colors_lps[0], ls=linestyles[2])
            else:
                ellipse_1s = None
                ellipse_3s = None

            ax_trajectories.ticklabel_format(useOffset=False, style="plain")
            # set the xspan and yspan identical
            xleft, xright = ax_trajectories.get_xlim()
            yleft, yright = ax_trajectories.get_ylim()
            xmid = (xleft + xright) / 2
            ymid = (yleft + yright) / 2
            max_distance = max(xright - xleft, yright - yleft)
            ax_trajectories.set_xlim(
                xmid - max_distance / 2,
                xmid + max_distance / 2)
            ax_trajectories.set_ylim(
                ymid - max_distance / 2,
                ymid + max_distance / 2)
            # set the xlim and ylim of the scatter plot identically
            ax_lps.set_xlim(
                xmid - max_distance / 2,
                xmid + max_distance / 2)
            ax_lps.set_ylim(
                ymid - max_distance / 2,
                ymid + max_distance / 2)

            ax_trajectories.set_xlabel("x in m")
            ax_trajectories.set_ylabel("y in m")
        elif plot_coordinate_type == "wgs84":
            # TODO plot trajectory in WGS84 instead of x,y
            for trajectory, color in zip(trajectories, colors):
                if trajectory:
                    x, y, alt = trajectory
                    ax_trajectories.plot(
                        xs=x, ys=y, zs=alt, color=color, linestyle="-")

            # TODO add some ellipses here as well
            ellipse_1s = None
            ellipse_3s = None
            ax_trajectories.ticklabel_format(useOffset=False, style="plain")
            ax_trajectories.set_xlabel("x in m")
            ax_trajectories.set_ylabel("y in m")
        else:
            raise ValueError(
                "Coordinate type {} is not supported for plotting! ".format(
                    plot_coordinate_type)
                + "Valid values are 'flat' and 'wgs84'.")
        ax_trajectories.set_zlabel("altitude in m")

        ellipses_1s.append(ellipse_1s)
        ellipses_3s.append(ellipse_3s)

        # Save and show the figure
        plt.suptitle("Dispersion Analysis of {} Simulations for {} ".format(
            n_simulations, general_parameters.stage_names[stage_nr]))
        plt.savefig(output_filename + "_diana.pdf")
        plt.savefig(output_filename + "_diana.png")
        if results_are_shown:
            plt.show()

        # Histograms
        n_bins = min(
            n_simulations, int(
                round(
                    1 + 3.322 * math.log(n_simulations), 0)))

        figHist = plt.figure(constrained_layout=True)
        spec = mpl.gridspec.GridSpec(nrows=1, ncols=3, figure=figHist,
                                     width_ratios=[1, 1, 1])
        ax_apogees = figHist.add_subplot(spec[0, 0])
        ax_distances = figHist.add_subplot(spec[0, 1])
        ax_bearings = figHist.add_subplot(spec[0, 2])

        # Histogram of apogee altitudes
        ax_apogees.set_title("Apogees")
        # histograms only make sense for more than one result
        if len(apogees[:, 2]) > 1:
            ax_apogees.hist(apogees[:, 2], bins=n_bins,
                            orientation="vertical", ec="k")
            ax_apogees.set_xlabel("Altitude in m")
            ax_apogees.set_ylabel("Number of Simulations")

        # Histogram of landing distance
        ax_distances.set_title("Landing Distances")
        # histograms only make sense for more than one result
        if len(distances) > 1:
            ax_distances.hist(
                distances,
                bins=n_bins,
                orientation="vertical",
                ec="k")
            ax_distances.set_xlabel("Distance in m")
            ax_distances.set_ylabel("Number of Simulations")

        # Histogram of landing bearing
        ax_bearings.set_title("Landing Bearing")
        # histograms only make sense for more than one result
        if len(bearings) > 1:
            ax_bearings.hist(
                np.degrees(bearings),
                bins=[
                    0,
                    45,
                    90,
                    135,
                    180,
                    225,
                    270,
                    315,
                    360],
                range=(
                    0,
                    360),
                orientation="vertical",
                ec="k")
            ax_bearings.set_xlabel("Bearing in °")
            ax_bearings.set_ylabel("Number of Simulations")

        # Save and show the figure
        plt.suptitle("Statistics of {} Simulations for {}".format(
            n_simulations, general_parameters.stage_names[stage_nr]))
        plt.savefig(output_filename + "_stats_general.pdf")
        plt.savefig(output_filename + "_stats_general.png")
        if results_are_shown:
            plt.show()

        figHistIgnition = plt.figure(constrained_layout=True)
        spec = mpl.gridspec.GridSpec(nrows=1, ncols=2, figure=figHistIgnition,
                                     width_ratios=[1, 1])
        ax_ignition_tilt = figHistIgnition.add_subplot(spec[0, 0])
        ax_ignition_altitude = figHistIgnition.add_subplot(spec[0, 1])

        # ignition stats for upper-most stage
        if stage_nr < 1 and general_parameters.num_stages > 1:
            # Histogram of tilt at ignition
            ax_ignition_tilt.set_title("Last Ignition Event")
            # histograms only make sense for more than one result
            if len(ignitions_theta) > 1:
                ax_ignition_tilt.hist(
                    ignitions_theta,
                    bins=n_bins,
                    orientation="vertical",
                    ec="k")
                ax_ignition_tilt.set_xlabel("Tilt in °")
                ax_ignition_tilt.set_ylabel("Number of Simulations")

            # Histogram of altitude at ignition
            ax_ignition_altitude.set_title("Last Ignition Event")
            # histograms only make sense for more than one result
            if len(ignitions_altitude) > 1:
                ax_ignition_altitude.hist(
                    ignitions_altitude,
                    bins=n_bins,
                    orientation="vertical",
                    ec="k")
                ax_ignition_altitude.set_xlabel("Altitude in m")
                ax_ignition_altitude.set_ylabel("Number of Simulations")

            # Save and show the figure
            plt.suptitle("Statistics of {} Simulations for {}".format(
                n_simulations, general_parameters.stage_names[stage_nr]))
            plt.savefig(output_filename + "_stats_ignition.pdf")
            plt.savefig(output_filename + "_stats_ignition.png")
            if results_are_shown:
                plt.show()

    # if it is a multistage rocket, plot nominal trajectories in a single plot
    if len(nominal_trajectories) > 1:
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
            x0 = 0
            y0 = 0
            ax_lps.set_xlabel(r"$\Delta x$ in m")
            ax_lps.set_ylabel(r"$\Delta y$ in m")
        elif plot_coordinate_type == "wgs84":
            # use world coordinates with OR's geodetic_computation
            # implementation
            x0 = launch_point.getLongitudeDeg()
            y0 = launch_point.getLatitudeDeg()
            ax_lps.set_xlabel("Longitude in °")
            ax_lps.set_ylabel("Latitude in °")
        ax_lps.plot(x0, y0, "bx", markersize=5, zorder=0, label="Launch")

        linestyles = mpl.rcParams["axes.prop_cycle"].by_key()["linestyle"]
        colors = line_color_map_jet(
            np.linspace(0, 1, len(nominal_trajectories)))

        for nominal_trajectory, ellipse_1s, ellipse_3s, color, stage_nr in zip(
                nominal_trajectories, ellipses_1s, ellipses_3s, colors, num_stages):

            if ellipse_1s is not None:
                ax_lps.plot(
                    ellipse_1s[:, 0], ellipse_1s[:, 1], label=f"Stage {stage_nr} " + r"$1\sigma$",
                    color=color, ls=linestyles[0])
                ax_lps.plot(
                    ellipse_3s[:, 0], ellipse_3s[:, 1], label=f"Stage {stage_nr} " + r"$3\sigma$",
                    color=color, ls=linestyles[2])

            # Plot the trajectories
            ax_trajectories.set_title("Trajectories")
            if plot_coordinate_type == "flat":
                traj_x, traj_y, alt = nominal_trajectory
                ax_trajectories.plot(
                    xs=traj_x,
                    ys=traj_y,
                    zs=alt,
                    color=color,
                    linestyle="-",
                    label=f"stage {stage_nr}")

                # plot confidence ellipse, x+y are still the landing plots
                zmin, zmax = ax_trajectories.get_zlim()
                if ellipse_1s is not None:
                    ax_trajectories.plot(
                        ellipse_1s[:, 0], ellipse_1s[:, 1], zmin + 0 * ellipse_1s[:, 0], label=r"$1\sigma$",
                        color=color, ls=linestyles[0])
                    ax_trajectories.plot(
                        ellipse_3s[:, 0], ellipse_3s[:, 1], zmin + 0 * ellipse_3s[:, 0], label=r"$3\sigma$",
                        color=color, ls=linestyles[2])

                # set the xspan and yspan identical
                xleft, xright = ax_trajectories.get_xlim()
                yleft, yright = ax_trajectories.get_ylim()
                xmid = (xleft + xright) / 2
                ymid = (yleft + yright) / 2
                max_distance = max(xright - xleft, yright - yleft)
                ax_trajectories.set_xlim(
                    xmid - max_distance / 2, xmid + max_distance / 2)
                ax_trajectories.set_ylim(
                    ymid - max_distance / 2, ymid + max_distance / 2)
                # set the xlim and ylim of the scatter plot identically
                ax_lps.set_xlim(
                    xmid - max_distance / 2, xmid + max_distance / 2)
                ax_lps.set_ylim(
                    ymid - max_distance / 2, ymid + max_distance / 2)
            elif plot_coordinate_type == "wgs84":
                # TODO plot trajectory in WGS84 instead of x,y
                x, y, alt = nominal_trajectory
                ax_trajectories.plot(
                    xs=x, ys=y, zs=alt, color=color, linestyle="-",
                    label=f"stage {stage_nr}")

        ax_lps.legend()
        ax_lps.ticklabel_format(useOffset=False, style="plain")

        ax_trajectories.set_zlabel("altitude in m")
        ax_trajectories.legend()

        if plot_coordinate_type == "flat":
            ax_trajectories.ticklabel_format(
                useOffset=False, style="plain")
            ax_trajectories.set_xlabel("x in m")
            ax_trajectories.set_ylabel("y in m")
        elif plot_coordinate_type == "wgs84":
            ax_trajectories.ticklabel_format(
                useOffset=False, style="plain")
            ax_trajectories.set_xlabel("x in m")
            ax_trajectories.set_ylabel("y in m")

        # Save and show the figure
        plt.suptitle(
            "Dispersion Analysis of {} Stages".format(
                general_parameters.num_stages))
        plt.savefig(output_filename_in + "_diana_all.pdf")
        plt.savefig(output_filename_in + "_diana_all.png")
        if results_are_shown:
            plt.show()


# TODO: Convert docstring style
def plot_confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Credits
    ----------
    https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html

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

# TODO: Convert docstring style


def calc_confidence_ellipse(x, y, n_std=3.0):
    """
    Calculate data of the covariance confidence ellipse of *x* and *y*.

    Credits
    ----------
    Algorithm from https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.


    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    2 by 36 tuple of coordinates for ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    radius_x = np.sqrt(1 + pearson)
    radius_y = np.sqrt(1 - pearson)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    x0 = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    y0 = np.mean(y)

    # 36 points for a smooth shape
    num_pts = 36

    # calculate points of confidence ellipse
    t = np.linspace(0, 2 * np.pi, num_pts)
    ellipse = np.array([radius_x * np.cos(t), radius_y * np.sin(t)])
    rotated_ellipse = rotation_matrix(45) @ ellipse
    scale, origin = np.asarray([scale_x, scale_y]), np.asarray([x0, y0])
    transformed_ellipse = (rotated_ellipse.T * scale + origin)
    return transformed_ellipse


def rotation_matrix(degree):
    theta = np.radians(degree)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


def get_object_methods(obj):
    """Return object methods for debugging/development"""
    object_methods = [method_name for method_name in dir(obj)
                      if callable(getattr(obj, method_name))]
    print(object_methods)


if __name__ == "__main__":
    diana()

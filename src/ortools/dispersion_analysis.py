import ortools.utility as utility

import orhelper
import numpy as np
import matplotlib as mpl
import matplotlib.patches
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
mpl.rcParams["figure.figsize"] = ((1920 - 160 - 30) / 5 / 25.4,
                                  (1080 - 90 - 50) / 5 / 25.4)
mpl.rcParams["figure.dpi"] = 254 / 2
mpl.rcParams["axes.unicode_minus"] = True
mpl.rcParams["axes.grid"] = True
mpl.rcParams["axes.prop_cycle"] = cycler.cycler(
    color=("#7570b3", "#d95f02", "#1b9e77"), linestyle=("-", "--", ":"))

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
@click.option("--coordinate-type", "-ct",
              type=click.Choice(["flat", "sphere"]),
              default="flat", show_default=True,
              help=("The type of coordinates used in the scatter plot of the "
                    "landing points."))
@click.option("--show", "-s", is_flag=True, default=False,
              help="Show the results on screen.")
def diana(directory, filename, config, output, coordinate_type, show):
    """Do a dispersion analysis of an OpenRocket simulation.

    A dispersion analysis runs multiple simulations with slightly
    varying parameters. The config file specifies which simulation
    parameters are varied and by how much as well as the total number of
    simulations run.

    Example usage:

        diana -d examples -o test.pdf -s
    """
    # TODO: Measure and print total execution time
    # TODO: Maybe put .ork file path in config file
    t0 = time.time()
    ork_file_path = filename or utility.find_latest_file(".ork", directory)
    config_file_path = config or utility.find_latest_file(".ini", directory)
    output_filename = output or "dispersion_analysis.pdf"
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
        create_plots([], output_filename, coordinate_type, results_are_shown)
        return

    with orhelper.OpenRocketInstance() as instance:
        t1 = time.time()
        orh = orhelper.Helper(instance)
        sim = get_simulation(
            orh, ork_file_path, int(config["General"]["SimulationIndex"]))
        random_parameters = set_up_random_parameters(sim, config)

        results = []
        n_simulations = int(config["General"]["NumberOfSimulations"])
        for i in range(n_simulations):
            print("Running simulation {:4} of {}".format(i + 1, n_simulations))
            randomize_simulation(sim, random_parameters)
            result = run_simulation(orh, sim, config, random_parameters)
            results.append(result)

        t2 = time.time()
        print_statistics(results)
        t3 = time.time()
        print("---")
        print("time for {} simulations = {:.1f}s".format(n_simulations,
                                                         t2 - t1))
        print("total execution time = {:.1f}s".format(t3 - t0))
        create_plots(results, output_filename, coordinate_type,
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


def set_up_random_parameters(sim, config):
    """Return a ``namedtuple`` containing all random parameters.

    The random parameters are actually lambdas that return a new random
    sample when called.
    """
    options = sim.getOptions()
    azimuth_mean = options.getLaunchRodDirection()
    azimuth_stddev = math.radians(float(config["LaunchRail"]["Azimuth"]))
    tilt_mean = options.getLaunchRodAngle()
    tilt_stddev = math.radians(float(config["LaunchRail"]["Tilt"]))
    thrust_factor_stddev = float(config["Propulsion"]["ThrustFactor"])

    print("Initial launch rail tilt = {:6.2f}°".format(
        math.degrees(tilt_mean)))
    print("Initial launch rail azimuth   = {:6.2f}°".format(
        math.degrees(azimuth_mean)))

    rng = np.random.default_rng()
    RandomParameters = collections.namedtuple("RandomParameters", [
        "tilt",
        "azimuth",
        "thrust_factor"])
    return RandomParameters(
        tilt=lambda: rng.normal(tilt_mean, tilt_stddev),
        azimuth=lambda: rng.normal(azimuth_mean, azimuth_stddev),
        thrust_factor=lambda: rng.normal(1, thrust_factor_stddev))


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


def run_simulation(orh, sim, config, random_parameters):
    """Run a single simulation and return the results.

    :return:
        A tuple containing (landing_point, launch_position,
        geodetic_computation, apogee)
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
    # TODO: Return results in a nicer way, using a dictionary or
    # namedtuple for example. This makes handling afterwards easier
    # since we don't have to know which result is at which index
    return (landing_point_listener.landing_points[0],
            launch_point_listener.launch_point,
            launch_point_listener.geodetic_computation,
            apogee)


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
        self.landing_points = []

    def endSimulation(self, status, simulation_exception):
        """Return the landing position from OpenRocket."""
        self.landing_points.append(status.getRocketWorldPosition())


class MotorListener(orhelper.AbstractSimulationListener):
    """Override the thrust of the motor."""

    def __init__(self, thrust_factor, nozzle_cross_section_mm2):
        self.thrust_factor = thrust_factor
        print("Used thrust factor = {:6.2f}".format(thrust_factor))
        self.nozzle_cross_section = nozzle_cross_section_mm2 * 1e-6
        print("Nozzle cross section = {:6.2g}mm^2".format(
            nozzle_cross_section_mm2))
        self.pressure = STANDARD_PRESSURE

    def postAtmosphericModel(self, status, atmospheric_conditions):
        """Get the ambient pressure from the atmospheric model."""
        self.pressure = atmospheric_conditions.getPressure()

    def postSimpleThrustCalculation(self, status, thrust):
        """Return the adapted thrust."""
        # FIXME: thrust_increase is not used appart for logging
        thrust_increase = (
            STANDARD_PRESSURE - self.pressure) * self.nozzle_cross_section
        logging.debug("Thrust increase due to decreased ambient pressure "
                      + "= {:6.2f}N".format(thrust_increase))
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
            print("Warning: wind model file '{}' ".format(wind_model_file)
                  + "not found! Default wind model will be used.")
            return

        self._default_wind_model_is_used = False
        altitudes_m = data[:, 0]
        wind_directions_degree = data[:, 1]
        wind_speeds_mps = data[:, 2]

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

        # TODO: Which fill_values shall be used above the aloft
        # data? zero? last value? extrapolate?
        # TODO: This is not safe if direction rotates over 180deg ->
        # use x/y coordinates
        self.interpolate_wind_speed_mps = scipy.interpolate.interp1d(
            altitudes_m, wind_speeds_mps, bounds_error=False,
            fill_value=(wind_speeds_mps[0], wind_speeds_mps[-1]))
        wind_directions_rad = np.radians(wind_directions_degree)
        self.interpolate_wind_direction_rad = scipy.interpolate.interp1d(
            altitudes_m, wind_directions_rad, bounds_error=False,
            fill_value=(wind_directions_rad[0], wind_directions_rad[-1]))

    def preWindModel(self, status):
        """Set the wind coordinates at every simulation step."""
        if self._default_wind_model_is_used:
            return None

        position = status.getRocketPosition()
        wind_speed_mps = self.interpolate_wind_speed_mps(position.z)
        wind_direction_rad = self.interpolate_wind_direction_rad(
            position.z)

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
    # TODO: There should already be a package to convert lon, lat to x, y
    dx = ((to.getLongitudeDeg() - from_.getLongitudeDeg())
          * METERS_PER_DEGREE_LONGITUDE_EQUATOR)
    dy = ((to.getLatitudeDeg() - from_.getLatitudeDeg())
          * METERS_PER_DEGREE_LATITUDE)
    logging.debug("Longitude {:.1f}°, Latitude {:.1f}°".format(
        to.getLongitudeDeg(), to.getLatitudeDeg()))
    logging.debug("dx {:.1f}m, dy {:.1f}m".format(dx, dy))
    distance = math.sqrt(dx * dx + dy * dy)
    bearing = math.pi / 2. - math.atan2(dy, dx)
    if bearing > math.pi:
        bearing = bearing - 2 * math.pi
    return distance, bearing


# TODO: Try to refactor this ugly plotting function
def create_plots(results, output_filename, coordinate_type="flat",
                 results_are_shown=False):
    """Create, store and optionally show the plots of the results.

    :raise ValueError:
        If Coordinate type is not "flat" or "sphere".
    """
    def to_ndarray(coordinate):
        return np.array([coordinate.getLatitudeDeg(),
                        coordinate.getLongitudeDeg(),
                        coordinate.getAltitude()])

    if PLOTS_ARE_TESTED:
        # Test Data
        rng = np.random.default_rng()
        n_simulations = 1000
        lat = rng.normal(55, 2, n_simulations)
        lon = rng.normal(20, 1, n_simulations)
        landing_points = np.array([lat, lon]).T
        alt = rng.normal(15346, 17, n_simulations)
        apogees = np.zeros((n_simulations, 3))
        apogees[:, 2] = alt
    else:
        landing_points = np.array([to_ndarray(r[0]) for r in results])
        launch_point = to_ndarray(results[0][1])
        geodetic_computation = results[0][2]
        apogees = np.array([to_ndarray(r[3]) for r in results])

    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = mpl.rcParams["axes.prop_cycle"].by_key()["linestyle"]

    fig, axs = plt.subplots(1, 2, tight_layout=True,
                            gridspec_kw={"width_ratios": [3, 1]})
    axs = axs.ravel()

    # Scatter plot of landing coordinates
    axs[0].set_title("Landing Points")
    if coordinate_type == "flat":
        if geodetic_computation == geodetic_computation.WGS84:
            # OR simulated with WGS84 -> use flat projection
            logging.info("Use WGS84-to-flatearth projection")
            crs_wgs = pyproj.CRS("epsg:4326")
            x0 = 0
            y0 = 0
            cust = pyproj.Proj(
                "+proj=aeqd +lat_0={0} +lon_0={1} +datum=WGS84 +units=m".format(
                    launch_point[0], launch_point[1]))
            x, y = pyproj.transform(
                crs_wgs, cust, landing_points[:, 0], landing_points[:, 1])
        else:
            # OR simulated with FLAT coordinates -> take them directly
            x = ((landing_points[:, 1] - launch_point[1])
                 * METERS_PER_DEGREE_LONGITUDE_EQUATOR)
            y = ((landing_points[:, 0] - launch_point[0])
                 * METERS_PER_DEGREE_LATITUDE)
            x0 = 0
            y0 = 0
        axs[0].set_xlabel(r"$\Delta x$ in m")
        axs[0].set_ylabel(r"$\Delta y$ in m")
    elif coordinate_type == "sphere":
        x = landing_points[:, 1]
        y = landing_points[:, 0]
        x0 = launch_point[1]
        y0 = launch_point[0]
        axs[0].set_xlabel("Longitude in °")
        axs[0].set_ylabel("Latitude in °")
    else:
        raise ValueError(
            "Coordinate type {} is not supported! ".format(coordinate_type)
            + "Valid values are 'flat' and 'sphere'.")
    axs[0].plot(x0, y0, "bx", markersize=5, zorder=0, label="Launch")
    axs[0].plot(x, y, "r.", markersize=3, zorder=0, label="Landing")
    axs[0].axis("square")
    confidence_ellipse(x, y, axs[0], n_std=1, label=r"$1\sigma$",
                       edgecolor=colors[2], ls=linestyles[0])
    confidence_ellipse(x, y, axs[0], n_std=2, label=r"$2\sigma$",
                       edgecolor=colors[1], ls=linestyles[1])
    confidence_ellipse(x, y, axs[0], n_std=3, label=r"$3\sigma$",
                       edgecolor=colors[0], ls=linestyles[2])
    axs[0].legend()
    axs[0].ticklabel_format(useOffset=False, style="plain")

    # Histogram of apogee altitudes
    axs[1].set_title("Apogees")
    n_simulations = apogees.shape[0]
    n_bins = int(round(1 + 3.322 * math.log(n_simulations), 0))
    axs[1].hist(apogees[:, 2], bins=n_bins, orientation="horizontal",
                fc=colors[2], ec="k")
    axs[1].set_ylabel("Altitude in m")
    axs[1].set_xlabel("Number of Simulations")

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


if __name__ == "__main__":
    diana()

import ortools.utility as utility

import orhelper
import numpy
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


if __name__ == "__main__":
    cli()

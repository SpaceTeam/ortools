import os


def find_latest_file(extension, directory="."):
    """Find the file with the given extension that was modified last.

    :arg directory:
        Directory in which the file is searched for. Subdirectories are
        not included.

    :raise FileNotFoundError:
        If no file with the given extension is found in the given
        directory

    :return:
        Path to the found file
    """
    files = [os.path.join(directory, f) for f in os.listdir(directory)
             if os.path.isfile(os.path.join(directory, f))
             if f.endswith(extension)]
    if not files:
        raise FileNotFoundError("No file with extension {} ".format(extension)
                                + "was found in {}.".format(directory))
    return max(files, key=os.path.getmtime)

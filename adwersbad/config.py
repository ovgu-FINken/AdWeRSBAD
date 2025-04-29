from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

from pkg_resources import resource_filename

__all__ = ["config"]


def config(filename: str = "mydatabase.ini", section: str = "postgresql"):
    parser = ConfigParser(interpolation=ExtendedInterpolation())
    filepath = Path(resource_filename("adwersbad", filename))

    if not filepath.exists():
        raise FileNotFoundError(f"Config {filepath} does not exist")

    parser.read(filepath)
    param_dict = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            if param[0] == "username":
                continue
            param_dict[param[0]] = param[1]
    else:
        raise RuntimeError(
            f"Section {section} not found in config {filename} at: {filepath}"
        )

    return param_dict

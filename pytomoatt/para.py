from pathlib import Path

from yamlium import parse

from .utils.common import init_axis, str2val


class ATTPara:
    """Class for read and write parameter file with ``yaml`` format
    """
    def __init__(self, fname: str) -> None:
        """
        :param fname: Path to parameter file
        :type fname: str
        """
        self.fname = fname
        self.input_params = parse(Path(fname))

    def init_axis(self):
        dep, lat, lon, dd, dt, dp = init_axis(
            self.input_params['domain']['min_max_dep'],
            self.input_params['domain']['min_max_lat'],
            self.input_params['domain']['min_max_lon'],
            self.input_params['domain']['n_rtp'],
        )
        return dep, lat, lon, dd, dt, dp

    def update_param(self, key: str, value) -> None:
        """Update a parameter in the YAML file.

        :param key: The key of parameter file to be set. Use '.' to separate the keys.
        :type key: str
        """
        keys = key.split('.')
        param = self.input_params
        for k in keys[:-1]:
            if k not in param:
                # Assignment lets yamlium wrap the dict in its Mapping node.
                # dict.setdefault() bypasses yamlium's conversion logic.
                param[k] = {}
            param = param[k]
        param[keys[-1]] = str2val(value)

    def write(self, fname=None):
        """write

        :param fname: Path to output file, for None to overwrite input file, defaults to None
        :type fname: str, optional
        """
        if fname is None:
            fname = self.fname
        self.input_params.yaml_dump(fname)

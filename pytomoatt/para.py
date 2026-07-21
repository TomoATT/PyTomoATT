from ruamel.yaml import YAML
from .utils.common import init_axis, str2val

yaml = YAML()
yaml.default_flow_style = True

class ATTPara:
    """Class for read and write parameter file with ``yaml`` format
    """
    def __init__(self, fname: str) -> None:
        """
        :param fname: Path to parameter file
        :type fname: str
        """
        self.fname = fname
        with open(fname, encoding='utf-8') as f:
            file_data = f.read()
        self.input_params = yaml.load(file_data)

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
            param = param.setdefault(k, {})
        param[keys[-1]] = str2val(value)

    def write(self, fname=None):
        """write

        :param fname: Path to output file, for None to overwrite input file, defaults to None
        :type fname: str, optional
        """
        if fname is None:
            fname = self.fname
        with open(fname, 'w') as f:
            yaml.dump(self.input_params, f)

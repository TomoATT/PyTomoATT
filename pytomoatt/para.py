# import yaml
from ruamel.yaml import YAML
from .utils.common import init_axis

yaml = YAML()

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

    def write(self, fname=None):
        """write

        :param fname: Path to output file, for None to overwrite input file, defaults to None
        :type fname: str, optional
        """
        if fname is None:
            fname = self.fname
        with open(fname, 'w') as f:
            yaml.dump(self.input_params, f)

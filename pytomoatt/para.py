import yaml


class ATTPara:
    """Class for read and write parameter file with ``yaml`` format
    """
    def __init__(self, fname: str) -> None:
        """
        :param fname: Path to parameter file
        :type fname: str
        """
        self.fname = fname
        with open(fname) as f:
            file_data = f.read()
        self.input_params = yaml.load(file_data, Loader=yaml.Loader)

    def write(self, fname=None):
        """write

        :param fname: Path to output file, for None to overwrite input file, defaults to None
        :type fname: str, optional
        """
        if fname is None:
            fname = self.fname
        yaml.dump(self.input_params, fname, Dumper=yaml.Dumper)
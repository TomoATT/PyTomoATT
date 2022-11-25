import yaml


class ATTPara:
    def __init__(self, fname) -> None:
        self.fname = fname
        with open(fname) as f:
            file_data = f.read()
        self.input_params = yaml.load(file_data, Loader=yaml.Loader)

    def write(self, fname=None):
        if fname is None:
            fname = self.fname
        yaml.dump(self.input_params, Dumper=yaml.Dumper)
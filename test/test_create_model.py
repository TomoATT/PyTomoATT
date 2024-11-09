from pytomoatt.model import ATTModel
from pytomoatt.checkerboard import Checker
from pytomoatt.para import ATTPara
from os.path import join, dirname, abspath

class TestATTModel():
    para_fname: str = join(dirname(dirname(abspath(__file__))), 'examples', 'input_params.yml')
    out_fname: str = 'text_crust.h5'

    def test_crust(self):
        self.mod = ATTModel(self.para_fname)
        self.mod.grid_data_crust1(type='vp')
        self.mod.write(self.out_fname)

    def test_checkerboard01(self):
        cm = Checker(self.out_fname, self.para_fname)
        cm.checkerboard(2,2,2)

    def test_checkerboard02(self):
        cm = Checker(self.out_fname, self.para_fname)
        cm.checkerboard(
            3,3,3,
            lim_x=[-1, 1],
            lim_y=[-0.5, 0.5],
            lim_z=[10, 120]
        )

    def test_checkerboard03(self):
        cm = Checker(self.out_fname, self.para_fname)
        cm.checkerboard(
            4,4,4,
            ani_dir=70,
            lim_x=[-1, 1],
            lim_y=[-0.5, 0.5],
            lim_z=[10, 120]
        )

    def test_read_model(self):
        mod = ATTModel.read(self.out_fname, para_fname=self.para_fname)
        dataset = mod.to_xarray()
        dataset.interp_dep(60.3, field='vel')
        dataset.interp_sec(
            start_point=[mod.min_max_lon[0], mod.min_max_lat[1]],
            end_point=[mod.min_max_lon[1], mod.min_max_lat[1]], field='vel')

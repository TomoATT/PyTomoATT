import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
from .para import ATTPara
from .io.crustmodel import CrustModel
from .io.asciimodel import ASCIIModel


class Model():
    def __init__(self, para_fname='input_params.yml') -> None:
        self.para_fname = para_fname
        self.read_param()
        self.eta = np.zeros(self.n_rtp)
        self.xi = np.zeros(self.n_rtp)
        self.zeta = np.zeros(self.n_rtp)
        self.vel = np.zeros(self.n_rtp)

    def read_param(self):
        para = ATTPara(self.para_fname)
        self.n_rtp = para.input_params['domain']['n_rtp']
        self.min_max_dep = para.input_params['domain']['min_max_dep']
        self.min_max_lat = para.input_params['domain']['min_max_lat']
        self.min_max_lon = para.input_params['domain']['min_max_lon']

    def grid_data_crust1(self, type='vp'):
        cm = CrustModel()
        self.vel = cm.griddata(
            self.min_max_dep,
            self.min_max_lat,
            self.min_max_lon, 
            self.n_rtp, type=type
        )
        
    def grid_data_ascii(self, model_fname:str, **kwargs):
        am = ASCIIModel(model_fname)
        self.vel = am.read_ascii(**kwargs)
        self.vel = am.griddata(
            self.min_max_dep,
            self.min_max_lat,
            self.min_max_lon, 
            self.n_rtp,
        )

    def smooth(self, sigma=5):
        self.vel = gaussian_filter(self.vel, sigma)

    def write(self, out_fname=None):
        """Write to h5 file with TomoATT format.

        :param fname: file name of output model, defaults to 'model_crust1.0.h5'
        :type fname: str, optional
        """
        if out_fname is None:
            out_fname = 'Sub_CRUST1.0_{}_{:d}_{:d}_{:d}.h5'.format(self.type, *self.n_rtp)
        with h5py.File(out_fname, 'w') as f:
            f.create_dataset('eta', data=self.eta)
            f.create_dataset('xi', data=self.xi)
            f.create_dataset('zeta', data=self.zeta)
            f.create_dataset('vel', data=self.vel)

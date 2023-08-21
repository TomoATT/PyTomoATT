import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
from .para import ATTPara
from .io.crustmodel import CrustModel
from .io.asciimodel import ASCIIModel
from .attarray import Dataset
from .utils import init_axis, acosd, atand


class ATTModel():
    """Create initial model from external models
    """
    def __init__(self, para_fname='input_params.yml') -> None:
        """
        :param para_fname: Path to parameter file, defaults to 'input_params.yml'
        :type para_fname: str, optional
        """
        self.para_fname = para_fname
        self.read_param()
        self.eta = np.zeros(self.n_rtp)
        self.xi = np.zeros(self.n_rtp)
        self.zeta = np.zeros(self.n_rtp)
        self.vel = np.zeros(self.n_rtp)

    def read_param(self):
        """Read ``n_rtp``, ``min_max_dep``, ``min_max_lat`` and ``min_max_lon`` from ``para_fname``
        """
        para = ATTPara(self.para_fname)
        self.n_rtp = para.input_params['domain']['n_rtp']
        self.min_max_dep = para.input_params['domain']['min_max_dep']
        self.min_max_lat = para.input_params['domain']['min_max_lat']
        self.min_max_lon = para.input_params['domain']['min_max_lon']
        self.depths, self.latitudes, self.longitudes, _, _, _ = init_axis(
            self.min_max_dep, self.min_max_lat, self.min_max_lon, self.n_rtp
        )
        self.radius = 6371. - self.depths

    @classmethod
    def read(cls, model_fname: str, para_fname='input_params.yml'):
        """Read an exists model

        :param model_fname: Path to the exists model
        :type model_fname: str
        :param para_fname: Path to parameter file, defaults to 'input_params.yml'
        :type para_fname: str, optional
        """
        mod = cls(para_fname)
        f = h5py.File(model_fname)
        mod.vel = f['vel'][:]
        mod.xi = f['xi'][:]
        mod.eta = f['eta'][:]
        # mod.zeta = f['zeta'][:]
        mod._check_axis()
        if not ((mod.xi==0).all() and (mod.eta==0).all()):
            mod.to_ani()
        f.close()
        return mod
    
    def _check_axis(self):
        if self.vel.shape != tuple(self.n_rtp):
            raise ValueError('conflicting size of data and n_rtp in parameters')

    def to_ani(self):
        """Convert to anisotropic strength (epsilon) and azimuth (phi)
        """
        self.epsilon = np.sqrt(self.eta**2+self.xi**2)
        self.phi = np.zeros_like(self.epsilon)
        idx = np.where(self.xi <= 0)
        self.phi[idx] = 90 + 0.5*atand(self.eta[idx]/self.xi[idx])
        idx = np.where((self.xi > 0) & (self.eta <= 0))
        self.phi[idx] = 180 + 0.5*atand(self.eta[idx]/self.xi[idx])
        idx = np.where((self.xi > 0) & (self.eta > 0))
        self.phi[idx] = 0.5*atand(self.eta[idx]/self.xi[idx])

    def to_xarray(self):
        """Convert to xarray
        """
        data_dict = {}
        data_dict['vel'] = (["r", "t", "p"], self.vel)
        data_dict['xi'] = (["r", "t", "p"], self.xi)
        data_dict['eta'] = (["r", "t", "p"], self.eta)
        # data_dict['zeta'] = (["r", "t", "p"], self.zeta)
        if hasattr(self, 'epsilon') and hasattr(self, 'phi'):
            data_dict['epsilon'] = (["r", "t", "p"], self.epsilon)
            data_dict['phi'] = (["r", "t", "p"], self.phi)
        if hasattr(self, 'dlnv'):
            data_dict['dlnv'] = (["r", "t", "p"], self.dlnv)
        dataset = Dataset(
            data_dict,
            coords={
                'dep': (['r'], self.depths),
                'rad': (['r'], self.radius),
                'lat': (['t'], self.latitudes),
                'lon': (['p'], self.longitudes),
            }
        )
        return dataset

    def grid_data_crust1(self, type='vp'):
        """Grid data from CRUST1.0 model

        :param type: Specify velocity type of ``vp`` or ``vs``, defaults to 'vp'
        :type type: str, optional
        """
        cm = CrustModel()
        self.vel = cm.griddata(
            self.min_max_dep,
            self.min_max_lat,
            self.min_max_lon, 
            self.n_rtp, type=type
        )
        
    def grid_data_ascii(self, model_fname:str, **kwargs):
        """Grid data from custom model file in ASCII format

        :param model_fname: Path to model file
        :type model_fname: str
        :param usecols: Columns order by longitude, latitude, depth and velocity, defaults to [0, 1, 2, 3]
        :type usecols: list or tuple
        """
        am = ASCIIModel(model_fname)
        self.vel = am.read_ascii(**kwargs)
        self.vel = am.griddata(
            self.min_max_dep,
            self.min_max_lat,
            self.min_max_lon, 
            self.n_rtp,
        )

    def smooth(self, sigma=5.0):
        """Gaussian smooth the 3D velocity model

        :param sigma: Standard division of gaussian kernel, defaults to 5
        :type sigma: float, optional
        """
        self.vel = gaussian_filter(self.vel, sigma)

    def calc_dv_avg(self):
        """calculate anomalies relative to average velocity at each depth
        """
        self.dlnv = np.zeros_like(self.vel)
        for i, _ in enumerate(self.depths):
            avg = np.mean(self.vel[i, :, :])
            self.dlnv[i, :, :] = 100 * (self.vel[i, :, :] - avg)/avg

    def calc_dv(self, ref_mod_fname: str):
        """calculate anomalies relative to another model

        :param ref_mod_fname: Path to reference model
        :type ref_mod_fname: str
        """
        with h5py.File(ref_mod_fname) as f:
            ref_vel = f['vel'][:]
            if self.vel.shape != ref_vel.shape:
                raise ValueError('reference model should be in same size as input model')
            self.dlnv = 100*(self.vel - ref_vel)/ref_vel

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

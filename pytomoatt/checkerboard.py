import h5py
import numpy as np
from .utils import init_axis, sind, cosd


class Checker():
    """Create checkerboard model by adding perturbations on an exist model
    """
    def __init__(self, fname:str) -> None:
        self.model_file = fname
        with h5py.File(fname) as f:
            self.vel = f['vel'][:]
            self.eta = f['eta'][:]
            self.xi = f['xi'][:]
            self.zeta = f['zeta'][:]

    def init_axis(self, min_max_dep, min_max_lat, min_max_lon, n_rtp):
        """Initialize axis

        :param min_max_dep: min and max depth, ``[min_dep, max_dep]``
        :type min_max_dep: list
        :param min_max_lat: Min and max latitude, ``[min_lat, max_lat]``
        :type min_max_lat: list
        :param min_max_lon: Min and max longitude, ``[min_lon, max_lon]``
        :type min_max_lon: list
        :param n_rtp: number of dimensions [ndep, nlat, nlon]
        :type n_rtp: list
        """
        self.dd, self.tt, self.pp, self.dr, self.dt, self.dp, = init_axis(
            min_max_dep, min_max_lat, min_max_lon, n_rtp
        )

    def _create_taper(self, xleft, xright, type='d'):
        if type == 'd':
            x = np.flip(self.dd); dx = self.dr
        elif type == 't':
            x = self.tt; dx = self.dt
        elif type == 'p':
            x = self.pp; dx = self.dp
        else:
            pass
        if xleft < x[0] or xright > x[-1]:
            raise ValueError('limitation out of range')
        ntaper_left = int((xleft-x[0])/dx)
        ntaper_right = int((x[-1]-xright)/dx)
        return ntaper_left, ntaper_right

    def checkerboard(self, period_x, period_y, period_z,
                     pert_vel=0.08, pert_ani=0.04,
                     lim_x=None, lim_y=None, lim_z=None):
        """Create checkerboard

        :param period_x: Multiple of period along X, e.g., set to 1 for 2 anomalies
        :type period_x: float
        :param period_y: Multiple of period along Y
        :type period_y: float
        :param period_z: Multiple of period along Z
        :type period_z: float
        :param pert_vel: Perturbation for velocity, defaults to 0.08
        :type pert_vel: float, optional
        :param pert_ani: Perturbation for anisotropy, defaults to 0.04
        :type pert_ani: float, optional
        :param lim_x: Left and right bound along X, defaults to None
        :type lim_x: list, optional
        :param lim_y: Left and right bound along Y, defaults to None
        :type lim_y: list, optional
        :param lim_z: Left and right bound along Z, defaults to None
        :type lim_z: list, optional
        """
        if lim_x is not None:
            ntaper_left, ntaper_right = self._create_taper(*lim_x, type='p')
        else:
            ntaper_left = 0
            ntaper_right = 0
        x_pert = np.zeros_like(self.pp)
        x_pert[ntaper_left:self.pp.size-ntaper_right] = \
            np.sin(period_x*2*np.pi*np.arange(self.pp.size-(ntaper_left+ntaper_right))/ \
            (self.pp.size-(ntaper_left+ntaper_right)))

        if lim_y is not None:
            ntaper_left, ntaper_right = self._create_taper(*lim_y, type='t')
        else:
            ntaper_left = 0
            ntaper_right = 0
        y_pert = np.zeros_like(self.tt)
        y_pert[ntaper_left:self.tt.size-ntaper_right] = \
            np.sin(period_y*2*np.pi*np.arange(self.tt.size-(ntaper_left+ntaper_right))/ \
            (self.tt.size-(ntaper_left+ntaper_right)))

        if lim_z is not None:
            ntaper_left, ntaper_right = self._create_taper(*lim_z, type='d')
        else:
            ntaper_left = 0
            ntaper_right = 0
        z_pert = np.zeros_like(self.dd)
        z_pert[ntaper_right:self.dd.size-ntaper_left] = \
            np.sin(period_z*2*np.pi*np.arange(self.dd.size-(ntaper_left+ntaper_right))/ \
            (self.dd.size-(ntaper_left+ntaper_right)))

        xx, yy, zz= np.meshgrid(z_pert, y_pert, x_pert, indexing='ij')
        self.perturbation = xx*yy*zz
        self.vel_pert = self.vel * (1+self.perturbation*pert_vel)
        self.dlnv = self.perturbation*pert_vel
        self.epsilon = np.abs(self.perturbation)*pert_ani
        self.phi = np.zeros_like(self.vel)
        self.phi[np.where(self.perturbation>0)] = 135.
        self.phi[np.where(self.perturbation<0)] = 45.
        self.xi = self.epsilon*cosd(2*self.phi)
        self.eta = self.epsilon*sind(2*self.phi)

    def write(self, fname: str):
        """Write new model to h5 file

        :param fname: Path to output file
        :type fname: str
        """
        if fname is None:
            fname = '.'.join(self.model_file.split('.')[:-1])+'_pert.h5'
        with h5py.File(fname, 'w') as f:
            f.create_dataset('xi', data=self.xi)
            f.create_dataset('eta', data=self.eta)
            f.create_dataset('zeta', data=self.zeta)
            f.create_dataset('vel', data=self.vel_pert)
            f.create_dataset('epsilon', data=self.epsilon)
            f.create_dataset('phi', data=self.phi)
            f.create_dataset('dlnv', data=self.dlnv)


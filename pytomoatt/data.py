import numpy as np
import h5py
from .para import ATTPara
from .attarray import Dataset
from .utils import asind, acosd


class ATTData():
    """Read data from HDF5 or ASCII file

    :param fname: Path to data file
    :type fname: str
    :param fname_params: Path to input parameter file
    :type fname_params: str
    :param fname_grid: Path to grid file
    :type fname_grid: str
    """
    def __init__(self, fname:str,
                 fname_params:str,
                 fname_grid='OUTPUT_FILES/out_data_grid.h5'):
        self.fname = fname
        self.fname_grid = fname_grid
        self.format = 'hdf5'
        self.fgrid = None
        self.fdata = None
        self.grid_glob_r = None
        self.grid_glob_t = None
        self.grid_glob_p = None
        self.fields = []
        self.input_params = ATTPara(fname_params).input_params
        self.ndiv_r, self.ndiv_t, self.ndiv_p = self.input_params['parallel']['ndiv_rtp']
        self.nr_glob, self.nt_glob, self.np_glob = self.input_params['domain']['n_rtp']

    def _add_field(self, name):
        exec('self.{} = None'.format(name))
        self.fields.append(name)

    def _read_h5(self):
        """read data file with HDF5 format
        """
        self.fgrid = h5py.File(self.fname_grid, 'r')
        self.fdata = h5py.File(self.fname, 'r')

    @classmethod
    def read(cls, fname:str, fname_params:str,
                fname_grid='OUTPUT_FILES/out_data_grid.h5',
                group_name='model', dataset_name=None,
                format='hdf5'):
        """Read data from HDF5 or ASCII file

        :param fname: Path to data file
        :type fname: str
        :param fname_params: Path to input parameter file
        :type fname_params: str
        :param fname_grid: Path to grid file
        :type fname_grid: str
        :param group_name: Name of the group in the HDF5 file
        :type group_name: str
        :param dataset_name: Name of the dataset in the HDF5 file
        :type dataset_name: str
        :param format: Format of the data file, defaults to 'hdf5'
        :type format: str, optional
        :return: An instance of ATTData
        :rtype: ATTData
        """
        attdata = cls(fname, fname_params, fname_grid)
        attdata.format = format
        # attdata.group_name = group_name
        # open grid data file
        if attdata.format == 'hdf5':
            attdata._read_h5()
        else:
            attdata.fdata = np.loadtxt(fname)
            attdata.fgrid = np.loadtxt(fname_grid)
        if isinstance(dataset_name, str) and attdata.format == 'hdf5':
            attdata._add_field(dataset_name)
            attdata.__dict__[key], attdata.grid_glob_r, \
            attdata.grid_glob_t, attdata.grid_glob_p = \
            attdata._data_retrieval(group_name=group_name, dataset_name=key)
        elif isinstance(dataset_name, str) and attdata.format != 'hdf5':
            attdata._add_field('data')
            attdata.data, attdata.grid_glob_r, attdata.grid_glob_t, attdata.grid_glob_p = \
                attdata._data_retrieval()
        elif isinstance(dataset_name, (list, tuple)) and attdata.format == 'hdf5':
            for key in dataset_name:
                if not (key in attdata.fdata[group_name].keys()):
                    raise ValueError('Error dataset_name of {}. \n{} are available.'.format(key, ', '.join(attdata.fgrid.keys())))
                attdata._add_field(key)
                print(attdata.vel)
                attdata.__dict__[key], attdata.grid_glob_r, \
                attdata.grid_glob_t, attdata.grid_glob_p = \
                attdata._data_retrieval(group_name=group_name, dataset_name=key)
        elif dataset_name is None and attdata.format == 'hdf5':
            for key in attdata.fdata[group_name].keys():
                attdata._add_field(key)
                attdata.__dict__[key], attdata.grid_glob_r, \
                attdata.grid_glob_t, attdata.grid_glob_p = \
                attdata._data_retrieval(group_name=group_name, dataset_name=key)
        else:
            raise ValueError('Error format of dataset_name')
        return attdata

    def _read_data_hdf5(self, offset, n_points_total_sub, group_name, dataset_name):
        data_sub = self.fdata[group_name][dataset_name][offset:offset+n_points_total_sub]
        grid_sub_p = self.fgrid["/Mesh/node_coords_p"][offset:offset+n_points_total_sub]
        grid_sub_t = self.fgrid["/Mesh/node_coords_t"][offset:offset+n_points_total_sub]
        grid_sub_r = self.fgrid["/Mesh/node_coords_r"][offset:offset+n_points_total_sub]
        return data_sub, grid_sub_p, grid_sub_t, grid_sub_r
    
    def _read_data_ascii(self, offset, n_points_total_sub):
        data_sub = self.fdata[offset:offset+n_points_total_sub]
        grid_sub_p = self.fgrid[offset:offset+n_points_total_sub,0]
        grid_sub_t = self.fgrid[offset:offset+n_points_total_sub,1]
        grid_sub_r = self.fgrid[offset:offset+n_points_total_sub,2]
        return data_sub, grid_sub_p, grid_sub_t, grid_sub_r

    def _data_retrieval(self, group_name=None, dataset_name=None):
        # prepare a 3D array to store the data
        data_glob = np.zeros(self.input_params['domain']['n_rtp'], dtype=np.float64)
        grid_glob_r = np.zeros(self.input_params['domain']['n_rtp'], dtype=np.float64)
        grid_glob_t = np.zeros(self.input_params['domain']['n_rtp'], dtype=np.float64)
        grid_glob_p = np.zeros(self.input_params['domain']['n_rtp'], dtype=np.float64)

        # load data data by each subdomain

        # offset
        offset = 0

        for ir_sub in range(self.ndiv_r):
            for it_sub in range(self.ndiv_t):
                for ip_sub in range(self.ndiv_p):

                    # number of data point for this sub domain
                    nr_sub = self.nr_glob//self.ndiv_r
                    nt_sub = self.nt_glob//self.ndiv_t
                    np_sub = self.np_glob//self.ndiv_p

                    # offset for each direction
                    offset_r = ir_sub*nr_sub
                    offset_t = it_sub*nt_sub
                    offset_p = ip_sub*np_sub

                    # add modulus to the last subdomains
                    if ir_sub == self.ndiv_r-1:
                        nr_sub += self.nr_glob%self.ndiv_r
                    if it_sub == self.ndiv_t-1:
                        nt_sub += self.nt_glob%self.ndiv_t
                    if ip_sub == self.ndiv_p-1:
                        np_sub += self.np_glob%self.ndiv_p

                    # add overlap layer if this subdomain is not the last one for each direction
                    if ir_sub != self.ndiv_r-1:
                        nr_sub += 1
                    if it_sub != self.ndiv_t-1:
                        nt_sub += 1
                    if ip_sub != self.ndiv_p-1:
                        np_sub += 1

                    # number of data point for this sub domain
                    n_points_total_sub = nr_sub*nt_sub*np_sub

                    # load data
                    if self.format == 'hdf5':
                        data_sub, grid_sub_p, grid_sub_t, grid_sub_r = self._read_data_hdf5(
                            offset, n_points_total_sub, group_name, dataset_name)
                    else:
                        data_sub, grid_sub_p, grid_sub_t, grid_sub_r = self._read_data_ascii(
                            offset, n_points_total_sub)

                    # reshape data
                    data_sub = data_sub.reshape(nr_sub, nt_sub, np_sub)
                    grid_sub_p = grid_sub_p.reshape(nr_sub, nt_sub, np_sub)
                    grid_sub_t = grid_sub_t.reshape(nr_sub, nt_sub, np_sub)
                    grid_sub_r = grid_sub_r.reshape(nr_sub, nt_sub, np_sub)

                    # put those data in global 3d array
                    data_glob[offset_r:offset_r+nr_sub, offset_t:offset_t+nt_sub, offset_p:offset_p+np_sub] = data_sub
                    grid_glob_p[offset_r:offset_r+nr_sub, offset_t:offset_t+nt_sub, offset_p:offset_p+np_sub] = grid_sub_p
                    grid_glob_t[offset_r:offset_r+nr_sub, offset_t:offset_t+nt_sub, offset_p:offset_p+np_sub] = grid_sub_t
                    grid_glob_r[offset_r:offset_r+nr_sub, offset_t:offset_t+nt_sub, offset_p:offset_p+np_sub] = grid_sub_r

                    # update offset
                    offset += n_points_total_sub
        return data_glob, grid_glob_r, grid_glob_t, grid_glob_p

    def to_xarray(self):
        """Convert to attarray.Dataset

        :return: A multi-dimensional data base inheriting from xarray.Dataset
        :rtype: attarray.DataSet
        """
        depths = 6371. - self.grid_glob_r[:, 0, 0]
        radius = self.grid_glob_r[:, 0, 0]
        latitudes = self.grid_glob_t[0, :, 0]
        longitudes = self.grid_glob_p[0, 0, :]
        data_dict = {}
        for dataset_name in self.fields:
            data_dict[dataset_name] = (["r", "t", "p"], self.__dict__[dataset_name])
        dataset = Dataset(
            data_dict,
            coords={
                'dep': (['r'], depths),
                'rad': (['r'], radius),
                'lat': (['t'], latitudes),
                'lon': (['p'], longitudes),
            }
        )
        return dataset


if __name__ == '__main__':
    attdata = ATTData.read('examples/out_data_sim_0.h5',
                           'examples/input_params.yml',
                           'examples/out_data_grid.h5',
                            dataset_name='T_res_src_0_inv_0000',
                            format='hdf5')
    
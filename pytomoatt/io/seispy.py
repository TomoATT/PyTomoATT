import glob
import pandas as pd
import numpy as np
from os.path import join, basename, dirname
from obspy.io.sac import SACTrace


class Seispy:
    def __init__(self, rf_path:str):
        """This is an I/O for Seispy

        :param rf_path: Path to Receiver Function data
        :type rf_path: str
        """
        self.rf_path = rf_path
        self.src_points = None

    def _load_sta_info(self):
        """Load station information from SAC header
        """
        self.sta_paths = sorted(glob.glob(join(self.rf_path, '*/')))
        self.stanames = [basename(dirname(sta_path)) for sta_path in self.sta_paths]
        self.sta_info = pd.DataFrame(columns=[
            'staname', 'stla', 'stlo', 'stel'
        ])
        for i, path in enumerate(self.sta_paths):
            sac = SACTrace.read(glob.glob(join(path, '*.sac'))[0])
            self.sta_info.loc[i] = [
                self.stanames[i], sac.stla, sac.stlo, sac.stel
            ]

    def get_rf_info(self):
        """Read event information form ``finallist.dat``
        """
        self.finallist_paths = [join(self.rf_path, sta, sta+'finallist.dat') 
                                for sta in self.stanames]
        dfs = []
        names = ['event_id', 'phase', 'evla', 'evlo', 'evdp',
                  'gcarc', 'baz', 'rayp', 'mag', 'f0']
        for i, fname in enumerate(self.finallist_paths):
            evts = pd.read_csv(fname, sep='\s+', header=None, names=names)
            sta_info = self.sta_info[self.sta_info['staname'] == self.stanames[i]]
            sta_info_repeated = pd.concat([sta_info]*evts.shape[0], ignore_index=True)
            evts = pd.concat([evts, sta_info_repeated], axis=1)
            dfs.append(evts)
        self.rf_info = pd.concat(dfs, ignore_index=True)

    def to_src_rec_points(self):
        """
        Convert to dataframe of ``pytomoatt.src_rec.SrcRec.src_points``
        """
        # Select all unique events and save to src_points
        self.src_points = self.rf_info[['event_id', 'evla', 'evlo', 'evdp', 'mag']]
        self.src_points = self.src_points.drop_duplicates(keep='first', ignore_index=True)
        self.src_points['num_rec'] = [0]*self.src_points.shape[0]
        self.src_points['weight'] = [1.]*self.src_points.shape[0]
        self.src_points['origin_time'] = pd.to_datetime(
            self.src_points['event_id'],
            format='%Y.%j.%H.%M.%S'
        )
        self.src_points.index.name = 'src_index'
        self.src_points = self.src_points[[
            'origin_time', 'evla', 'evlo', 'evdp',
            'mag', 'num_rec', 'event_id', 'weight'
        ]]

        # Select all receivers and save to rec_points
        columns=[
            'src_index', 'rec_index', 'staname',
            'stla', 'stlo', 'stel', 'phase', 'dist_deg', 'tt', 'weight'
        ]
        rec_lst = []
        for i, evt in self.src_points.iterrows():
            stas = self.rf_info[self.rf_info['event_id'] == evt['event_id']]
            evt['num_rec'] = stas.shape[0]
            df = pd.DataFrame({
                'src_index': [i]*stas.shape[0],
                'rec_index': np.arange(stas.shape[0]),
                'staname': stas['staname'],
                'stla': stas['stla'].values, 
                'stlo': stas['stlo'].values, 
                'stel': stas['stel'].values,
                'phase': stas['phase'].values,
                'dist_deg': stas['gcarc'].values, 
                'tt': [0.]*stas.shape[0],
                'weight': [1.]*stas.shape[0]
            })
            rec_lst.append(df)
        self.rec_points =  pd.concat(rec_lst, ignore_index=True) 
        return self.src_points, self.rec_points

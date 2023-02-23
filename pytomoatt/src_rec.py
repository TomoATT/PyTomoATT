import numpy as np
import tqdm
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

class SrcRec():

    def __init__(self, fname='', src_only=False) -> None:
        """I/O for source <--> receiver file

        :param fname: Path to src_rec file
        :type fname: str
        :param src_only: Whether to read only source information, defaults to False
        :type src_only: bool, optional
        """

        self.src_only = src_only
        self.src_points = None
        self.rec_points = None
        self.fnames = [fname]

    def __repr__(self):
        return f"PyTomoATT SrcRec Object: \n\
                fnames={self.fnames}, \n\
                src_only={self.src_only}, \n\
                number of sources={self.src_points.shape[0]}, \n\
                number of receivers={self.rec_points.shape[0]}"

    @classmethod
    def read(cls, fname:str, dist_in_data=False, name_net_and_sta=False, **kwargs):
        """Read source <--> receiver file to pandas.DataFrame

        :param fname: Path to src_rec file
        :type fname: str
        :param dist_in_data: Whether distance is included in the src_rec file
        :type dist_in_data: bool
        :param name_net_and_sta: Whether to include network and station name in the src_rec file
        :type name_net_and_sta: bool
        :return: class of SrcRec
        :rtype: SrcRec
        """
        sr = cls(fname=fname, **kwargs)
        alldf = pd.read_table(fname, sep='\s+|\t', engine='python',
                              header=None, comment="#")

        last_col_src = 12
        # this is a source line if the last column is not NaN
        sr.src_points = alldf[pd.notna(alldf[last_col_src])]
        # add weight column if not included
        if sr.src_points.shape[1] == last_col_src+1:
            # add another column for weight
            sr.src_points.loc[:, last_col_src+1] = 1.0

        # src id dataframe
        sr.src_points.index = sr.src_points.iloc[:, 0]
        sr.src_points.index.name = 'src_index'

        # event datetime dataframe
        datedf = sr.src_points.loc[:, 1:6]
        type_dict = {
            1: int,
            2: int,
            3: int,
            4: int,
            5: int,
            6: float,
        }
        try:
            datedf = datedf.astype(type_dict)
        except:
            print("Error: please check the date format in the src_rec file")
            return sr.src_points
        dateseris = datedf.astype(str).apply(lambda x: '.'.join(x), axis=1).apply(
            pd.to_datetime, format='%Y.%m.%d.%H.%M.%S.%f')
        dateseris.name = 'origin_time'
        # event data dataframe
        src_data = sr.src_points.loc[:, 7:]
        src_data.columns = ['evla', 'evlo',
                            'evdp', 'mag', 'num_rec',
                            'event_id', 'weight']
        type_dict = {
            'evla': float,
            'evlo': float,
            'evdp': float,
            'mag': float,
            'num_rec': int,
            'event_id': str,
            'weight': float
        }
        try:
            src_data = src_data.astype(type_dict)
        except:
            print("Error2: please check the event data format in the src_rec file")
            return sr.src_points

        # concat all the 3 dataframes
        sr.src_points = pd.concat([dateseris, src_data], axis=1)

        # read receiver data if not src_only
        if not sr.src_only:
            # number of columns is 8 if distance is not included
            if not dist_in_data:
                last_col = 7
            else:
                last_col = 8

            if name_net_and_sta:
                last_col += 1

            # extract the rows if the last_col is not NaN and the 12th column is NaN
            sr.rec_points = alldf[(alldf[last_col].notna())
                                & (alldf[last_col_src].isna())].reset_index(drop=True)

            # add weight column if not included
            if sr.rec_points.loc[:, last_col+1].isna().all():
                sr.rec_points.loc[:, last_col+1] = 1.0

            # warning if weigh value is greater than 10
            if (sr.rec_points.loc[:, last_col+1] > 10).any():
                print("""
Warning: at least one weight value is greater than 10.
Probably your src_rec file includes distance data.
In this case, please set dist_in_data=True and read again.""")

            # extract only the first part of columns (cut unnecessary columns)
            sr.rec_points = sr.rec_points.loc[:, :last_col+1]

            if name_net_and_sta == False:
                if not dist_in_data:
                    sr.rec_points.columns = [
                        'src_index', 'rec_index', 'staname',
                        'stla', 'stlo', 'stel', 'phase', 'tt', 'weight'
                    ]
                else:
                    sr.rec_points.columns = [
                        'src_index', 'rec_index', 'staname',
                        'stla', 'stlo', 'stel', 'phase', 'dist_deg', 'tt', 'weight'
                    ]
            else:
                if not dist_in_data:
                    sr.rec_points.columns = [
                        'src_index', 'rec_index', 'netname', 'staname',
                        'stla', 'stlo', 'stel', 'phase', 'tt', 'weight'
                    ]
                else:
                    sr.rec_points.columns = [
                        'src_index', 'rec_index', 'netname', 'staname',
                        'stla', 'stlo', 'stel', 'phase', 'dist_deg', 'tt', 'weight'
                    ]

                # concatenate network and station name with "_"
                sr.rec_points['staname'] = sr.rec_points['netname'] + '_' + sr.rec_points['staname']
                # drop network name column
                sr.rec_points.drop('netname', axis=1, inplace=True)

        return sr

    def write(self, fname='src_rec_file'):
        """Write sources and receivers to ASCII file for TomoATT

        :param fname: Path to the src_rec file, defaults to 'src_rec_file'
        :type fname: str, optional
        """
        with open(fname, 'w') as f:
            for idx, src in tqdm.tqdm(self.src_points.iterrows(),
                                      total=len(self.src_points)):
                time_lst = src['origin_time'].strftime('%Y_%m_%d_%H_%M_%S.%f').split('_')
                f.write('{:d} {} {} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {} {} {:.4f}\n'.format(
                    idx, *time_lst, src['evla'], src['evlo'], src['evdp'],
                    src['mag'], src['num_rec'], src['event_id'], src['weight']
                ))
                if self.src_only:
                    continue
                rec_data = self.rec_points[self.rec_points['src_index']==idx]
                for _, rec in rec_data.iterrows():
                    f.write('   {:d} {:d} {} {:6.4f} {:6.4f} {:6.4f} {} {:6.4f} {:6.4f}\n'.format(
                        idx, rec['rec_index'], rec['staname'], rec['stla'],
                        rec['stlo'], rec['stel'], rec['phase'], rec['tt'], rec['weight']
                    ))

    def reset_index(self):
        # reset src_index to be 0, 1, 2, ... for both src_points and rec_points
        self.rec_points['src_index'] = self.rec_points['src_index'].map(
            dict(zip(self.src_points.index, np.arange(len(self.src_points)))))
        self.src_points.index = np.arange(len(self.src_points))
        self.src_points.index.name = 'src_index'

        # reset rec_index to be 0, 1, 2, ... for rec_points
        self.rec_points['rec_index'] = self.rec_points.groupby('src_index').cumcount()
        #sr.rec_points['rec_index'] = sr.rec_points['rec_index'].astype(int)

    def append(self, sr):
        """
        Append another SrcRec object to the current one

        :param sr: Another SrcRec object
        :type sr: SrcRec
        """
        if not isinstance(sr, SrcRec):
            raise TypeError('Input must be a SrcRec object')

        if self.src_only != sr.src_only:
            raise ValueError('Cannot append src_only and non-src_only SrcRec objects')

        # number of sources to be added
        n_src_offset = self.src_points.shape[0]

        # add column for source file tag if not included
        if 'fname' not in self.src_points.columns:
            self.src_points['fname'] = self.fnames[0]
            self.rec_points['fname'] = self.fnames[0]
        if 'fname' not in sr.src_points.columns:
            sr.src_points['fname'] = sr.fnames[0]
            sr.rec_points['fname'] = sr.fnames[0]

        # append src_points
        self.src_points = pd.concat([self.src_points, sr.src_points], ignore_index=True)
        self.src_points.index.name = 'src_index'
        self.src_points.index += 1  # start from 1

        if not self.src_only:
            # update src_index in rec_points
            sr.rec_points['src_index'] += n_src_offset
            # append rec_points
            self.rec_points = pd.concat([self.rec_points, sr.rec_points], ignore_index=True)

        # store fnames
        self.fnames.extend(sr.fnames)

    def remove_rec_by_new_src(self):
        """
        remove rec_points by new src_points
        """
        print('rec_points before removing: ', self.rec_points.shape)
        self.rec_points = self.rec_points[self.rec_points['src_index'].isin(self.src_points.index)]
        print('rec_points after removing: ', self.rec_points.shape)

    def update_num_rec(self):
        """
        update num_rec in src_points by current rec_points
        """
        self.src_points['num_rec'] = self.rec_points.groupby('src_index').size()

    def erase_duplicate_events(self, thre_deg:float, thre_dep:float, thre_time_in_min:float):
        """
        check and count how many events are duplicated,
        under given threshold of distance, depth, and time.
        thre_deg : float
            threshold of distance in degree
        thre_dep : float
            threshold of depth in km
        thre_time_in_min : float
            threshold of time in minutes
        """

        # sort event data
        self.src_points.sort_values(by=["origin_time", "evlo", "evla"], inplace=True)

        num_duplicated = 99999
        iter_count = 0

        while (num_duplicated > 0):

            # difference with row +1
            self.src_points['diff_evlo+1'] = self.src_points['evlo'].diff().abs()
            self.src_points['diff_evla+1'] = self.src_points['evla'].diff().abs()
            self.src_points['diff_evdp+1'] = self.src_points['evdp'].diff().abs()
            self.src_points['diff_time+1'] = self.src_points['origin_time'].diff().abs()
            self.src_points['diff_nrec+1'] = self.src_points['num_rec'].diff()
            # difference with row -1
            self.src_points['diff_evlo-1'] = self.src_points['evlo'].diff(periods=-1).abs()
            self.src_points['diff_evla-1'] = self.src_points['evla'].diff(periods=-1).abs()
            self.src_points['diff_evdp-1'] = self.src_points['evdp'].diff(periods=-1).abs()
            self.src_points['diff_time-1'] = self.src_points['origin_time'].diff(periods=-1).abs()
            self.src_points['diff_nrec-1'] = self.src_points['num_rec'].diff(periods=-1)

            self.src_points["duplicated+1"] = self.src_points.apply(lambda x: 1 if x['diff_evlo+1']<thre_deg and x['diff_evla+1']<thre_deg and x['diff_evdp+1']<thre_dep and x['diff_time+1']<pd.Timedelta(minutes=thre_time_in_min) else 0, axis=1)
            self.src_points["duplicated-1"] = self.src_points.apply(lambda x: 1 if x['diff_evlo-1']<thre_deg and x['diff_evla-1']<thre_deg and x['diff_evdp-1']<thre_dep and x['diff_time-1']<pd.Timedelta(minutes=thre_time_in_min) else 0, axis=1)

            # drop rows (duplicated == 1 and diff_nrec <= 0)
            self.src_points = self.src_points[~((self.src_points['duplicated+1']==1) & (self.src_points['diff_nrec+1']<0))]
            # drow upper row of (duplicated == 1 and diff_nrec > 0)
            self.src_points = self.src_points[~((self.src_points['duplicated-1']==1) & (self.src_points['diff_nrec-1']<=0))]

            # print iterate count and number of rows, number of duplicated rows
            num_duplicated = self.src_points[(self.src_points["duplicated+1"]==1) | (self.src_points["duplicated-1"]==1)].shape[0]
            print("iteration: ", iter_count, "num_duplicated: ", num_duplicated)

            iter_count += 1

        # erase all columns starting with diff_*
        self.src_points.drop(self.src_points.columns[self.src_points.columns.str.startswith('diff_')], axis=1, inplace=True)
        # erase all clumuns starting with duplicated
        self.src_points.drop(self.src_points.columns[self.src_points.columns.str.startswith('duplicated')], axis=1, inplace=True)

        # remove rec_points by new src_points
        self.remove_rec_by_new_src()

        # sort by src_index
        self.src_points.sort_values(by=['src_index'], inplace=True)
        self.rec_points.sort_values(by=['src_index', 'rec_index'], inplace=True)

    def select_phase(self, phase_list):
        """
        select interested phase and remove others
        phase_list : list of str
        """
        if not isinstance(phase_list, (list, str)):
            raise TypeError('phase_list should be in list or str')
        print('rec_points before selecting: ', self.rec_points.shape)
        self.rec_points = self.rec_points[self.rec_points['phase'].isin(phase_list)]
        print('rec_points after selecting: ', self.rec_points.shape)

        # modify num_rec in src_points
        self.src_points['num_rec'] = self.rec_points.groupby('src_index').size()

        # sort by src_index
        self.src_points.sort_values(by=['src_index'], inplace=True)
        self.rec_points.sort_values(by=['src_index', 'rec_index'], inplace=True)

    def select_box_region(self, region):
        """
        Select sources and station in a box region
        param region: Box region defined as [lon1, lon2, lat1, lat2]
        :type region: iterable
        """
        # select source within this region.
        self.src_points = self.src_points[
            (self.src_points['evlo'] >= region[0]) &
            (self.src_points['evlo'] <= region[1]) &
            (self.src_points['evla'] >= region[2]) &
            (self.src_points['evla'] <= region[3])
        ]

        # Remove receivers whose events have been removed 
        self.remove_rec_by_new_src()

        # Remove rest receivers out of region.
        self.rec_points = self.rec_points[
            (self.rec_points['stlo'] >= region[0]) &
            (self.rec_points['stlo'] <= region[1]) &
            (self.rec_points['stla'] >= region[2]) &
            (self.rec_points['stla'] <= region[3])
        ]

        # Remove empty sources
        self.src_points = self.src_points[self.src_points.index.isin(self.rec_points['src_index'])]
        self.update_num_rec()

    def select_one_event_in_each_subgrid(self, d_deg:float, d_km:float):
        """
        select one event in each subgrid
        d_deg : float
            grid size in degree
        d_km : float
            grid size in km
        """
        print('src_points before selecting: ', self.src_points.shape)
        print('processing... (this may take a few minutes)')

        # store index of src_points as 'src_index'
        self.src_points['src_index'] = self.src_points.index

        # add 'lat_group' and 'lon_group' to src_points by module d_deg
        self.src_points['lat_group'] = self.src_points['evla'].apply(lambda x: int(x/d_deg))
        self.src_points['lon_group'] = self.src_points['evlo'].apply(lambda x: int(x/d_deg))

        # add 'dep_group' to src_points by module d_km
        self.src_points['dep_group'] = self.src_points['evdp'].apply(lambda x: int(x/d_km))

        # sort src_points by 'lat_group' and 'lon_group' and 'dep_group'
        self.src_points = self.src_points.sort_values(by=['lat_group', 'lon_group', 'dep_group'])

        # find all events in the same lat_group and lon_group and dep_group
        # and keep only on with largest nrec
        self.src_points = self.src_points.groupby(['lat_group', 'lon_group', 'dep_group']).apply(lambda x: x.sort_values(by='num_rec', ascending=False).iloc[0])

        # drop 'lat_group' and 'lon_group' and 'dep_group'
        self.src_points = self.src_points.drop(columns=['lat_group', 'lon_group', 'dep_group'])

        # restore index from 'src_index'
        self.src_points = self.src_points.set_index('src_index')

        # sort src_points by index
        self.src_points = self.src_points.sort_index()

        print('src_points after selecting: ', self.src_points.shape)

        # remove rec_points by new src_points
        self.remove_rec_by_new_src()

    def count_events_per_station(self):
        """
        count events per station
        """
        # count the number of staname
        self.rec_points['num_events'] = self.rec_points.groupby('staname').cumcount() + 1
        # reflect the total number of events for each station
        self.rec_points['num_events'] = self.rec_points.groupby('staname')['num_events'].transform('max')

    def merge_adjacent_stations(self, d_deg:float, d_km:float):
        """
        merge adjacent stations as one station
        d_deg : float
            grid size in degree
        d_km : float
            grid size in km
        """

        # count the number of events per station
        self.count_events_per_station()

        # number of unique stations before merging
        print('number of unique stations before merging: ', self.rec_points['staname'].nunique())

        # create 'lat_group', 'lon_group' and 'dep_group' columns from 'stla', 'stlo' and 'stel'
        def create_groups(row, column, d):
            return int(row[column]/d)

        self.rec_points['lat_group'] = self.rec_points.apply(lambda x: create_groups(x, 'stla', d_deg), axis=1)
        self.rec_points['lon_group'] = self.rec_points.apply(lambda x: create_groups(x, 'stlo', d_deg), axis=1)
        self.rec_points['dep_group'] = self.rec_points.apply(lambda x: create_groups(x, 'stel', d_km*1000), axis=1)

        # sort src_points by 'lat_group' and 'lon_group' and 'dep_group'
        self.rec_points = self.rec_points.sort_values(by=['lat_group', 'lon_group', 'dep_group', 'num_events'], ascending=[True, True, True, False])

        # find all events in the same lat_group and lon_group and dep_group
        # and copy the 'staname' 'stlo' 'stla' 'stel' to all rows within the same group from the row where 'count' is the largest
        self.rec_points['staname'] = self.rec_points.groupby(['lat_group', 'lon_group', 'dep_group'])['staname'].transform(lambda x: x.iloc[0])
        self.rec_points['stlo'] = self.rec_points.groupby(['lat_group', 'lon_group', 'dep_group'])['stlo'].transform(lambda x: x.iloc[0])
        self.rec_points['stla'] = self.rec_points.groupby(['lat_group', 'lon_group', 'dep_group'])['stla'].transform(lambda x: x.iloc[0])
        self.rec_points['stel'] = self.rec_points.groupby(['lat_group', 'lon_group', 'dep_group'])['stel'].transform(lambda x: x.iloc[0])

        # drop 'lat_group' and 'lon_group' and 'dep_group'
        self.rec_points = self.rec_points.drop(columns=['lat_group', 'lon_group', 'dep_group'])

        # sort
        self.rec_points = self.rec_points.sort_values(by=['src_index','rec_index'])

        # update the num_events
        self.count_events_per_station()

        # number of unique stations after merging
        print('number of unique stations after merging: ', self.rec_points['staname'].nunique())

    def merge_duplicated_station(self):
        """
        merge duplicated stations as one station
        duplicated stations are defined as stations with the same staname
        """

        # number of unique stations before merging
        print('number of unique stations before merging: ', self.rec_points['staname'].nunique())

        # sort rec_points by 'src_index' then 'staname'
        self.rec_points = self.rec_points.sort_values(by=['src_index', 'staname'])

        # find all duplicated stations in each src_index and drop except the first one
        self.rec_points = self.rec_points.drop_duplicates(subset=['src_index', 'staname'], keep='first')

        # sort rec_points by 'src_index' then 'rec_index'
        self.rec_points = self.rec_points.sort_values(by=['src_index', 'rec_index'])

        # update the num_events
        self.count_events_per_station()

        # number of unique stations after merging
        print('number of unique stations after merging: ', self.rec_points['staname'].nunique())

    def write_receivers(self, fname:str):
        """
        Write receivers to a txt file
        :param fname: Path to output txt file of stations
        """
        recs = self.rec_points[['staname', 'stla', 'stlo', 'stel', 'weight']].drop_duplicates()
        recs.to_csv(fname, sep=' ', header=False, index=False)

    def write_sources(self, fname:str):
        """
        Write sources to a txt file
        :param fname: Path to output txt file of sources
        """
        srcs = self.src_points[['event_id', 'evla', 'evlo', 'evdp', 'weight']]
        srcs.to_csv(fname, sep=' ', header=False, index=False)

    @classmethod
    def from_seispy(cls, rf_path:str):
        """Read and convert source and station information from
        receiver function data calculated by Seispy

        :param rf_path: Path to receiver functions calculated by Seispy
        :type rf_path: str
        :return: New instance of class SrcRec
        :rtype: SrcRec
        """
        from .io.seispy import Seispy
        sr = cls()
        # Initial an instance of Seispy
        seispyio = Seispy(rf_path)

        # Load station info from SAC header
        seispyio._load_sta_info()

        # Read finallist.dat
        seispyio.get_rf_info()

        # Convert to SrcRec format
        sr.src_points, sr.rec_points = seispyio.to_src_rec_points()
        return sr

    # implemented in vis.py
    def plot(self, weight=False, fname=None):
        """Plot source and receivers for preview

        :param weight: _description_, defaults to False
        :type weight: bool, optional
        :param fname: _description_, defaults to None
        :type fname: _type_, optional
        :return: _description_
        :rtype: _type_
        """
        from .vis import plot_srcrec
        return plot_srcrec(self, weight=weight, fname=fname)


if __name__ == '__main__':
    sr = SrcRec.read('src_rec_file_checker_data_test1.dat_noised_evweighted')
    sr.write()
    print(sr.rec_points)
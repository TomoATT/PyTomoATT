import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import tqdm

class SrcRec():
    def __init__(self, fname:str, src_only=False) -> None:
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
    def read(cls, fname: str, dist_in_data=False, name_net_and_sta=False, **kwargs):
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
        sr = cls(fname, **kwargs)
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
        """Append another SrcRec object to the current one

        :param sr: Another SrcRec object
        :type sr: SrcRec
        """
        if not isinstance(sr, SrcRec):
            raise TypeError('Input must be a SrcRec object')

        if self.src_only != sr.src_only:
            raise ValueError('Cannot append src_only and non-src_only SrcRec objects')

        # number of sources to be added
        n_src_offset = self.src_points.shape[0]

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

if __name__ == '__main__':
    sr = SrcRec.read('src_rec_file_checker_data_test1.dat_noised_evweighted')
    sr.write()
    print(sr.rec_points)
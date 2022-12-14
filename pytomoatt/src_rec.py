import pandas as pd

class SrcRec():
    def __init__(self, fname:str, src_only=False) -> None:
        """I/O for source <--> receiver file

        :param fname: Path to src_rec file
        :type fname: str
        :param src_only: Whether to read only source information, defaults to False
        :type src_only: bool, optional
        """

        self.fname = fname
        self.src_only = src_only
        self.src_points = None
        self.rec_points = None

    def __repr__(self):
        return f"PyTomoATT SrcRec Object: \n\
                fname={self.fname}, \n\
                src_only={self.src_only}, \n\
                number of sources={self.src_points.shape[0]}, \n\
                number of receivers={self.rec_points.shape[0]}"

    @classmethod
    def read(cls, fname: str, dist_in_data=False, **kwargs):
        """Read source <--> receiver file to pandas.DataFrame

        :param fname: Path to src_rec file
        :type fname: str
        :param dist_in_data: Whether distance is included in the src_rec file
        :type dist_in_data: bool
        :return: class of SrcRec
        :rtype: SrcRec
        """
        sr = cls(fname, **kwargs)
        alldf = pd.read_table(fname, sep='\s+|\t', engine='python',
                              header=None, comment="#")

        last_col = 12
        # this is a source line if the last column is not NaN
        sr.src_points = alldf[pd.notna(alldf[last_col])]
        # add weight column if not included
        if sr.src_points.shape[1] == last_col+1:
            # add another column for weight
            sr.src_points.loc[:, last_col+1] = 1.0

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

            # extract the rows if the last_col is not NaN and the 12th column is NaN
            sr.rec_points = alldf[(alldf[last_col].notna())
                                & (alldf[12].isna())].reset_index(drop=True)

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

        return sr

    def write(self, fname='src_rec_file'):
        """Write sources and receivers to ASCII file for TomoATT

        :param fname: Path to the src_rec file, defaults to 'src_rec_file'
        :type fname: str, optional
        """
        with open(fname, 'w') as f:
            for idx, src in self.src_points.iterrows():
                time_lst = src['origin_time'].strftime('%Y_%m_%d_%H_%M_%S.%f').split('_')
                f.write('{:d} {} {} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {} {} {:.4f}\n'.format(
                    idx, *time_lst, src['evla'], src['evlo'], src['evdp'],
                    src['mag'], src['num_rec'], src['event_id'], src['weight']
                ))
                if self.src_only:
                    continue
                rec_data = self.rec_points[self.rec_points['src_index']==idx]
                for _, rec in rec_data.iterrows():
                    f.write('{:d} {:d} {} {:6.4f} {:6.4f} {:6.4f} {} {:6.4f}\n'.format(
                        idx, rec['rec_index'], rec['staname'], rec['stla'],
                        rec['stlo'], rec['stel'], rec['phase'], rec['tt']
                    ))

if __name__ == '__main__':
    sr = SrcRec.read('src_rec_file_checker_data_test1.dat_noised_evweighted')
    sr.write()
    print(sr.rec_points)
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
    
    @classmethod
    def read(cls, fname:str, **kwargs):
        """Read source <--> receiver file to pandas.DataFrame

        :param fname: Path to src_rec file
        :type fname: str
        :return: class of SrcRec
        :rtype: SrcRec
        """
        sr = cls(fname, **kwargs)
        alldf = pd.read_table(fname, sep='\s+|\t', engine='python', header=None)
        sr.src_points = alldf[pd.notna(alldf[13])]
        sr.src_points.index = sr.src_points[0]
        if sr.src_points.shape[1] == 13:
            sr.src_points = pd.concat([sr.src_points, pd.Series([1.0]*sr.src_points.shape[0])])
        datedf = sr.src_points[sr.src_points.columns[1:7]]
        type_dict = {
            1: int,
            2: int,
            3: int,
            4: int,
            5: int,
            6: float,
        }
        datedf = datedf.astype(type_dict)
        dateseris = datedf.astype(str).apply(lambda x: '.'.join(x), axis=1).apply(
            pd.to_datetime, format='%Y.%m.%d.%H.%M.%S.%f')
        dateseris.name = 'origin_time'
        src_data = sr.src_points[sr.src_points.columns[7:]]
        src_data.columns = ['evla', 'evlo',
                            'evdp', 'mag', 'num_rec', 
                            'event_id', 'weight',]
        type_dict = {
            'evla': float,
            'evlo': float,
            'evdp': float,
            'mag': float,
            'num_rec': int,
            'event_id': str,
            'weight': float
        }
        src_data = src_data.astype(type_dict)
        sr.src_points = pd.concat([dateseris, src_data], axis=1)
        if not sr.src_only:
            sr.rec_points = alldf[pd.isna(alldf[8])].reset_index(drop=True)
            sr.rec_points = sr.rec_points[sr.rec_points.columns[0:8]]
            sr.rec_points.columns = [
                'src_index', 'rec_index', 'staname', 'stla', 'stlo', 'stel', 'phase', 'tt'
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
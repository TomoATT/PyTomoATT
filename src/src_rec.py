import re
import pandas as pd
from datetime import datetime

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
        self.data = pd.DataFrame(columns=[
            'origin_time', 'evla', 'evlo',
            'evdp', 'mag', 'num_rec', 
            'event_id', 'weight', 'rec_data'
        ])
    
    @classmethod
    def read(cls, fname:str, **kwargs):
        """Read source <--> receiver file to pandas.DataFrame

        :param fname: Path to src_rec file
        :type fname: str
        :return: class of SrcRec
        :rtype: SrcRec
        """
        sr = cls(fname, **kwargs)
        with open(fname) as f:
            lines = f.readlines()
        for line in lines:
            line_sp = line.split()
            if len(line_sp) >= 13:
                ot = datetime.strptime(
                    '.'.join(line_sp[1:7]),
                    '%Y.%m.%d.%H.%M.%S.%f')
                evinfo = [float(ss) for ss in line_sp[7:11]]
                num_rec = int(line_sp[11])
                src_info = [
                    ot, *evinfo, num_rec, line_sp[12]
                ]
                rec_data = pd.DataFrame(
                    columns=['staname', 'stla', 'stlo', 'stel', 'phase', 'tt']
                )
                if len(line_sp) == 13:
                    src_info += [1.0, rec_data]
                else:
                    src_info += [float(line_sp[13]), rec_data]
                sr.data.loc[int(line_sp[0])] = src_info
            else:
                if sr.src_only:
                    continue
                stinfo = [float(v) for v in line_sp[3:6]]
                sr.data.loc[int(line_sp[0])]['rec_data'].loc[int(line_sp[1])] = \
                    [line_sp[2], *stinfo, line_sp[6], float(line_sp[7])]
        return sr
    
    def write(self, fname='src_rec_file'):
        """Write sources and receivers to ASCII file for TomoATT

        :param fname: Path to the src_rec file, defaults to 'src_rec_file'
        :type fname: str, optional
        """
        with open(fname, 'w') as f:
            for idx, src in self.data.iterrows():
                time_lst = src['origin_time'].strftime('%Y_%m_%d_%H_%M_%S.%f').split('_')
                f.write('{:d} {} {} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {} {} {:.4f}\n'.format(
                    idx, *time_lst, src['evla'], src['evlo'], src['evdp'],
                    src['mag'], src['num_rec'], src['event_id'], src['weight'] 
                ))
                if self.src_only:
                    continue
                for ridx, rec in src['rec_data'].iterrows():
                    f.write('{:d} {:d} {:6.4f} {:6.4f} {:6.4f} {} {:6.4f}\n'.format(
                        idx, ridx, rec['stla'], rec['stlo'], rec['stel'], rec['phase'], rec['tt']
                    ))

if __name__ == '__main__':
    sr = SrcRec.read('src_rec_file_eg')
    sr.write()
    print(sr.data.loc[0]['rec_data'])
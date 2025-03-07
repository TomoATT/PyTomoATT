import numpy as np
import tqdm
import pandas as pd
from .distaz import DistAZ
from .setuplog import SetupLog
from .utils.src_rec_utils import define_rec_cols, setup_rec_points_dd, \
                                 get_rec_points_types, update_position, \
                                 download_src_rec_file
from sklearn.metrics.pairwise import haversine_distances
import copy
from io import StringIO
import os

pd.options.mode.chained_assignment = None  # default='warn'


class SrcRec:
    """
    I/O for source <--> receiver file

    :param fname: Path to src_rec file
    :type fname: str
    :param src_only: Whether to read only source information, defaults to False
    :type src_only: bool, optional
    """

    def __init__(self, fname: str, src_only=False) -> None:
        """ """
        self.src_only = src_only
        self.src_points = pd.DataFrame()
        self.rec_points = pd.DataFrame()
        self.rec_points_cr = pd.DataFrame()
        self.rec_points_cs = pd.DataFrame()
        self.sources = pd.DataFrame()
        self.receivers = pd.DataFrame()
        self.fnames = [fname]
        self.log = SetupLog()

    def __repr__(self):
        return f"PyTomoATT SrcRec Object: \n\
                fnames={self.fnames}, \n\
                src_only={self.src_only}, \n\
                number of sources={self.sources.shape[0]}, \n\
                number of receivers={self.receivers.shape[0]}\n\
                number of traces={self.rec_points.shape[0]+self.rec_points_cs.shape[0]+self.rec_points_cr.shape[0]}"

    @property
    def src_points(self):
        """
        Return a DataFrame of all sources

        :return: All sources
        :rtype: pandas.DataFrame

        Sources contain 8 columns:

        ================ ===================================================
        Column            Description
        ================ ===================================================
        ``origin_time``  Origin time of the source
        ``evla``         Latitude of the source
        ``evlo``         Longitude of the source
        ``evdp``         Focal depth
        ``mag``          Magnitude of the source
        ``num_rec``      Number of receivers that recorded the source
        ``event_id``     ID of the source
        ``weight``       Weight of the source applied on objective function
        ================ ===================================================
        """
        return self._src_points

    @src_points.setter
    def src_points(self, value):
        if value is None or isinstance(value, pd.DataFrame):
            self._src_points = value
            if not self._src_points.empty:
                try:
                    self._src_points = self._src_points.astype(
                        {
                            "evla": float,
                            "evlo": float,
                            "evdp": float,
                            "mag": float,
                            "num_rec": int,
                            "event_id": str,
                            "weight": float,
                        }
                    )
                except:
                    pass
            self._src_points.index.name = "src_index"
        else:
            raise TypeError("src_points should be in DataFrame")

    @property
    def rec_points(self):
        """
        Return a DataFrame of all receivers

        :return: All receivers
        :rtype: pandas.DataFrame

        Receivers contain 9 ~ 11 columns:

        Common fields
        -----------------

        ================ =====================================================
        Column            Description
        ================ =====================================================
        ``src_index``    Index of source recorded by the receiver
        ``rec_index``    Index of receivers that recorded the same source
        ``staname``      Name of the receiver
        ``stla``         Latitude of the receiver
        ``stlo``         Longitude of the receiver
        ``stel``         Elevation of the receiver
        ``phase``        Phase name
        ``tt``           Travel time of the source receiver pair
        ``weight``       Weight of the receiver applied on objective function
        ================ =====================================================

        Optional fields
        ----------------

        ================ ===========================================================================
        Column            Description
        ================ ===========================================================================
        ``netname``      Name of the network (when ``name_net_and_sta=True`` in ``SrcRec.read``)
        ``dist_deg``     Epicentral distance in deg (when ``dist_in_data=True`` in ``SrcRec.read``)
        ================ ===========================================================================

        """
        return self._rec_points

    @rec_points.setter
    def rec_points(self, value):
        if value is None or isinstance(value, pd.DataFrame):
            self._rec_points = value
        else:
            raise TypeError("rec_points should be in DataFrame")

    @property
    def rec_points_cs(self):
        """
        Return a DataFrame of all common sources

        :return: All common sources
        :rtype: pandas.DataFrame

        Common sources contain 14 columns:

        ================ =====================================================
        Column            Description
        ================ =====================================================
        ``src_index``    Index of source recorded by the receiver
        ``rec_index1``   Index of receivers that recorded the same source
        ``staname1``     Name of the receiver
        ``stla1``        Latitude of the receiver
        ``stlo1``        Longitude of the receiver
        ``stel1``        Elevation of the receiver
        ``rec_index2``   Index of the source recorded by the receiver
        ``staname2``     Name of the receiver
        ``stla2``        Latitude of the receiver
        ``stlo2``        Longitude of the receiver
        ``stel2``        Elevation of the receiver
        ``phase``        Phase name
        ``tt``           Travel time of the source receiver pair
        ``weight``       Weight of the receiver applied on objective function
        ================ =====================================================
        """
        return self._rec_points_cs
    
    @rec_points_cs.setter
    def rec_points_cs(self, value):
        if value is None or isinstance(value, pd.DataFrame):
            self._rec_points_cs = value
        else:
            raise TypeError("rec_points_cs should be in DataFrame")

    @property
    def rec_points_cr(self):
        """
        Return a DataFrame of all common receivers

        :return: All common receivers
        :rtype: pandas.DataFrame

        Common receivers contain 13 columns:

        ================ =====================================================
        Column            Description
        ================ =====================================================
        ``src_index``    Index of source recorded by the receiver
        ``rec_index``    Index of receivers that recorded the same source
        ``staname``      Name of the receiver
        ``stla``         Latitude of the receiver
        ``stlo``         Longitude of the receiver
        ``stel``         Elevation of the receiver
        ``src_index2``   Index of the source recorded by the receiver
        ``event_id2``    ID of the source
        ``evla2``        Latitude of the source
        ``evlo2``        Longitude of the source
        ``evdp2``        Focal depth
        ``phase``        Phase name
        ``tt``           Travel time of the source receiver pair
        ``weight``       Weight of the receiver applied on objective function
        ================ =====================================================
        """
        return self._rec_points_cr

    @rec_points_cr.setter
    def rec_points_cr(self, value):
        if value is None or isinstance(value, pd.DataFrame):
            self._rec_points_cr = value
        else:
            raise TypeError("rec_points_cr should be in DataFrame")

    @classmethod
    def read(cls, fname: str, dist_in_data=False, name_net_and_sta=False, **kwargs):
        """
        Read source <--> receiver file to pandas.DataFrame

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
        if not os.path.exists(fname):
            sr.log.SrcReclog.info("Downloading src_rec file from {}".format(fname))
            src_rec_data = download_src_rec_file(fname)
            if src_rec_data is None:
                sr.log.SrcReclog.error("No src_rec file found")
                return sr
        else:
            src_rec_data = fname 
        alldf = pd.read_csv(
                fname, sep=r"\s+", header=None, comment="#", low_memory=False
            )

        last_col_src = 12
        dd_col = 11
        # this is a source line if the last column is not NaN
        # sr.src_points = alldf[pd.notna(alldf[last_col_src])]
        sr.src_points = alldf[~(alldf[dd_col].astype(str).str.contains("cs")| \
                                alldf[dd_col].astype(str).str.contains("cr")| \
                                pd.isna(alldf[last_col_src]))]
        # add weight column if not included
        if sr.src_points.shape[1] == last_col_src + 1:
            # add another column for weight
            sr.src_points.loc[:, last_col_src + 1] = 1.0

        # src id dataframe
        sr.src_points.index = sr.src_points.iloc[:, 0]
        sr.src_points.index.name = "src_index"

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
            sr.log.SrcReclog.error("please check the date format in the src_rec file")
            # return sr.src_points
            return sr
        dateseris = (
            datedf.astype(str)
            .apply(lambda x: ".".join(x), axis=1)
            .apply(pd.to_datetime, format="%Y.%m.%d.%H.%M.%S.%f")
        )
        dateseris.name = "origin_time"
        # event data dataframe
        src_data = sr.src_points.loc[:, 7:]
        src_data.columns = [
            "evla",
            "evlo",
            "evdp",
            "mag",
            "num_rec",
            "event_id",
            "weight",
        ]
        type_dict = {
            "evla": float,
            "evlo": float,
            "evdp": float,
            "mag": float,
            "num_rec": int,
            "event_id": str,
            "weight": float,
        }
        try:
            src_data = src_data.astype(type_dict)
        except:
            sr.log.SrcReclog.error(
                "please check the event data format in the src_rec file"
            )
            return sr.src_points

        # concat all the 3 dataframes
        sr.src_points = pd.concat([dateseris, src_data], axis=1)

        # read receiver data if not src_only
        if not sr.src_only:
            # number of columns is 8 if distance is not included
            cols, last_col = define_rec_cols(dist_in_data, name_net_and_sta)

            # extract the rows if the last_col is not NaN and the 12th column is NaN
            sr.rec_points = alldf[
                (alldf[last_col].notna()) & \
                (alldf[last_col_src].isna())
            ].reset_index(drop=True)

            # add weight column if not included
            if sr.rec_points.loc[:, last_col + 1].isna().all():
                sr.rec_points.loc[:, last_col + 1] = 1.0

            # warning if weigh value is greater than 10
            if (sr.rec_points.loc[:, last_col + 1] > 10).any():
                sr.log.SrcReclog.warning(
                    """
at least one weight value is greater than 10.
Probably your src_rec file includes distance data.
In this case, please set dist_in_data=True and read again."""
                )

            # extract only the first part of columns (cut unnecessary columns)
            sr.rec_points = sr.rec_points.loc[:, : last_col + 1]

            # define column names
            sr.rec_points.columns = cols

            sr.rec_points = sr.rec_points.astype(get_rec_points_types(dist_in_data))
            
            if name_net_and_sta:
                # concatenate network and station name with "."
                sr.rec_points["staname"] = (
                    sr.rec_points["netname"] + "." + sr.rec_points["staname"]
                )
                # drop network name column
                sr.rec_points.drop("netname", axis=1, inplace=True)
                # define src and rec list

            # read common receiver data
            last_col = 12
            sr.rec_points_cr = alldf[
                alldf[11].astype(str).str.contains("cr")
            ].reset_index(drop=True)
            if sr.rec_points_cr.shape[1] == last_col + 1:
                if not sr.rec_points_cr.empty:
                    # add another column for weight
                    sr.rec_points_cr.loc[:, last_col + 1] = 1.0
                else:
                    sr.rec_points_cr[last_col + 1] = pd.Series()
            elif sr.rec_points_cr.shape[1] not in [last_col+1, last_col+2]:
                sr.log.SrcReclog.error(
                    f"Only common receiver data with {last_col+1} or {last_col+2} columns are supported, "
                    "please check the format of common receiver data"
                )
                return sr
            # set column names and types
            cols, data_type = setup_rec_points_dd(type='cr')
            sr.rec_points_cr.columns = cols
            sr.rec_points_cr = sr.rec_points_cr.astype(data_type)

            # read common source data
            sr.rec_points_cs = alldf[
                alldf[11].astype(str).str.contains("cs")
            ].reset_index(drop=True)
            if sr.rec_points_cs.shape[1] == last_col + 1:
                if not sr.rec_points_cs.empty:
                    # add another column for weight
                    sr.rec_points_cs.loc[:, last_col + 1] = 1.0
                else:
                    sr.rec_points_cs[last_col + 1] = pd.Series()
            elif sr.rec_points_cs.shape[1] not in [last_col+1, last_col+2]:
                sr.log.SrcReclog.error(
                    f"Only common source data with {last_col+1} or {last_col+2} columns are supported, "
                    "please check the format of common source data"
                )
                return sr
            # set column names and types
            cols, data_type = setup_rec_points_dd(type='cs')
            sr.rec_points_cs.columns = cols
            sr.rec_points_cs = sr.rec_points_cs.astype(data_type)

            sr.update_unique_src_rec()
        return sr

    def write(self, fname="src_rec_file"):
        """
        Write sources and receivers to ASCII file for TomoATT

        :param fname: Path to the src_rec file, defaults to 'src_rec_file'
        :type fname: str, optional
        """
        output = StringIO()
        src_points = self.src_points
        rec_points = self.rec_points
        rec_points_cs = self.rec_points_cs
        rec_points_cr = self.rec_points_cr

        for src in tqdm.tqdm(
            src_points.itertuples(),
            total=src_points.shape[0],
            desc="Writing src_rec file",
        ):
            idx = src.Index
            time_lst = (
                src.origin_time.strftime("%Y_%m_%d_%H_%M_%S.%f").split("_")
            )
            output.write("{:d} {} {} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {} {} {:.4f}\n".format(
                    idx,
                    *time_lst,
                    src.evla,
                    src.evlo,
                    src.evdp,
                    src.mag,
                    src.num_rec,
                    src.event_id,
                    src.weight,
                ))

            if self.src_only:
                continue

            rec_data = rec_points[rec_points["src_index"] == idx]
            for rec in rec_data.itertuples():
                output.write("   {:d} {:d} {} {:6.4f} {:6.4f} {:6.4f} {} {:6.4f} {:6.4f}\n".format(
                        idx,
                        rec.rec_index,
                        rec.staname,
                        rec.stla,
                        rec.stlo,
                        rec.stel,
                        rec.phase,
                        rec.tt,
                        rec.weight,
                    ))

            if not rec_points_cs.empty:
                rec_data = rec_points_cs[rec_points_cs["src_index"] == idx]
                for rec in rec_data.itertuples():
                    output.write("   {:d} {:d} {} {:6.4f} {:6.4f} {:6.4f} {:d} {} {:6.4f} {:6.4f} {:6.4f} {} {:.4f} {:6.4f}\n".format(
                            idx,
                            rec.rec_index1,
                            rec.staname1,
                            rec.stla1,
                            rec.stlo1,
                            rec.stel1,
                            rec.rec_index2,
                            rec.staname2,
                            rec.stla2,
                            rec.stlo2,
                            rec.stel2,
                            rec.phase,
                            rec.tt,
                            rec.weight,
                        ))
            if not rec_points_cr.empty:
                rec_data = rec_points_cr[rec_points_cr["src_index"] == idx]
                for rec in rec_data.itertuples():
                    output.write("   {:d} {:d} {} {:6.4f} {:6.4f} {:6.4f} {:d} {} {:6.4f} {:6.4f} {:6.4f} {} {:.4f} {:6.4f}\n".format(
                            idx,
                            rec.rec_index,
                            rec.staname,
                            rec.stla,
                            rec.stlo,
                            rec.stel,
                            rec.src_index2,
                            rec.event_id2,
                            rec.evla2,
                            rec.evlo2,
                            rec.evdp2,
                            rec.phase,
                            rec.tt,
                            rec.weight,
                        ))
        with open(fname, "w") as f:
            f.write(output.getvalue())

    def copy(self):
        """
        Return a copy of SrcRec object

        :return: Copy of SrcRec object
        :rtype: SrcRec
        """
        return copy.deepcopy(self)
    
    def update_unique_src_rec(self):
        """
        Update unique sources and receivers

        The unique sources and receivers are stored 
        in ``SrcRec.sources`` and ``SrcRec.receivers`` respectively.
        """
        # get sources
        src_col = ["event_id", "evla", "evlo", "evdp"]
        sources = self.src_points[src_col].values
        if not self.rec_points_cr.empty:
            sources= np.vstack(
                [sources, self.rec_points_cr[
                ["event_id2", "evla2", "evlo2", "evdp2"]
                ].values])
        self.sources = pd.DataFrame(
            sources, columns=src_col
        ).drop_duplicates(ignore_index=True)
        self.sources = self.sources.astype(
            {
                "evla": float,
                "evlo": float,
                "evdp": float,
            }
        )
        self.sources.index = np.arange(len(self.sources))

        # get receivers
        rec_col = ["staname", "stla", "stlo", "stel"]
        receivers = self.rec_points[rec_col].values
        if not self.rec_points_cs.empty:
            receivers = np.vstack(
                [receivers, self.rec_points_cs[
                ["staname1", "stla1", "stlo1", "stel1"]
            ].values])
            receivers = np.vstack(
                [receivers, self.rec_points_cs[
                ["staname2", "stla2", "stlo2", "stel2"]
            ].values])
        if not self.rec_points_cr.empty:
            receivers = np.vstack(
                [receivers, self.rec_points_cr[
                ["staname", "stla", "stlo", "stel"]
            ].values])
        self.receivers = pd.DataFrame(
            receivers, columns=rec_col
        ).drop_duplicates(ignore_index=True)
        self.receivers = self.receivers.astype(
            {
                "stla": float,
                "stlo": float,
                "stel": float,
            }
        )
        self.receivers.index = np.arange(len(self.receivers))

    def sort(self, by="origin_time"):
        """
        Sort sources by given column

        :param by: Column to sort by, defaults to ``origin_time``
                   available columns are ``origin_time``, ``evla``, ``evlo``, ``evdp``,
                   ``mag``, ``num_rec``, ``event_id``, ``weight``
        :type by: str, optional
        """
        self.src_points.sort_values(by=by, inplace=True)
        self.update()

    def reset_index(self):
        """
        Reset index of source and receivers.
        """
        # self.src_points.index = np.arange(len(self.src_points))
        # use index in self.sources when self.src_points['event_id'] == self.sources['event_id']
        self.sources.index = np.arange(len(self.sources))
        new_index = self.src_points["event_id"].map(
            dict(zip(self.sources["event_id"], self.sources.index))
        )

        # reset src_index to be 0, 1, 2, ... for both src_points and rec_points
        self.rec_points["src_index"] = self.rec_points["src_index"].map(
            dict(zip(self.src_points.index, new_index))
        )
        self.rec_points.reset_index(drop=True, inplace=True)

        if not self.rec_points_cs.empty:
            self.rec_points_cs["src_index"] = self.rec_points_cs["src_index"].map(
                dict(zip(self.src_points.index, new_index))
            )
            self.rec_points_cs.reset_index(drop=True, inplace=True)
        
        if not self.rec_points_cr.empty:
            self.rec_points_cr["src_index"] = self.rec_points_cr["src_index"].map(
                dict(zip(self.src_points.index, new_index))
            )
            # update event_id2
            self.rec_points_cr["src_index2"] = self.rec_points_cr["event_id2"].map(
                dict(zip(self.src_points["event_id"], new_index))
            )
            self.rec_points_cr.reset_index(drop=True, inplace=True)

        self.src_points.set_index(new_index, inplace=True)
        self.src_points.index.name = "src_index"

        # reset rec_index to be 0, 1, 2, ... for rec_points
        self.rec_points["rec_index"] = self.rec_points.groupby("src_index").cumcount()
        if not self.rec_points_cs.empty:
            self.rec_points_cs["rec_index1"] = self.rec_points_cs.groupby("src_index").cumcount()
        if not self.rec_points_cr.empty:
            self.rec_points_cr["rec_index"] = self.rec_points_cr.groupby("src_index").cumcount()

    def append(self, sr):
        """
        Append another SrcRec object to the current one

        :param sr: Another SrcRec object
        :type sr: SrcRec
        """
        if not isinstance(sr, SrcRec):
            raise TypeError("Input must be a SrcRec object")

        if self.src_only != sr.src_only:
            raise ValueError("Cannot append src_only and non-src_only SrcRec objects")

        self.reset_index()
        sr.reset_index()
        
        self.log.SrcReclog.info(f"src_points before appending: {self.src_points.shape[0]}")
        self.log.SrcReclog.info(f"rec_points before appending: {self._count_records()}")
        # number of sources to be added
        n_src_offset = self.src_points.shape[0]

        # add column for source file tag if not included
        if "fname" not in self.src_points.columns:
            self.src_points["fname"] = self.fnames[0]
            self.rec_points["fname"] = self.fnames[0]
        if "fname" not in sr.src_points.columns:
            sr.src_points["fname"] = sr.fnames[0]
            sr.rec_points["fname"] = sr.fnames[0]

        # append src_points
        self.src_points = pd.concat([self.src_points, sr.src_points], ignore_index=True)
        self.src_points.index.name = "src_index"
        # self.src_points.index += 1  # start from 1

        if not self.src_only:
            # update src_index in rec_points
            sr.rec_points["src_index"] += n_src_offset
            # append rec_points
            self.rec_points = pd.concat(
                [self.rec_points, sr.rec_points], ignore_index=True
            )

            # append rec_points_cs
            if not sr.rec_points_cs.empty:
                sr.rec_points_cs["src_index"] += n_src_offset
                self.rec_points_cs = pd.concat(
                    [self.rec_points_cs, sr.rec_points_cs], ignore_index=True
                )
            
            # append rec_points_cr
            if not sr.rec_points_cr.empty:
                sr.rec_points_cr["src_index"] += n_src_offset
                sr.rec_points_cr["src_index2"] += n_src_offset
                self.rec_points_cr = pd.concat(
                    [self.rec_points_cr, sr.rec_points_cr], ignore_index=True
                )

        self.update()
        # store fnames
        self.fnames.extend(sr.fnames)

    def remove_rec_by_new_src(self,):
        """
        remove ``rec_points`` by new ``src_points``
        """
        self.rec_points = self.rec_points[
            self.rec_points["src_index"].isin(self.src_points.index)
        ]
        if not self.rec_points_cs.empty:
            self.rec_points_cs = self.rec_points_cs[
                self.rec_points_cs["src_index"].isin(self.src_points.index)
            ]
        if not self.rec_points_cr.empty:
            self.rec_points_cr = self.rec_points_cr[
                self.rec_points_cr["src_index"].isin(self.src_points.index)
            ]
            self.rec_points_cr = self.rec_points_cr[
                self.rec_points_cr['event_id2'].isin(self.src_points['event_id'])
            ]
          
    def remove_src_by_new_rec(self):
        """
        remove ``src_points`` by new receivers
        """
        self.src_points = self.src_points[
            self.src_points.index.isin(self.rec_points["src_index"])
        ]
        if not self.rec_points_cr.empty:
            self.src_points = pd.concat(
                [self.src_points, self.src_points[
                self.src_points.index.isin(self.rec_points_cr["src_index"])
            ]])
        if not self.rec_points_cs.empty:
            self.src_points = pd.concat( 
                [self.src_points, self.src_points[
                self.src_points.index.isin(self.rec_points_cs["src_index"])
            ]])
        self.src_points = self.src_points.drop_duplicates()

    def update_num_rec(self):
        """
        update num_rec in ``src_points`` by current ``rec_points``
        """
        self.src_points["num_rec"] = self.rec_points.groupby("src_index").size()
        if not self.rec_points_cr.empty:
            num = self.rec_points_cr.groupby("src_index").size()
            self.src_points.loc[num.index, "num_rec"] += num
        if not self.rec_points_cs.empty:
            num = self.rec_points_cs.groupby("src_index").size()
            self.src_points.loc[num.index, "num_rec"] += num

    def update(self):
        """
        Update ``SrcRec.src_points``, ``SrcRec.rec_points``,
        ``SrcRec.rec_points_cr`` and ``SrcRec.rec_points_cs`` with procedures:

        1. remove receivers by new sources
        2. remove sources by new receivers
        3. update num_rec
        4. reset index
        5. update unique sources and receivers
        """
        self.update_unique_src_rec()
        self.remove_rec_by_new_src()
        self.remove_src_by_new_rec()
        self.update_num_rec()
        self.reset_index()
        self.remove_src_by_duplicate_event_id()
        # sort by src_index
        self.src_points.sort_values(by=["src_index"], inplace=True)
        self.rec_points.sort_values(by=["src_index", "rec_index"], inplace=True)
        if not self.rec_points_cr.empty:
            self.rec_points_cr.sort_values(by=["src_index", "rec_index"], inplace=True)
        if not self.rec_points_cs.empty:
            self.rec_points_cs.sort_values(by=["src_index", "rec_index1"], inplace=True)
    
    def remove_src_by_duplicate_event_id(self, keep="first"):
        """
        remove ``src_points`` by duplicated ``event_id``

        :param keep: keep first or last duplicated event_id, defaults to "first"
                     available options are "first" and "last"
        :type keep: str, optional
        """
        self.src_points = self.src_points[
            ~self.src_points["event_id"].duplicated(keep=keep)
        ]
        self.update_num_rec()

    def erase_src_with_no_rec(self):
        """
        erase ``src_points`` with no ``rec_points``
        """
        self.log.SrcReclog.info("src_points before removing: ", self.src_points.shape)
        self.src_points = self.src_points[self.src_points["num_rec"] > 0]
        self.log.SrcReclog.info("src_points after removing: ", self.src_points.shape)

    def erase_duplicate_events(
        self, thre_deg: float, thre_dep: float, thre_time_in_min: float
    ):
        """
        check and count how many events are duplicated,
        under given threshold of distance, depth, and time.

        :param thre_deg: threshold of distance in degree
        :type thre_deg: float
        :param thre_dep: threshold of distance in degree
        :type thre_dep: float
        :param thre_time_in_min: hreshold of time in minutes
        :type thre_time_in_min: float
        """

        # sort event data
        self.src_points.sort_values(by=["origin_time", "evlo", "evla"], inplace=True)

        num_duplicated = 99999
        iter_count = 0

        while num_duplicated > 0:
            # difference with row +1
            self.src_points["diff_evlo+1"] = self.src_points["evlo"].diff().abs()
            self.src_points["diff_evla+1"] = self.src_points["evla"].diff().abs()
            self.src_points["diff_evdp+1"] = self.src_points["evdp"].diff().abs()
            self.src_points["diff_time+1"] = self.src_points["origin_time"].diff().abs()
            self.src_points["diff_nrec+1"] = self.src_points["num_rec"].diff()
            # difference with row -1
            self.src_points["diff_evlo-1"] = (
                self.src_points["evlo"].diff(periods=-1).abs()
            )
            self.src_points["diff_evla-1"] = (
                self.src_points["evla"].diff(periods=-1).abs()
            )
            self.src_points["diff_evdp-1"] = (
                self.src_points["evdp"].diff(periods=-1).abs()
            )
            self.src_points["diff_time-1"] = (
                self.src_points["origin_time"].diff(periods=-1).abs()
            )
            self.src_points["diff_nrec-1"] = self.src_points["num_rec"].diff(periods=-1)

            self.src_points["duplicated+1"] = self.src_points.apply(
                lambda x: 1
                if x["diff_evlo+1"] < thre_deg
                and x["diff_evla+1"] < thre_deg
                and x["diff_evdp+1"] < thre_dep
                and x["diff_time+1"] < pd.Timedelta(minutes=thre_time_in_min)
                else 0,
                axis=1,
            )
            self.src_points["duplicated-1"] = self.src_points.apply(
                lambda x: 1
                if x["diff_evlo-1"] < thre_deg
                and x["diff_evla-1"] < thre_deg
                and x["diff_evdp-1"] < thre_dep
                and x["diff_time-1"] < pd.Timedelta(minutes=thre_time_in_min)
                else 0,
                axis=1,
            )

            # drop rows (duplicated == 1 and diff_nrec <= 0)
            self.src_points = self.src_points[
                ~(
                    (self.src_points["duplicated+1"] == 1)
                    & (self.src_points["diff_nrec+1"] < 0)
                )
            ]
            # drow upper row of (duplicated == 1 and diff_nrec > 0)
            self.src_points = self.src_points[
                ~(
                    (self.src_points["duplicated-1"] == 1)
                    & (self.src_points["diff_nrec-1"] <= 0)
                )
            ]

            # print iterate count and number of rows, number of duplicated rows
            num_duplicated = self.src_points[
                (self.src_points["duplicated+1"] == 1)
                | (self.src_points["duplicated-1"] == 1)
            ].shape[0]
            self.log.SrcReclog.info(
                "iteration: {}; num_duplicated: {}".format(iter_count, num_duplicated)
            )

            iter_count += 1

        # erase all columns starting with diff_*
        self.src_points.drop(
            self.src_points.columns[self.src_points.columns.str.startswith("diff_")],
            axis=1,
            inplace=True,
        )
        # erase all clumuns starting with duplicated
        self.src_points.drop(
            self.src_points.columns[
                self.src_points.columns.str.startswith("duplicated")
            ],
            axis=1,
            inplace=True,
        )
        self.update()

    def select_by_phase(self, phase_list):
        """
        select interested phase and remove others

        :param phase_list: List of phases for travel times used for inversion
        :type phase_list: list of str
        """
        if not isinstance(phase_list, (list, tuple, str)):
            raise TypeError("phase_list should be in list or str")
        if isinstance(phase_list, str):
            phase_list = [phase_list]
        self.log.SrcReclog.info(
            "rec_points before selection: {}".format(self._count_records())
        )
        self.rec_points = self.rec_points[self.rec_points["phase"].isin(phase_list)]
        self.rec_points_cs = self.rec_points_cs[
            self.rec_points_cs["phase"].isin([f'{ph},cs' for ph in phase_list])
        ]
        self.rec_points_cr = self.rec_points_cr[
            self.rec_points_cr["phase"].isin([f'{ph},cr' for ph in phase_list])
        ]
        self.update()
        self.log.SrcReclog.info(
            "rec_points after selection: {}".format(self._count_records())
        )

    def select_by_datetime(self, time_range):
        """
        select sources and station in a time range

        :param time_range: Time range defined as [start_time, end_time]
        :type time_range: iterable
        """
        # select source within this time range.
        self.log.SrcReclog.info(
            "src_points before selection: {}".format(self.src_points.shape[0])
        )
        self.log.SrcReclog.info(
            "rec_points before selection: {}".format(self._count_records())
        )
        self.src_points = self.src_points[
            (self.src_points["origin_time"] >= time_range[0])
            & (self.src_points["origin_time"] <= time_range[1])
        ]
        self.update()
        self.log.SrcReclog.info(
            "src_points after selection: {}".format(self.src_points.shape[0])
        )
        self.log.SrcReclog.info(
            "rec_points after selection: {}".format(self._count_records())
        )

    def remove_specified_recs(self, rec_list):
        """Remove specified receivers

        :param rec_list: List of receivers to be removed
        :type rec_list: list
        """
        self.log.SrcReclog.info(
            "rec_points before removing: {}".format(self.rec_points.shape)
        )
        self.rec_points = self.rec_points[~self.rec_points["staname"].isin(rec_list)]
        self.update()
        self.log.SrcReclog.info(
            "rec_points after removing: {}".format(self.rec_points.shape)
        )

    def select_by_box_region(self, region):
        """
        Select sources and station in a box region

        :param region: Box region defined as ``[lon1, lon2, lat1, lat2]``
        :type region: iterable
        """
        # select source within this region.
        self.log.SrcReclog.info(
            "src_points before selection: {}".format(self.src_points.shape[0])
        )
        self.log.SrcReclog.info(
            "rec_points before selection: {}".format(self._count_records())
        )
        self.src_points = self.src_points[
            (self.src_points["evlo"] >= region[0])
            & (self.src_points["evlo"] <= region[1])
            & (self.src_points["evla"] >= region[2])
            & (self.src_points["evla"] <= region[3])
        ]

        # Remove receivers whose events have been removed
        self.remove_rec_by_new_src()

        # Remove rest receivers out of region.
        self.rec_points = self.rec_points[
            (self.rec_points["stlo"] >= region[0])
            & (self.rec_points["stlo"] <= region[1])
            & (self.rec_points["stla"] >= region[2])
            & (self.rec_points["stla"] <= region[3])
        ]

        # Remove empty sources
        self.update()
        self.log.SrcReclog.info(
            "src_points after selection: {}".format(self.src_points.shape)
        )
        self.log.SrcReclog.info(
            "rec_points after selection: {}".format(self.rec_points.shape)
        )

    def select_by_depth(self, dep_min_max):
        """Select sources in a range of depth

        :param dep_min_max: limit of depth, ``[dep_min, dep_max]``
        :type dep_min_max: sequence
        """
        self.log.SrcReclog.info('src_points before selection: {}'.format(self.src_points.shape))
        self.log.SrcReclog.info(
            "rec_points before selection: {}".format(self.rec_points.shape)
        )
        self.src_points = self.src_points[
            (self.src_points['evdp'] >= dep_min_max[0]) &
            (self.src_points['evdp'] <= dep_min_max[1])
        ]
        self.update()
        self.log.SrcReclog.info('src_points after selection: {}'.format(self.src_points.shape))
        self.log.SrcReclog.info(
            "rec_points after selection: {}".format(self.rec_points.shape)
        )

    def calc_distaz(self):
        """Calculate epicentral distance and azimuth for each receiver"""
        self.rec_points["dist_deg"] = 0.0
        self.rec_points["az"] = 0.0
        self.rec_points["baz"] = 0.0
        rec_group = self.rec_points.groupby("src_index")
        for idx, rec in rec_group:
            da = DistAZ(
                self.src_points.loc[idx]["evla"],
                self.src_points.loc[idx]["evlo"],
                rec["stla"].values,
                rec["stlo"].values,
            )
            self.rec_points.loc[rec.index, "dist_deg"] = da.delta
            self.rec_points.loc[rec.index, "az"] = da.az
            self.rec_points.loc[rec.index, "baz"] = da.baz

    def select_by_distance(self, dist_min_max, recalc_dist=False):
        """Select stations in a range of distance
        
        .. note::
            This criteria only works for absolute travel time data.

        :param dist_min_max: limit of distance in deg, ``[dist_min, dist_max]``
        :type dist_min_max: list or tuple
        """
        self.log.SrcReclog.info(
            "rec_points before selection: {}".format(self._count_records())
        )
        # rec_group = self.rec_points.groupby('src_index')
        if ("dist_deg" not in self.rec_points) or recalc_dist:
            self.log.SrcReclog.info("Calculating epicentral distance...")
            self.calc_distaz()
        elif not recalc_dist:
            pass
        else:
            self.log.SrcReclog.error(
                "No such field of dist, please set up recalc_dist to True"
            )
        # for _, rec in rec_group:
        mask = (self.rec_points["dist_deg"] < dist_min_max[0]) | (
            self.rec_points["dist_deg"] > dist_min_max[1]
        )
        drop_idx = self.rec_points[mask].index
        self.rec_points = self.rec_points.drop(index=drop_idx)
        self.update()
        self.log.SrcReclog.info(
            "rec_points after selection: {}".format(self._count_records())
        )

    def select_by_azi_gap(self, max_azi_gap: float):
        """Select sources with azimuthal gap greater and equal than a number
    
        :param azi_gap: threshold of minimum azimuthal gap
        :type azi_gap: float
        """
        self.log.SrcReclog.info(
            "src_points before selection: {}".format(self.src_points.shape[0])
        )
        self.log.SrcReclog.info(
            "rec_points before selection: {}".format(self._count_records())
        )
        if ("az" not in self.rec_points):
            self.log.SrcReclog.info("Calculating azimuth...")
            self.calc_distaz()
        # calculate maximum azimuthal gap for each source
        def calc_azi_gap(az):
            sorted_az = np.sort(az)
            az_diffs = np.diff(np.concatenate((sorted_az, [sorted_az[0] + 360])))
            return np.max(az_diffs)
        max_gap = self.rec_points.groupby('src_index')['az'].apply(lambda x: calc_azi_gap(x.values))
        self.src_points = self.src_points[(max_gap < max_azi_gap)]     
        
        self.update()
        self.log.SrcReclog.info(
            "src_points after selection: {}".format(self.src_points.shape[0])
        )
        self.log.SrcReclog.info(
            "rec_points after selection: {}".format(self._count_records())
        )

    def select_by_num_rec(self, num_rec: int):
        """select sources with recievers greater and equal than a number

        :param num_rec: threshold of minimum receiver number
        :type num_rec: int
        """
        self.update_num_rec()
        self.log.SrcReclog.info(
            "src_points before selection: {}".format(self.src_points.shape[0])
        )
        self.log.SrcReclog.info(
            "rec_points before selection: {}".format(self._count_records())
        )
        self.src_points = self.src_points[(self.src_points["num_rec"] >= num_rec)]
        # self.remove_rec_by_new_src()
        self.update()
        self.log.SrcReclog.info(
            "src_points after selection: {}".format(self.src_points.shape[0])
        )
        self.log.SrcReclog.info(
            "rec_points after selection: {}".format(self._count_records())
        )

    def _evt_group(self, d_deg: float, d_km: float):
        """group events by grid size

        :param d_deg: grid size along lat and lon in degree
        :type d_deg: float
        :param d_km: grid size along depth axis in km
        :type d_km: float
        """
        # add 'lat_group' and 'lon_group' to src_points by module d_deg
        self.src_points["lat_group"] = self.src_points["evla"].apply(
            lambda x: int(x / d_deg)
        )
        self.src_points["lon_group"] = self.src_points["evlo"].apply(
            lambda x: int(x / d_deg)
        )
        self.src_points["dep_group"] = self.src_points["evdp"].apply(
            lambda x: int(x / d_km)
        )
        # sort src_points by 'lat_group' and 'lon_group' and 'dep_group'
        self.src_points = self.src_points.sort_values(
            by=["lat_group", "lon_group", "dep_group"]
        )

    def select_one_event_in_each_subgrid(self, d_deg: float, d_km: float):
        """select one event in each subgrid

        :param d_deg: grid size along lat and lon in degree
        :type d_deg: float
        :param d_km: grid size along depth axis in km
        :type d_km: float
        """

        self.log.SrcReclog.info(
            "src_points before selection: {}".format(self.src_points.shape[0])
        )
        # self.log.SrcReclog.info("processing... (this may take a few minutes)")

        # store index of src_points as 'src_index'
        # self.src_points["src_index"] = self.src_points.index
        
        # group events by grid size
        self._evt_group(d_deg, d_km)

        # find all events in the same lat_group and lon_group and dep_group
        # and keep only on with largest nrec
        self.src_points = self.src_points.groupby(
            ["lat_group", "lon_group", "dep_group"]
        ).apply(lambda x: x.sort_values(by="num_rec", ascending=False).iloc[0])

        # drop 'lat_group' and 'lon_group' and 'dep_group'
        self.src_points = self.src_points.drop(
            columns=["lat_group", "lon_group", "dep_group"]
        )

        # restore index from 'src_index'
        self.src_points = self.src_points.set_index("src_index")

        # sort src_points by index
        self.src_points = self.src_points.sort_index()

        self.log.SrcReclog.info(
            "src_points after selection: {}".format(self.src_points.shape[0])
        )

        # remove rec_points by new src_points
        # self.remove_rec_by_new_src()
        self.update()

    def box_weighting(self, d_deg: float, d_km: float, obj="both", dd_weight='average'):
        """Weighting sources and receivers by number in each subgrid

        :param d_deg: grid size along lat and lon in degree
        :type d_deg: float
        :param d_km: grid size along depth axis in km, (only used when obj=``src`` or ``both``)
        :type d_km: float
        :param obj: Object to be weighted, options: ``src``, ``rec`` or ``both``, defaults to ``both``
        :type obj: str, optional
        :param dd_weight: Weighting method for double difference, options: ``average``, `multiply`, defaults to ``average``
        """
        if obj == "src":
            self._box_weighting_ev(d_deg, d_km)
        elif obj == "rec":
            self._box_weighting_st(d_deg, dd_weight)
        elif obj == "both":
            self._box_weighting_ev(d_deg, d_km)
            self._box_weighting_st(d_deg, dd_weight)
        else:
            self.log.SrcReclog.error(
                "Only 'src', 'rec' or 'both' are supported for obj"
            )

    def _box_weighting_ev(self, d_deg: float, d_km: float):
        """Weighting sources by number of sources in each subgrid

        :param d_deg: grid size along lat and lon in degree
        :type d_deg: float
        :param d_km: grid size along depth axis in km
        :type d_km: float
        """
        self.log.SrcReclog.info(
            "Box weighting for sources: d_deg={}, d_km={}".format(d_deg, d_km)
        )

        # group events by grid size
        self._evt_group(d_deg, d_km)

        # count num of sources in the same lat_group and lon_group and dep_group
        self.src_points["num_sources"] = self.src_points.groupby(
            ["lat_group", "lon_group", "dep_group"]
        )["lat_group"].transform("count")

        # calculate weight for each event
        self.src_points["weight"] = 1 / np.sqrt(self.src_points["num_sources"])

        # assign weight to sources
        self.sources["weight"] = self.sources.apply(
            lambda x: self.src_points[
                (self.src_points["event_id"] == x["event_id"])
            ]["weight"].values[0],
            axis=1,
        )

        # drop 'lat_group' and 'lon_group' and 'dep_group'
        self.src_points = self.src_points.drop(
            columns=["lat_group", "lon_group", "dep_group", "num_sources"]
        )

    def _box_weighting_st(self, d_deg: float, dd_weight='average'):
        """Weighting receivers by number of sources in each subgrid

        :param d_deg: grid size along lat and lon in degree
        :type d_deg: float
        """
        self.log.SrcReclog.info(
            "Box weighting for receivers: d_deg={}".format(d_deg)
        )

        # group events by grid size
        self.receivers["lat_group"] = self.receivers["stla"].apply(
            lambda x: int(x / d_deg)
        )
        self.receivers["lon_group"] = self.receivers["stlo"].apply(
            lambda x: int(x / d_deg)
        )

        # count num of sources in the same lat_group and lon_group
        self.receivers["num_receivers"] = self.receivers.groupby(
            ["lat_group", "lon_group"]
        )["lat_group"].transform("count")

        # calculate weight for each event
        self.receivers["weight"] = 1 / np.sqrt(self.receivers["num_receivers"])

        # assign weight to rec_points
        self.rec_points["weight"] = self.rec_points.apply(
            lambda x: self.receivers[
                (self.receivers["staname"] == x["staname"])
            ]["weight"].values[0],
            axis=1,
        )

        # assign weight to rec_points_cs
        # the weight is the average of the two receivers
        if not self.rec_points_cs.empty:
            self.rec_points_cs["weight"] = self.rec_points_cs.apply(
                lambda x: self._cal_dd_weight(
                    self.receivers[
                        (self.receivers["staname"] == x["staname1"])
                    ]["weight"].values[0],
                    self.receivers[
                        (self.receivers["staname"] == x["staname2"])
                    ]["weight"].values[0],
                    dd_weight
                ),
                axis=1,
            )
        
        # assign weight to rec_points_cr
        # the weight is the average of the one receiver and the other source
        if not self.rec_points_cr.empty:
            self.rec_points_cr["weight"] = self.rec_points_cr.apply(
                lambda x: self._cal_dd_weight(
                    self.receivers[
                        (self.receivers["staname"] == x["staname"])
                    ]["weight"].values[0],
                    self.src_points[
                        (self.src_points["event_id"] == x["event_id2"])
                    ]["weight"].values[0],
                    dd_weight
                ),
                axis=1,
            )

        # drop 'lat_group' and 'lon_group'
        self.receivers = self.receivers.drop(
            columns=["lat_group", "lon_group", "num_receivers"]
        )

    def count_events_per_station(self):
        """
        count events per station
        """
        # count the number of staname
        self.rec_points["num_events"] = (
            self.rec_points.groupby("staname").cumcount() + 1
        )
        # reflect the total number of events for each station
        self.rec_points["num_events"] = self.rec_points.groupby("staname")[
            "num_events"
        ].transform("max")

    def generate_double_difference(self, type='cs', max_azi_gap=15, max_dist_gap=2.5, dd_weight='average', recalc_baz=False):
        """
        Generate double difference data

        :param type: Type of double difference, options: ``cr``, ``cs`` or ``both``, defaults to ``cs``
        :type type: str, optional
        :param max_azi_gap: Maximum azimuthal gap for selecting events, defaults to 15
        :type max_azi_gap: float, optional
        :param max_dist_gap: Maximum distance gap for selecting events, defaults to 2.5
        :type max_dist_gap: float, optional
        :param dd_weight: Weighting method for double difference, options: ``average``, ``multiply``, defaults to ``average``
        :param recalc_baz: Recalculate azimuth and back azimuth, defaults to ``False``
        :type recalc_baz: bool, optional

        ``self.rec_points_cr`` or ``self.rec_points_cs`` are generated
        """

        if ("dist_deg" not in self.rec_points or "baz" not in self.rec_points) or recalc_baz:
            self.calc_distaz()

        if type == 'cs':
            self._generate_cs(max_azi_gap, max_dist_gap, dd_weight)
        elif type == 'cr':
            self._generate_cr(max_azi_gap, max_dist_gap, dd_weight)
        elif type == 'both':
            self._generate_cs(max_azi_gap, max_dist_gap, dd_weight)
            self._generate_cr(max_azi_gap, max_dist_gap, dd_weight)
        else:
            self.log.SrcReclog.error(
                "Only 'cs', 'cr' or 'both' are supported for type of double difference"
            )

        self.update()

    def _generate_cs(self, max_azi_gap, max_dist_gap, dd_weight='average'):
        names, _ = setup_rec_points_dd('cs')
        self.rec_points_cs = pd.DataFrame(columns=names)
        src = self.rec_points.groupby("src_index")
        dd_data = []
        for idx, rec_data in tqdm.tqdm(
                src, total=len(src),
                desc="Generating cs",
            ):
            if rec_data.shape[0] < 2:
                continue
            baz_values = rec_data['baz'].values
            dist_deg_values = rec_data['dist_deg'].values
            rec_indices = rec_data['rec_index'].values
            stanames = rec_data['staname'].values
            stlas = rec_data['stla'].values
            stlos = rec_data['stlo'].values
            stels = rec_data['stel'].values
            tts = rec_data['tt'].values
            phases = rec_data['phase'].values
            weights = rec_data['weight'].values
            for i in range(rec_data.shape[0]):
                for j in range(i + 1, rec_data.shape[0]):
                    if abs(baz_values[i] - baz_values[j]) < max_azi_gap and \
                       abs(dist_deg_values[i] - dist_deg_values[j]) < max_dist_gap and \
                       phases[i] == phases[j]:
                        data_row = {
                            "src_index": idx,
                            "rec_index1": rec_indices[i],
                            "staname1": stanames[i],
                            "stla1": stlas[i],
                            "stlo1": stlos[i],
                            "stel1": stels[i],
                            "rec_index2": rec_indices[j],
                            "staname2": stanames[j],
                            "stla2": stlas[j],
                            "stlo2": stlos[j],
                            "stel2": stels[j],
                            "phase": f"{phases[i]},cs",
                            "tt": tts[i] - tts[j],
                            "weight": self._cal_dd_weight(weights[i], weights[j], dd_weight),
                        }
                        # set src_index to index
                        dd_data.append(data_row)
        if dd_data:
            self.rec_points_cs = pd.DataFrame(dd_data)
        self.log.SrcReclog.info(
            "rec_points_cs after generation: {}".format(self.rec_points_cs.shape)
        )

    def _generate_cr(self, max_azi_gap, max_dist_gap, dd_weight='average'):
        names, _ = setup_rec_points_dd('cr')
        self.rec_points_cr = pd.DataFrame(columns=names)
        src_id = self.src_points["event_id"].values
        src_la = self.src_points["evla"].values
        src_lo = self.src_points["evlo"].values
        src_dp = self.src_points["evdp"].values
        src_weights = self.src_points["weight"].values
        results = []
        for rec in tqdm.tqdm(
                self.receivers.itertuples(index=False),
                total=len(self.receivers),
                desc="Generating cr"
            ):
            rec_data = self.rec_points[self.rec_points["staname"] == rec.staname]
            if rec_data.shape[0] < 2:
                continue
            baz_values = rec_data['baz'].values
            dist_deg_values = rec_data['dist_deg'].values
            rec_indices = rec_data['rec_index'].values
            src_indices = rec_data['src_index'].values
            rec_weights = rec_data['weight'].values
            rec_phases = rec_data['phase'].values
            tts = rec_data['tt'].values
            for i in range(rec_data.shape[0]):
                for j in range(i + 1, rec_data.shape[0]):
                    src_index = src_indices[j]
                    if abs(baz_values[i] - baz_values[j]) < max_azi_gap and \
                       abs(dist_deg_values[i] - dist_deg_values[j]) < max_dist_gap and \
                       rec_phases[i] == rec_phases[j]:
                        data_row = {
                            "src_index": src_indices[i],
                            "rec_index": rec_indices[i],
                            "staname": rec.staname,
                            "stla": rec.stla,
                            "stlo": rec.stlo,
                            "stel": rec.stel,
                            "src_index2": src_index,
                            "event_id2": src_id[src_index],
                            "evla2": src_la[src_index],
                            "evlo2": src_lo[src_index],
                            "evdp2": src_dp[src_index],
                            "phase": f"{rec_phases[i]},cr",
                            "tt": tts[i] - tts[j],
                            "weight": self._cal_dd_weight(src_weights[src_index], rec_weights[i], dd_weight),
                        }
                        results.append(data_row)

        if results:
            self.rec_points_cr = pd.DataFrame(results)
        self.log.SrcReclog.info(
            "rec_points_cr after generation: {}".format(self.rec_points_cr.shape)
        )

    def _count_records(self):
        count = 0
        count += self.rec_points.shape[0]
        count += self.rec_points_cs.shape[0]
        count += self.rec_points_cr.shape[0]
        return count

    def _calc_weights(self, lat, lon, scale):
        points = pd.concat([lon, lat], axis=1)
        points_rad = points * (np.pi / 180)
        dist = haversine_distances(points_rad) * 6371.0 / 111.19
        dist_ref = scale * np.mean(dist)
        om = np.exp(-((dist / dist_ref) ** 2)) * points.shape[0]
        return 1 / np.mean(om, axis=0)
    
    def _cal_dd_weight(self, w1, w2, dd_weight='average'):
        if dd_weight == "average":
            return (w1 + w2) / 2
        elif dd_weight == "multiply":
            return w1 * w2
        else:
            raise ValueError("Only 'average' or 'multiply' are supported for dd_weight")

    def geo_weighting(self, scale=0.5, obj="both", dd_weight="average"):
        """Calculating geographical weights for sources

        :param scale: Scale of reference distance parameter. 
                      See equation 22 in Ruan et al., (2019). The reference distance is given by ``scale* dis_average``, defaults to 0.5
        :type scale: float, optional
        :param obj: Object to be weighted, options: ``src``, ``rec`` or ``both``, defaults to ``both``
        :type obj: str, optional
        :param dd_weight: Weighting method for double difference data, options: ``average`` or ``multiply``, defaults to ``average``
        """

        if obj == "src" or obj == "both":
            self.src_points["weight"] = self._calc_weights(
                self.src_points["evla"], self.src_points["evlo"], scale
            )
            # assign weight to sources
            self.sources["weight"] = self.sources.apply(
                lambda x: self.src_points[
                    (self.src_points["event_id"] == x["event_id"])
                ]["weight"].values[0],
                axis=1,
            )
        if obj == "rec" or obj == "both":
            weights = self._calc_weights(
                self.receivers['stla'],
                self.receivers['stlo'],
                scale
            )
            # apply weights to rec_points
            self.receivers['weight'] = weights
            for row in self.receivers.itertuples(index=False):
                self.rec_points.loc[self.rec_points['staname'] == row.staname, 'weight'] = row.weight

            if not self.rec_points_cs.empty:
                for row in self.rec_points_cs.itertuples(index=True):
                    w1 = self.receivers.loc[self.receivers['staname'] == row.staname1, 'weight'].values[0]
                    w2 = self.receivers.loc[self.receivers['staname'] == row.staname2, 'weight'].values[0]
                    self.rec_points_cs.loc[row.Index, 'weight'] = self._cal_dd_weight(w1, w2, dd_weight)

            if not self.rec_points_cr.empty:
                for row in self.rec_points_cr.itertuples(index=True):
                    w1 = self.receivers.loc[self.receivers['staname'] == row.staname, 'weight'].values[0]
                    w2 = self.src_points.loc[self.src_points['event_id'] == row.event_id2, 'weight'].values[0]
                    self.rec_points_cr.loc[row.Index, 'weight'] = self._cal_dd_weight(w1, w2, dd_weight)

    def add_noise(self, range_in_sec=0.1, mean_in_sec=0.0, shape="gaussian"):
        """Add random noise on travel time

        :param mean_in_sec: Mean of the noise in sec, defaults to 0.0
        :type mean_in_sec: float, optional
        :param range_in_sec: Maximun noise in sec, defaults to 0.1
        :type range_in_sec: float, optional
        :param shape: shape of the noise distribution probability
        :type shape: str. options: gaussian or uniform
        """
        self.log.SrcReclog.info(f"Adding {shape} noise to travel time data...")
        for rec_type in [self.rec_points, self.rec_points_cs, self.rec_points_cr]:
            if rec_type.empty:
                continue
            if rec_type.equals(self.rec_points_cs) or rec_type.equals(self.rec_points_cr):
                range_in_sec = range_in_sec * np.sqrt(2)
            if shape == "uniform":
                noise = np.random.uniform(
                        low=mean_in_sec-range_in_sec, high=mean_in_sec+range_in_sec, size=rec_type.shape[0]
                    )
            elif shape == "gaussian":
                noise = np.random.normal(
                    loc=mean_in_sec, scale=range_in_sec, size=rec_type.shape[0]
                )
            rec_type["tt"] += noise

    def add_noise_to_source(self, lat_pert=0.1, lon_pert=0.1, depth_pert=10, tau_pert=0.5):
        """Add random noise on source location

        :param lat_pert: Maximum perturbation on latitude in degree, defaults to 0.1
        :type lat_pert: float, optional
        :param lon_pert: Maximum perturbation on longitude in degree, defaults to 0.1
        :type lon_pert: float, optional
        :param depth_pert: Maximum perturbation on depth in km, defaults to 10
        :type depth_pert: float, optional
        :param tau_pert: Maximum perturbation on origin time in sec, defaults to 0.0
        :type tau_pert: float, optional
        """
        self.log.SrcReclog.info("Adding noise on source location...")
        self.src_points["evla"] += np.random.uniform(-lat_pert, lat_pert, self.src_points.shape[0])
        self.src_points["evlo"] += np.random.uniform(-lon_pert, lon_pert, self.src_points.shape[0])
        self.src_points["evdp"] += np.random.uniform(-depth_pert, depth_pert, self.src_points.shape[0])
        self.src_points["origin_time"] +=  pd.to_timedelta(np.random.uniform(-tau_pert, tau_pert, self.src_points.shape[0]))

    def rotate(self, clat:float, clon:float, angle:float, reverse=False):
        """Rotate sources and receivers around a center point

        :param clat: Latitude of the center
        :type clat: float
        :param clon: Longitude of the center
        :type clon: float
        :param angle: anti-clockwise angle in degree
        :type angle: float
        """
        from .utils.rotate import rtp_rotation, rtp_rotation_reverse

        rotation_func = rtp_rotation_reverse if reverse else rtp_rotation

        self.sources["evla"], self.sources["evlo"] = rotation_func(
            self.sources["evla"].to_numpy(), self.sources["evlo"].to_numpy(), clat, clon, angle
        )

        self.receivers["stla"], self.receivers["stlo"] = rotation_func(
            self.receivers["stla"].to_numpy(), self.receivers["stlo"].to_numpy(), clat, clon, angle
        )

        update_position(self)

    def to_utm(self, zone):
        """Convert sources and receivers to UTM coordinates

        :param zone: UTM zone number
        :type zone: int
        """
        from pyproj import Proj

        latlon2utm = Proj(proj="utm", zone=zone, ellps="WGS84")

        self.sources["evlo"], self.sources["evla"] = latlon2utm(
            self.sources["evlo"], self.sources["evla"]
        )
        self.receivers["stlo"], self.receivers["stla"] = latlon2utm(
            self.receivers["stlo"], self.receivers["stla"]
        )

        update_position(self)
        
    def write_receivers(self, fname: str):
        """
        Write receivers to a txt file.

        :param fname: Path to output txt file of receivers
        """
        self.receivers.to_csv(fname, sep=" ", header=False, index=False)

    def write_sources(self, fname: str):
        """
        Write sources to a txt file.

        :param fname: Path to output txt file of sources
        """
        self.sources.to_csv(fname, sep=" ", header=False, index=False)

    @classmethod
    def from_seispy(cls, rf_path: str):
        """Read and convert source and station information from
        receiver function data calculated by Seispy

        :param rf_path: Path to receiver functions calculated by Seispy
        :type rf_path: str
        :return: New instance of class SrcRec
        :rtype: SrcRec
        """
        from .io.seispy import Seispy

        sr = cls("")
        # Initial an instance of Seispy
        seispyio = Seispy(rf_path)

        # Load station info from SAC header
        seispyio._load_sta_info()

        # Read finallist.dat
        seispyio.get_rf_info()

        # Convert to SrcRec format
        sr.src_points, sr.rec_points = seispyio.to_src_rec_points()

        # update number of receivers
        sr.update_num_rec()

        return sr

    # implemented in vis.py
    def plot(self, weight=False, fname=None):
        """Plot source and receivers for preview

        :param weight: Draw colors of weights, defaults to False
        :type weight: bool, optional
        :param fname: Path to output file, defaults to None
        :type fname: str, optional
        :return: matplotlib figure
        :rtype: matplotlib.figure.Figure
        """
        from .vis import plot_srcrec

        return plot_srcrec(self, weight=weight, fname=fname)


if __name__ == "__main__":
    sr = SrcRec.read("src_rec_file_checker_data_test1.dat_noised_evweighted")
    sr.write()
    print(sr.rec_points)

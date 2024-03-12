import numpy as np
import tqdm
import pandas as pd
from .distaz import DistAZ
from .setuplog import SetupLog
from .utils import WGS84_to_cartesian
from scipy.spatial import distance
from sklearn.metrics.pairwise import haversine_distances
import copy

pd.options.mode.chained_assignment = None  # default='warn'


class SrcRec:
    """I/O for source <--> receiver file

    :param fname: Path to src_rec file
    :type fname: str
    :param src_only: Whether to read only source information, defaults to False
    :type src_only: bool, optional
    """

    def __init__(self, fname: str, src_only=False) -> None:
        """ """
        self.src_only = src_only
        self.src_points = None
        self.rec_points = None
        self.sources = None
        self.receivers = None
        self.fnames = [fname]
        self.log = SetupLog()

    def __repr__(self):
        return f"PyTomoATT SrcRec Object: \n\
                fnames={self.fnames}, \n\
                src_only={self.src_only}, \n\
                number of sources={self.src_points.shape[0]}, \n\
                number of receivers={self.rec_points.shape[0]}"

    @property
    def src_points(self):
        """Return a DataFrame of all sources

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
        else:
            raise TypeError("src_points should be in DataFrame")

    @property
    def rec_points(self):
        """Return a DataFrame of all receivers

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
        sr = cls(fname=fname, **kwargs)
        alldf = pd.read_table(
            fname, sep="\s+|\t", engine="python", header=None, comment="#"
        )

        last_col_src = 12
        # this is a source line if the last column is not NaN
        sr.src_points = alldf[pd.notna(alldf[last_col_src])]
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
            return sr.src_points
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
            if not dist_in_data:
                last_col = 7
            else:
                last_col = 8

            if name_net_and_sta:
                last_col += 1

            # extract the rows if the last_col is not NaN and the 12th column is NaN
            sr.rec_points = alldf[
                (alldf[last_col].notna()) & (alldf[last_col_src].isna())
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

            if name_net_and_sta == False:
                if not dist_in_data:
                    sr.rec_points.columns = [
                        "src_index",
                        "rec_index",
                        "staname",
                        "stla",
                        "stlo",
                        "stel",
                        "phase",
                        "tt",
                        "weight",
                    ]
                else:
                    sr.rec_points.columns = [
                        "src_index",
                        "rec_index",
                        "staname",
                        "stla",
                        "stlo",
                        "stel",
                        "phase",
                        "dist_deg",
                        "tt",
                        "weight",
                    ]
            else:
                if not dist_in_data:
                    sr.rec_points.columns = [
                        "src_index",
                        "rec_index",
                        "netname",
                        "staname",
                        "stla",
                        "stlo",
                        "stel",
                        "phase",
                        "tt",
                        "weight",
                    ]
                else:
                    sr.rec_points.columns = [
                        "src_index",
                        "rec_index",
                        "netname",
                        "staname",
                        "stla",
                        "stlo",
                        "stel",
                        "phase",
                        "dist_deg",
                        "tt",
                        "weight",
                    ]
                # change type of rec_index to int
                sr.rec_points["rec_index"] = sr.rec_points["rec_index"].astype(int)
                # concatenate network and station name with "_"
                sr.rec_points["staname"] = (
                    sr.rec_points["netname"] + "_" + sr.rec_points["staname"]
                )
                # drop network name column
                sr.rec_points.drop("netname", axis=1, inplace=True)
                # define src and rec list
            sr.sources = sr.src_points[
                ["event_id", "evla", "evlo", "evdp", "weight"]
            ]
            sr.receivers = sr.rec_points[
                ["staname", "stla", "stlo", "stel", "weight"]
            ].drop_duplicates()
        return sr

    def write(self, fname="src_rec_file"):
        """Write sources and receivers to ASCII file for TomoATT

        :param fname: Path to the src_rec file, defaults to 'src_rec_file'
        :type fname: str, optional
        """
        with open(fname, "w") as f:
            for idx, src in tqdm.tqdm(
                self.src_points.iterrows(), total=len(self.src_points)
            ):
                time_lst = (
                    src["origin_time"].strftime("%Y_%m_%d_%H_%M_%S.%f").split("_")
                )
                f.write(
                    "{:d} {} {} {} {} {} {} {:.4f} {:.4f} {:.4f} {:.4f} {} {} {:.4f}\n".format(
                        idx,
                        *time_lst,
                        src["evla"],
                        src["evlo"],
                        src["evdp"],
                        src["mag"],
                        src["num_rec"],
                        src["event_id"],
                        src["weight"],
                    )
                )
                if self.src_only:
                    continue
                rec_data = self.rec_points[self.rec_points["src_index"] == idx]
                for _, rec in rec_data.iterrows():
                    f.write(
                        "   {:d} {:d} {} {:6.4f} {:6.4f} {:6.4f} {} {:6.4f} {:6.4f}\n".format(
                            idx,
                            rec["rec_index"],
                            rec["staname"],
                            rec["stla"],
                            rec["stlo"],
                            rec["stel"],
                            rec["phase"],
                            rec["tt"],
                            rec["weight"],
                        )
                    )

    def copy(self):
        """Return a copy of SrcRec object

        :return: Copy of SrcRec object
        :rtype: SrcRec
        """
        return copy.deepcopy(self)

    def reset_index(self):
        """Reset index of source and receivers."""
        # reset src_index to be 0, 1, 2, ... for both src_points and rec_points
        self.rec_points["src_index"] = self.rec_points["src_index"].map(
            dict(zip(self.src_points.index, np.arange(len(self.src_points))))
        )
        self.src_points.index = np.arange(len(self.src_points))
        self.src_points.index.name = "src_index"

        # reset rec_index to be 0, 1, 2, ... for rec_points
        self.rec_points["rec_index"] = self.rec_points.groupby("src_index").cumcount()
        # sr.rec_points['rec_index'] = sr.rec_points['rec_index'].astype(int)

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

        # store fnames
        self.fnames.extend(sr.fnames)

    def remove_rec_by_new_src(self, verbose=True):
        """
        remove rec_points by new src_points
        """
        if verbose:
            self.log.SrcReclog.info(
                "rec_points before removing: {}".format(self.rec_points.shape)
            )
        self.rec_points = self.rec_points[
            self.rec_points["src_index"].isin(self.src_points.index)
        ]
        if verbose:
            self.log.SrcReclog.info(
                "rec_points after removing: {}".format(self.rec_points.shape)
            )

    def remove_src_by_new_rec(self):
        """remove src_points by new receivers"""
        self.src_points = self.src_points[
            self.src_points.index.isin(self.rec_points["src_index"])
        ]

    def update_num_rec(self):
        """
        update num_rec in src_points by current rec_points
        """
        self.src_points["num_rec"] = self.rec_points.groupby("src_index").size()

    def erase_src_with_no_rec(self):
        """
        erase src_points with no rec_points
        """
        print("src_points before removing: ", self.src_points.shape)
        self.src_points = self.src_points[self.src_points["num_rec"] > 0]
        print("src_points after removing: ", self.src_points.shape)

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

        # remove rec_points by new src_points
        self.remove_rec_by_new_src()

        # sort by src_index
        self.src_points.sort_values(by=["src_index"], inplace=True)
        self.rec_points.sort_values(by=["src_index", "rec_index"], inplace=True)

    def select_phase(self, phase_list):
        """
        select interested phase and remove others

        :param phase_list: List of phases for travel times used for inversion
        :type phase_list: list of str
        """
        if not isinstance(phase_list, (list, str)):
            raise TypeError("phase_list should be in list or str")
        self.log.SrcReclog.info(
            "rec_points before selecting: {}".format(self.rec_points.shape)
        )
        self.rec_points = self.rec_points[self.rec_points["phase"].isin(phase_list)]
        self.log.SrcReclog.info(
            "rec_points after selecting: {}".format(self.rec_points.shape)
        )

        # modify num_rec in src_points
        self.src_points["num_rec"] = self.rec_points.groupby("src_index").size()

        # sort by src_index
        self.src_points.sort_values(by=["src_index"], inplace=True)
        self.rec_points.sort_values(by=["src_index", "rec_index"], inplace=True)

    def select_by_datetime(self, time_range):
        """
        select sources and station in a time range

        :param time_range: Time range defined as [start_time, end_time]
        :type time_range: iterable
        """
        # select source within this time range.
        self.log.SrcReclog.info(
            "src_points before selecting: {}".format(self.src_points.shape)
        )
        self.log.SrcReclog.info(
            "rec_points before selecting: {}".format(self.rec_points.shape)
        )
        self.src_points = self.src_points[
            (self.src_points["origin_time"] >= time_range[0])
            & (self.src_points["origin_time"] <= time_range[1])
        ]

        # Remove receivers whose events have been removed
        self.remove_rec_by_new_src(verbose=False)

        self.reset_index()
        self.log.SrcReclog.info(
            "src_points after selecting: {}".format(self.src_points.shape)
        )
        self.log.SrcReclog.info(
            "rec_points after selecting: {}".format(self.rec_points.shape)
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
        self.remove_src_by_new_rec()
        self.update_num_rec()
        self.reset_index()
        self.log.SrcReclog.info(
            "rec_points after removing: {}".format(self.rec_points.shape)
        )

    def select_box_region(self, region):
        """
        Select sources and station in a box region

        :param region: Box region defined as [lon1, lon2, lat1, lat2]
        :type region: iterable
        """
        # select source within this region.
        self.log.SrcReclog.info(
            "src_points before selecting: {}".format(self.src_points.shape)
        )
        self.log.SrcReclog.info(
            "rec_points before selecting: {}".format(self.rec_points.shape)
        )
        self.src_points = self.src_points[
            (self.src_points["evlo"] >= region[0])
            & (self.src_points["evlo"] <= region[1])
            & (self.src_points["evla"] >= region[2])
            & (self.src_points["evla"] <= region[3])
        ]

        # Remove receivers whose events have been removed
        self.remove_rec_by_new_src(verbose=False)

        # Remove rest receivers out of region.
        self.rec_points = self.rec_points[
            (self.rec_points["stlo"] >= region[0])
            & (self.rec_points["stlo"] <= region[1])
            & (self.rec_points["stla"] >= region[2])
            & (self.rec_points["stla"] <= region[3])
        ]

        # Remove empty sources
        self.src_points = self.src_points[
            self.src_points.index.isin(self.rec_points["src_index"])
        ]
        self.update_num_rec()
        self.reset_index()
        self.log.SrcReclog.info(
            "src_points after selecting: {}".format(self.src_points.shape)
        )
        self.log.SrcReclog.info(
            "rec_points after selecting: {}".format(self.rec_points.shape)
        )

    def calc_distance(self):
        """Calculate epicentral distance"""
        self.rec_points["dist"] = 0.0
        rec_group = self.rec_points.groupby("src_index")
        for idx, rec in rec_group:
            dist = DistAZ(
                self.src_points.loc[idx]["evla"],
                self.src_points.loc[idx]["evlo"],
                rec["stla"].values,
                rec["stlo"].values,
            ).delta
            self.rec_points["dist"].loc[rec.index] = dist

    def select_distance(self, dist_min_max, recalc_dist=False):
        """Select stations in a range of distance

        :param dist_min_max: limit of distance in deg, ``[dist_min, dist_max]``
        :type dist_min_max: list or tuple
        """
        self.log.SrcReclog.info(
            "rec_points before selecting: {}".format(self.rec_points.shape)
        )
        # rec_group = self.rec_points.groupby('src_index')
        if ("dist" not in self.rec_points) or recalc_dist:
            self.log.SrcReclog.info("Calculating epicentral distance...")
            self.calc_distance()
        elif not recalc_dist:
            pass
        else:
            self.log.SrcReclog.error(
                "No such field of dist, please set up recalc_dist to True"
            )
        # for _, rec in rec_group:
        mask = (self.rec_points["dist"] < dist_min_max[0]) | (
            self.rec_points["dist"] > dist_min_max[1]
        )
        drop_idx = self.rec_points[mask].index
        self.rec_points = self.rec_points.drop(index=drop_idx)
        self.remove_src_by_new_rec()
        self.update_num_rec()
        self.reset_index()
        self.log.SrcReclog.info(
            "rec_points after selecting: {}".format(self.rec_points.shape)
        )

    def select_by_num_rec(self, num_rec: int):
        """select sources with recievers greater and equal than a number
        :param num_rec: threshold of minimum receiver number
        :type num_rec: int
        """
        self.update_num_rec()
        self.log.SrcReclog.info(
            "src_points before selecting: {}".format(self.src_points.shape)
        )
        self.log.SrcReclog.info(
            "rec_points before selecting: {}".format(self.rec_points.shape)
        )
        self.src_points = self.src_points[(self.src_points["num_rec"] >= num_rec)]
        self.remove_rec_by_new_src(False)
        self.log.SrcReclog.info(
            "src_points after selecting: {}".format(self.src_points.shape)
        )
        self.log.SrcReclog.info(
            "rec_points after selecting: {}".format(self.rec_points.shape)
        )

    def select_one_event_in_each_subgrid(self, d_deg: float, d_km: float):
        """select one event in each subgrid

        :param d_deg: grid size along lat and lon in degree
        :type d_deg: float
        :param d_km: grid size along depth axis in km
        :type d_km: float
        """

        self.log.SrcReclog.info(
            "src_points before selecting: {}".format(self.src_points.shape)
        )
        self.log.SrcReclog.info("processing... (this may take a few minutes)")

        # store index of src_points as 'src_index'
        self.src_points["src_index"] = self.src_points.index

        # add 'lat_group' and 'lon_group' to src_points by module d_deg
        self.src_points["lat_group"] = self.src_points["evla"].apply(
            lambda x: int(x / d_deg)
        )
        self.src_points["lon_group"] = self.src_points["evlo"].apply(
            lambda x: int(x / d_deg)
        )

        # add 'dep_group' to src_points by module d_km
        self.src_points["dep_group"] = self.src_points["evdp"].apply(
            lambda x: int(x / d_km)
        )

        # sort src_points by 'lat_group' and 'lon_group' and 'dep_group'
        self.src_points = self.src_points.sort_values(
            by=["lat_group", "lon_group", "dep_group"]
        )

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
            "src_points after selecting: {}".format(self.src_points.shape)
        )

        # remove rec_points by new src_points
        self.remove_rec_by_new_src()

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

    def _calc_weights(self, lat, lon, scale):
        points = pd.concat([lon, lat], axis=1)
        points_rad = points * (np.pi / 180)
        dist = haversine_distances(points_rad) * 6371.0 / 111.19
        dist_ref = scale * np.mean(dist)
        om = np.exp(-((dist / dist_ref) ** 2)) * points.shape[0]
        return 1 / np.mean(om, axis=0)

    def geo_weighting(self, scale=0.5, rec_weight=False):
        """Calculating geographical weights for sources

        :param scale: Scale of reference distance parameter See equation 22 in Ruan et al., (2019),
                      The reference distance is given by ``scale``* dis_average, defaults to 0.5
        :type scale: float, optional
        """

        self.src_points["weight"] = self._calc_weights(
            self.src_points["evla"], self.src_points["evlo"], scale
        )
        if rec_weight:
            weights = self._calc_weights(
                self.receivers['stla'],
                self.receivers['stlo'],
                scale
            )
            # apply weights to rec_points
            for staname, weight in zip(self.receivers['staname'], weights):
                self.rec_points.loc[self.rec_points['staname'] == staname, 'weight'] = weight
    #
    # This function is comment out temprarly because it includes verified bug and not modified.
    #
    # def merge_adjacent_stations(self, d_deg:float, d_km:float):
    #    """
    #    merge adjacent stations as one station
    #    d_deg : float
    #        grid size in degree
    #    d_km : float
    #        grid size in km
    #    """

    #    # count the number of events per station
    #    self.count_events_per_station()

    #    # number of unique stations before merging
    #    print('number of unique stations before merging: ', self.rec_points['staname'].nunique())

    #    # create 'lat_group', 'lon_group' and 'dep_group' columns from 'stla', 'stlo' and 'stel'
    #    def create_groups(row, column, d):
    #        return int(row[column]/d)

    #    self.rec_points['lat_group'] = self.rec_points.apply(lambda x: create_groups(x, 'stla', d_deg), axis=1)
    #    self.rec_points['lon_group'] = self.rec_points.apply(lambda x: create_groups(x, 'stlo', d_deg), axis=1)
    #    self.rec_points['dep_group'] = self.rec_points.apply(lambda x: create_groups(x, 'stel', d_km*1000), axis=1)

    #    # sort src_points by 'lat_group' and 'lon_group' and 'dep_group'
    #    self.rec_points = self.rec_points.sort_values(by=['lat_group', 'lon_group', 'dep_group', 'num_events'], ascending=[True, True, True, False])

    #    # find all events in the same lat_group and lon_group and dep_group
    #    # and copy the 'staname' 'stlo' 'stla' 'stel' to all rows within the same group from the row where 'count' is the largest
    #    self.rec_points['staname'] = self.rec_points.groupby(['lat_group', 'lon_group', 'dep_group'])['staname'].transform(lambda x: x.iloc[0])
    #    self.rec_points['stlo'] = self.rec_points.groupby(['lat_group', 'lon_group', 'dep_group'])['stlo'].transform(lambda x: x.iloc[0])
    #    self.rec_points['stla'] = self.rec_points.groupby(['lat_group', 'lon_group', 'dep_group'])['stla'].transform(lambda x: x.iloc[0])
    #    self.rec_points['stel'] = self.rec_points.groupby(['lat_group', 'lon_group', 'dep_group'])['stel'].transform(lambda x: x.iloc[0])

    #    # drop 'lat_group' and 'lon_group' and 'dep_group'
    #    self.rec_points = self.rec_points.drop(columns=['lat_group', 'lon_group', 'dep_group'])

    #    # sort
    #    self.rec_points = self.rec_points.sort_values(by=['src_index','rec_index'])

    #    # update the num_events
    #    self.count_events_per_station()

    #    # number of unique stations after merging
    #    print('number of unique stations after merging: ', self.rec_points['staname'].nunique())

    #
    # This function is comment out temprarly because it includes verified bug and not modified.
    #
    # def merge_duplicated_station(self):
    #    """
    #    merge duplicated stations as one station
    #    duplicated stations are defined as stations with the same staname
    #    """

    #    # number of unique stations before merging
    #    print('number of unique stations before merging: ', self.rec_points['staname'].nunique())

    #    # sort rec_points by 'src_index' then 'staname'
    #    self.rec_points = self.rec_points.sort_values(by=['src_index', 'staname'])

    #    # find all duplicated stations in each src_index and drop except the first one
    #    self.rec_points = self.rec_points.drop_duplicates(subset=['src_index', 'staname'], keep='first')

    #    # sort rec_points by 'src_index' then 'rec_index'
    #    self.rec_points = self.rec_points.sort_values(by=['src_index', 'rec_index'])

    #    # update the num_events
    #    self.count_events_per_station()

    #    # number of unique stations after merging
    #    print('number of unique stations after merging: ', self.rec_points['staname'].nunique())

    def add_noise(self, range_in_sec=0.1, mean_in_sec=0.0, shape="gaussian"):
        """Add random noise on travel time

        :param mean_in_sec: Mean of the noise in sec, defaults to 0.0
        :type mean_in_sec: float, optional
        :param range_in_sec: Maximun noise in sec, defaults to 0.1
        :type range_in_sec: float, optional
        :param shape: shape of the noise distribution probability
        :type shape: str. options: gaussian or uniform
        """
        if shape == "uniform":
            noise = (
                np.random.uniform(
                    low=-range_in_sec, high=range_in_sec, size=self.rec_points.shape[0]
                )
                + mean_in_sec
            )
        elif shape == "gaussian":
            noise = np.random.normal(
                loc=mean_in_sec, scale=range_in_sec, size=self.rec_points.shape[0]
            )
        self.rec_points["tt"] += noise

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

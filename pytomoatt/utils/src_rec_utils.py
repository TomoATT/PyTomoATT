
def define_rec_cols(dist_in_data, name_net_and_sta):
    if not dist_in_data:
        last_col = 7
    else:
        last_col = 8

    if name_net_and_sta:
        last_col += 1

    if name_net_and_sta == False:
        if not dist_in_data:
            columns = [
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
            columns = [
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
            columns = [
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
            columns = [
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
    return columns, last_col

def get_rec_points_types(dist):
    common_type = {
        "src_index": int,
        "rec_index": int,
        "staname": str,
        "stla": float,
        "stlo": float,
        "stel": float,
        "phase": str,
        "tt": float,
        "weight": float,
    }
    if dist:
        common_type["dist_deg"] = float
    return common_type

def setup_rec_points_dd(type='cs'):
    if type == 'cs':       
        columns = [
            "src_index",
            "rec_index1",
            "staname1",
            "stla1",
            "stlo1",
            "stel1",
            "rec_index2",
            "staname2",
            "stla2",
            "stlo2",
            "stel2",
            "phase",
            "tt",
            "weight"
        ]
        data_type = {
            "src_index": int,
            "rec_index1": int,
            "staname1": str,
            "stla1": float,
            "stlo1": float,
            "stel1": float,
            "rec_index2": int,
            "staname2": str,
            "stla2": float,
            "stlo2": float,
            "stel2": float,
            "phase": str,
            "tt": float,
            "weight": float
        }
    elif type == 'cr':
        columns = [
            "src_index",
            "rec_index",
            "staname",
            "stla",
            "stlo",
            "stel",
            "src_index2",
            "event_id2",
            "evla2",
            "evlo2",
            "evdp2",
            "phase",
            "tt",
            "weight"
        ]
        data_type = {
            "src_index": int,
            "rec_index": int,
            "staname": str,
            "stla": float,
            "stlo": float,
            "stel": float,
            "src_index2": int,
            "event_id2": str,
            "evla2": float,
            "evlo2": float,
            "evdp2": float,
            "phase": str,
            "tt": float,
            "weight": float
        }
    else:
        raise ValueError('type should be either "cs" or "cr"')
    return columns, data_type
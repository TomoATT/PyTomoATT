import urllib3
import io
import tqdm


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


def update_position(sr):
    sr.src_points = sr.src_points.merge(
        sr.sources[['event_id', 'evlo', 'evla']],
        on='event_id',
        how='left',
        suffixes=('', '_new')
    )
    sr.src_points['evlo'] = sr.src_points['evlo_new']
    sr.src_points['evla'] = sr.src_points['evla_new']
    sr.src_points.drop(columns=['evlo_new', 'evla_new'], inplace=True)

    sr.rec_points = sr.rec_points.merge(
        sr.receivers[['staname', 'stlo', 'stla']],
        on='staname',
        how='left',
        suffixes=('', '_new')
    )
    sr.rec_points['stlo'] = sr.rec_points['stlo_new']
    sr.rec_points['stla'] = sr.rec_points['stla_new']
    sr.rec_points.drop(columns=['stlo_new', 'stla_new'], inplace=True)

    if not sr.rec_points_cs.empty:
        sr.rec_points_cs = sr.rec_points_cs.merge(
            sr.receivers[['staname', 'stlo', 'stla']],
            left_on='staname1',
            right_on='staname',
            how='left',
        )
        sr.rec_points_cs['stlo1'] = sr.rec_points_cs['stlo']
        sr.rec_points_cs['stla1'] = sr.rec_points_cs['stla']
        sr.rec_points_cs.drop(columns=['stlo', 'stla', 'staname'], inplace=True)

        sr.rec_points_cs = sr.rec_points_cs.merge(
            sr.receivers[['staname', 'stlo', 'stla']],
            left_on='staname2',
            right_on='staname',
            how='left',
        )
        sr.rec_points_cs['stlo2'] = sr.rec_points_cs['stlo']
        sr.rec_points_cs['stla2'] = sr.rec_points_cs['stla']
        sr.rec_points_cs.drop(columns=['stlo', 'stla', 'staname'], inplace=True)

    if not sr.rec_points_cr.empty:
        sr.rec_points_cr = sr.rec_points_cr.merge(
            sr.receivers[['staname', 'stlo', 'stla']],
            on='staname',
            how='left',
            suffixes=('', '_new')
        )
        sr.rec_points_cr['stlo'] = sr.rec_points_cr['stlo_new']
        sr.rec_points_cr['stla'] = sr.rec_points_cr['stla_new']
        sr.rec_points_cr.drop(columns=['stlo_new', 'stla_new'], inplace=True)

        sr.rec_points_cr = sr.rec_points_cr.merge(
            sr.sources[['event_id', 'evlo', 'evla']],
            left_on='event_id2',
            right_on='event_id',
            how='left',
        )
        sr.rec_points_cr['evlo2'] = sr.rec_points_cr['evlo']
        sr.rec_points_cr['evla2'] = sr.rec_points_cr['evla']
        sr.rec_points_cr.drop(columns=['evlo', 'evla', 'event_id'], inplace=True)


def download_src_rec_file(url):
    http = urllib3.PoolManager()
    response = http.request('GET', url, preload_content=False)
    if response.status == 200:
        data = io.StringIO()
        total_size = int(response.headers.get('Content-Length', 0))
        with tqdm.tqdm(total=total_size, unit='KB', unit_scale=True, desc='Downloading') as pbar:
            while True:
                chunk = response.read(1024)
                if not chunk:
                    break
                data.write(chunk.decode('utf-8'))
                pbar.update(len(chunk))
        data.seek(0)  
        response.release_conn()
        return data
    else:
        response.release_conn()
        return None

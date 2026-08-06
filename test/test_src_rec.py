import unittest
from pytomoatt.src_rec import SrcRec
from pytomoatt.utils.src_rec_utils import (
    define_rec_cols,
    get_rec_points_types,
    setup_rec_points_dd,
    update_position,
    download_src_rec_file,
    linear_regression,
)
from os.path import dirname, join
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from sklearn.metrics.pairwise import haversine_distances


class TestSrcRec(unittest.TestCase):
    fname: str = join(dirname(dirname(__file__)), 'examples', 'src_rec_file_eg')
    fname1: str = join(dirname(__file__), 'test_srcrec_a.dat')
    duplicate_index_fname: str = join(
        dirname(__file__), 'src_rec_duplicate_index.dat'
    )

    def test_read_missing_local_file(self):
        missing_file = join(dirname(__file__), "missing_src_rec.dat")

        with self.assertRaisesRegex(
            FileNotFoundError, "src_rec file not found"
        ):
            SrcRec.read(missing_file)

    def test_conflicting_receiver_error_lists_station_locations(self):
        sr = SrcRec("unused")
        sr.src_points = pd.DataFrame({
            "event_id": ["EVENT_0"],
            "evla": [1.0],
            "evlo": [2.0],
            "evdp": [3.0],
        })
        sr.rec_points = pd.DataFrame({
            "staname": [
                "STA_CONFLICT",
                "STA_CONFLICT",
                "STA_CONFLICT",
                "STA_OK",
            ],
            "stla": [10.0, 10.0, 10.1, 20.0],
            "stlo": [30.0, 30.0, 30.0, 40.0],
            "stel": [0.0, 0.0, 0.0, 0.0],
        })

        with self.assertRaises(ValueError) as raised:
            sr.update_unique_src_rec()

        message = str(raised.exception)
        self.assertIn("1 receiver(s)", message)
        self.assertIn("staname", message)
        self.assertIn("stla", message)
        self.assertIn("stlo", message)
        self.assertIn("stel", message)
        self.assertIn("count", message)
        self.assertIn("STA_CONFLICT", message)
        self.assertIn("10.0", message)
        self.assertIn("10.1", message)
        self.assertIn("30.0", message)
        self.assertIn("2", message)

    def test_remove_conflicting_receivers_removes_all_record_types(self):
        sr = SrcRec("unused")
        sr.src_points = pd.DataFrame({
            "event_id": ["EVENT_0", "EVENT_1"],
            "evla": [1.0, 2.0],
            "evlo": [3.0, 4.0],
            "evdp": [5.0, 6.0],
            "num_rec": [5, 1],
        })
        sr.rec_points = pd.DataFrame({
            "src_index": [0, 1, 0],
            "staname": ["STA_CONFLICT", "STA_CONFLICT", "STA_OK"],
            "stla": [10.0, 10.1, 20.0],
            "stlo": [30.0, 30.0, 40.0],
            "stel": [0.0, 0.0, 0.0],
        })
        sr.rec_points_cs = pd.DataFrame({
            "src_index": [0, 0],
            "staname1": ["STA_CONFLICT", "STA_CS_1"],
            "stla1": [10.0, 21.0],
            "stlo1": [30.0, 41.0],
            "stel1": [0.0, 0.0],
            "staname2": ["STA_CS_0", "STA_CS_2"],
            "stla2": [22.0, 23.0],
            "stlo2": [42.0, 43.0],
            "stel2": [0.0, 0.0],
        })
        sr.rec_points_cr = pd.DataFrame({
            "src_index": [0, 0],
            "event_id2": ["EVENT_1", "EVENT_1"],
            "evla2": [2.0, 2.0],
            "evlo2": [4.0, 4.0],
            "evdp2": [6.0, 6.0],
            "staname": ["STA_CONFLICT", "STA_CR_OK"],
            "stla": [10.0, 24.0],
            "stlo": [30.0, 44.0],
            "stel": [0.0, 0.0],
        })

        sr.update_unique_src_rec(conflicting_receiver_action="remove")

        self.assertEqual(sr.rec_points["staname"].tolist(), ["STA_OK"])
        self.assertEqual(sr.rec_points_cs["staname1"].tolist(), ["STA_CS_1"])
        self.assertEqual(sr.rec_points_cr["staname"].tolist(), ["STA_CR_OK"])
        self.assertNotIn("STA_CONFLICT", sr.receivers["staname"].tolist())
        self.assertEqual(sr.src_points["num_rec"].tolist(), [3, 0])

    def test_rename_conflicting_receivers_updates_all_record_types(self):
        sr = SrcRec("unused")
        sr.src_points = pd.DataFrame({
            "event_id": ["EVENT_0", "EVENT_1"],
            "evla": [1.0, 2.0],
            "evlo": [3.0, 4.0],
            "evdp": [5.0, 6.0],
            "num_rec": [3, 1],
        })
        sr.rec_points = pd.DataFrame({
            "src_index": [0, 1],
            "staname": ["STA_CONFLICT", "STA_CONFLICT"],
            "stla": [10.0, 10.1],
            "stlo": [30.0, 30.0],
            "stel": [0.0, 0.0],
        })
        sr.rec_points_cs = pd.DataFrame({
            "src_index": [0],
            "staname1": ["STA_CONFLICT"],
            "stla1": [10.1],
            "stlo1": [30.0],
            "stel1": [0.0],
            "staname2": ["STA_CONFLICT"],
            "stla2": [10.0],
            "stlo2": [30.0],
            "stel2": [0.0],
        })
        sr.rec_points_cr = pd.DataFrame({
            "src_index": [0],
            "event_id2": ["EVENT_1"],
            "evla2": [2.0],
            "evlo2": [4.0],
            "evdp2": [6.0],
            "staname": ["STA_CONFLICT"],
            "stla": [10.0],
            "stlo": [30.0],
            "stel": [0.0],
        })

        sr.update_unique_src_rec(conflicting_receiver_action="rename")

        self.assertEqual(
            sr.rec_points["staname"].tolist(),
            ["STA_CONFLICT_A", "STA_CONFLICT_B"],
        )
        self.assertEqual(sr.rec_points_cs["staname1"].iloc[0], "STA_CONFLICT_B")
        self.assertEqual(sr.rec_points_cs["staname2"].iloc[0], "STA_CONFLICT_A")
        self.assertEqual(sr.rec_points_cr["staname"].iloc[0], "STA_CONFLICT_A")
        self.assertEqual(
            sr.receivers["staname"].tolist(),
            ["STA_CONFLICT_A", "STA_CONFLICT_B"],
        )

    def test_max_count_conflicting_receivers_removes_other_locations(self):
        sr = SrcRec("unused")
        sr.src_points = pd.DataFrame({
            "event_id": ["EVENT_0", "EVENT_1"],
            "evla": [1.0, 2.0],
            "evlo": [3.0, 4.0],
            "evdp": [5.0, 6.0],
            "num_rec": [3, 1],
        })
        sr.rec_points = pd.DataFrame({
            "src_index": [0, 1],
            "staname": ["STA_CONFLICT", "STA_CONFLICT"],
            "stla": [10.0, 20.0],
            "stlo": [30.0, 40.0],
            "stel": [100.0, 200.0],
        })
        sr.rec_points_cs = pd.DataFrame({
            "src_index": [0],
            "staname1": ["STA_CONFLICT"],
            "stla1": [10.0],
            "stlo1": [30.0],
            "stel1": [100.0],
            "staname2": ["STA_CS"],
            "stla2": [0.0],
            "stlo2": [0.0],
            "stel2": [0.0],
        })
        sr.rec_points_cr = pd.DataFrame({
            "src_index": [0],
            "event_id2": ["EVENT_1"],
            "evla2": [2.0],
            "evlo2": [4.0],
            "evdp2": [6.0],
            "staname": ["STA_CONFLICT"],
            "stla": [10.0],
            "stlo": [30.0],
            "stel": [100.0],
        })

        sr.update_unique_src_rec(conflicting_receiver_action="max_count")

        expected = np.array([10.0, 30.0, 100.0])
        self.assertEqual(sr.rec_points["staname"].tolist(), ["STA_CONFLICT"])
        self.assertTrue(
            np.allclose(
                sr.rec_points.loc[0, ["stla", "stlo", "stel"]]
                .to_numpy(dtype=float),
                expected,
            )
        )
        self.assertEqual(sr.rec_points_cs.shape[0], 1)
        self.assertTrue(
            np.allclose(
                sr.rec_points_cs.loc[0, ["stla1", "stlo1", "stel1"]]
                .to_numpy(dtype=float),
                expected,
            )
        )
        self.assertEqual(sr.rec_points_cr.shape[0], 1)
        self.assertTrue(
            np.allclose(
                sr.rec_points_cr.loc[0, ["stla", "stlo", "stel"]]
                .to_numpy(dtype=float),
                expected,
            )
        )
        self.assertEqual(
            sr.receivers[sr.receivers["staname"] == "STA_CONFLICT"].shape[0],
            1,
        )

    def test_normalize_phase_labels_corrects_pg_pn_case(self):
        sr = SrcRec("unused")
        sr.rec_points = pd.DataFrame({"phase": ["PG", "PN", "P"]})
        sr.rec_points_cs = pd.DataFrame({"phase": ["PG,cs", "PN,cs"]})
        sr.rec_points_cr = pd.DataFrame({"phase": ["PG,cr", "PN,cr"]})

        sr._normalize_phase_labels()

        self.assertEqual(sr.rec_points["phase"].tolist(), ["Pg", "Pn", "P"])
        self.assertEqual(sr.rec_points_cs["phase"].tolist(), ["Pg,cs", "Pn,cs"])
        self.assertEqual(sr.rec_points_cr["phase"].tolist(), ["Pg,cr", "Pn,cr"])

    def test_read_accepts_conflicting_receiver_action(self):
        src_rec_data = """\
0 2020 1 1 0 0 0.0 1.0 2.0 3.0 1.0 1 EVENT_0 1.0
0 0 STA_CONFLICT 10.0 30.0 0.0 P 1.0 1.0
1 2020 1 2 0 0 0.0 4.0 5.0 6.0 1.0 1 EVENT_1 1.0
1 0 STA_CONFLICT 10.1 30.0 0.0 P 1.0 1.0
"""
        with TemporaryDirectory() as directory:
            input_file = join(directory, "conflicting_src_rec.dat")
            with open(input_file, "w") as output:
                output.write(src_rec_data)

            with self.assertRaisesRegex(ValueError, "STA_CONFLICT"):
                SrcRec.read(input_file)

            sr = SrcRec.read(
                input_file,
                conflicting_receiver_action="remove",
            )
            renamed_sr = SrcRec.read(
                input_file,
                conflicting_receiver_action="rename",
            )

        self.assertTrue(sr.rec_points.empty)
        self.assertTrue(sr.receivers.empty)
        self.assertEqual(sr.src_points["num_rec"].tolist(), [0, 0])
        self.assertEqual(
            renamed_sr.rec_points["staname"].tolist(),
            ["STA_CONFLICT_A", "STA_CONFLICT_B"],
        )

    def test_remove_specified_recs_also_removes_double_difference_records(self):
        sr = SrcRec.read(self.duplicate_index_fname)

        sr.remove_specified_recs(["STA1"])

        self.assertNotIn("STA1", sr.rec_points["staname"].tolist())
        self.assertTrue(sr.rec_points_cs.empty)
        self.assertTrue(sr.rec_points_cr.empty)

    def test_read_reindexes_duplicate_file_src_indices(self):
        sr = SrcRec.read(self.duplicate_index_fname)

        self.assertEqual(sr.src_points.index.tolist(), [0, 1])
        self.assertEqual(sr.src_points['event_id'].tolist(), ['EVT_A', 'EVT_B'])
        self.assertEqual(sr.rec_points['src_index'].tolist(), [0, 1])
        self.assertEqual(sr.rec_points_cs['src_index'].tolist(), [0])
        self.assertEqual(sr.rec_points_cr['src_index'].tolist(), [0])
        self.assertEqual(sr.rec_points_cr['src_index2'].tolist(), [1])

        with TemporaryDirectory() as directory:
            output_file = join(directory, 'src_rec.dat')
            sr.write(output_file)
            reread = SrcRec.read(output_file)

        self.assertEqual(reread.src_points.index.tolist(), [0, 1])
        self.assertEqual(reread.rec_points['src_index'].tolist(), [0, 1])
        self.assertEqual(reread.rec_points_cr['src_index2'].tolist(), [1])

    def test_subcase_01(self):
        sr = SrcRec.read(self.fname)
        sr.select_by_distance([0, 1])
        self.assertEqual(sr.rec_points.shape[0], 19378)

    def test_subcase_02(self):
        sr = SrcRec.read(self.fname)
        sr.select_by_box_region([-1, 0, -1, 0])
        self.assertEqual(sr.rec_points.shape[0], 671)
        self.assertEqual(sr.src_points.shape[0], 85)

    def test_subcase_03(self):
        sr = SrcRec.read(self.fname)
        sr.geo_weighting(0.1, "rec")
        sr.write('test_src_rec_eg')

    def test_subcase_04(self):
        sr = SrcRec.read(self.fname)
        sr.add_noise()
        sr.add_noise(shape='uniform')

    def test_subcase_05(self):
        sr = SrcRec.read(self.fname)
        sr.select_by_depth([0, 10])
        self.assertEqual(sr.src_points.shape[0], 1413)
        self.assertEqual(sr.rec_points.shape[0], 11226)
    
    def test_subcase_06(self):
        sr = SrcRec.read(self.fname)
        sr1 = SrcRec.read(self.fname1, dist_in_data=True)
        sr.append(sr1)
        self.assertEqual(sr.src_points.shape[0], 2506)
        self.assertEqual(sr._count_records(), 19926)

    def test_subcase_07(self):
        sr = SrcRec.read(self.fname)
        sr.select_by_azi_gap(120)
        self.assertEqual(sr.src_points.shape[0], 329)
        self.assertEqual(sr.rec_points.shape[0], 3815)

    def test_subcase_08(self):
        sr = SrcRec.read(self.fname)
        sr.select_by_phase('P')

    def test_subcase_09(self):
        sr = SrcRec.read(self.fname)
        sr.generate_double_difference('cs', max_azi_gap=15, max_dist_gap=1.4)
        sr.generate_double_difference('cr', max_azi_gap=15, max_dist_gap=0.01)

    def test_generate_double_difference_dis_type(self):
        sr_cs = SrcRec("unused")
        sr_cs.src_points = pd.DataFrame(
            {
                "evla": [0.0],
                "evlo": [0.0],
                "evdp": [1.0],
                "event_id": ["E0"],
                "weight": [1.0],
            },
            index=pd.Index([0], name="src_index"),
        )
        sr_cs.rec_points = pd.DataFrame(
            {
                "src_index": [0, 0],
                "rec_index": [0, 1],
                "staname": ["STA_E", "STA_W"],
                "stla": [0.0, 0.0],
                "stlo": [1.0, -1.0],
                "stel": [0.0, 0.0],
                "phase": ["P", "P"],
                "tt": [1.0, 2.0],
                "weight": [1.0, 1.0],
            }
        )
        sr_cs.calc_distaz()

        sr_cs._generate_cs(200.0, 0.1, dis_type="dis_dif")
        self.assertEqual(sr_cs.rec_points_cs.shape[0], 1)
        sr_cs._generate_cs(200.0, 0.1, dis_type="dis_pair")
        self.assertEqual(sr_cs.rec_points_cs.shape[0], 0)

        sr_cr = SrcRec("unused")
        sr_cr.src_points = pd.DataFrame(
            {
                "evla": [1.0, -1.0],
                "evlo": [0.0, 0.0],
                "evdp": [1.0, 1.0],
                "event_id": ["E0", "E1"],
                "weight": [1.0, 1.0],
            },
            index=pd.Index([0, 1], name="src_index"),
        )
        sr_cr.rec_points = pd.DataFrame(
            {
                "src_index": [0, 1],
                "rec_index": [0, 0],
                "staname": ["STA", "STA"],
                "stla": [0.0, 0.0],
                "stlo": [0.0, 0.0],
                "stel": [0.0, 0.0],
                "phase": ["P", "P"],
                "tt": [1.0, 2.0],
                "weight": [1.0, 1.0],
            }
        )
        sr_cr.receivers = pd.DataFrame(
            {
                "staname": ["STA"],
                "stla": [0.0],
                "stlo": [0.0],
                "stel": [0.0],
            }
        )
        sr_cr.calc_distaz()

        sr_cr._generate_cr(200.0, 0.1, dis_type="dis_dif")
        self.assertEqual(sr_cr.rec_points_cr.shape[0], 1)
        sr_cr._generate_cr(200.0, 0.1, dis_type="dis_pair")
        self.assertEqual(sr_cr.rec_points_cr.shape[0], 0)

        with self.assertRaisesRegex(ValueError, "dis_type"):
            sr_cr.generate_double_difference(dis_type="bad")

    def test_subcase_10(self):
        sr = SrcRec.read(self.fname)
        sr.box_weighting(0.4, 10, obj='both')

    def test_box_weighting_receiver_does_not_require_depth_size(self):
        sr = SrcRec('unused')

        with (
            patch.object(sr, '_box_weighting_ev') as weight_sources,
            patch.object(sr, '_box_weighting_st') as weight_receivers,
        ):
            sr.box_weighting(d_deg=0.4, obj='rec')

        weight_sources.assert_not_called()
        weight_receivers.assert_called_once_with(0.4, 'average')

    def test_box_weighting_requires_depth_size_for_sources(self):
        sr = SrcRec('unused')

        with self.assertRaisesRegex(ValueError, 'd_km'):
            sr.box_weighting(d_deg=0.4, obj='src')
        with self.assertRaisesRegex(ValueError, 'd_km'):
            sr.box_weighting(d_deg=0.4, obj='both')

    def test_box_weighting_receiver_uses_horizontal_cells_only(self):
        sr = SrcRec('unused')
        sr.receivers = pd.DataFrame({
            'staname': ['STA0', 'STA1', 'STA2'],
            'stla': [0.1, 0.2, 2.1],
            'stlo': [0.1, 0.2, 2.1],
            'stel': [0.0, 5000.0, 100.0],
        })
        sr.rec_points = pd.DataFrame({
            'staname': ['STA0', 'STA1', 'STA2'],
            'weight': [1.0, 1.0, 1.0],
        })
        sr.src_points = pd.DataFrame({
            'event_id': ['E0'],
            'weight': [0.5],
        })
        sr.rec_points_cs = pd.DataFrame({
            'staname1': ['STA0'],
            'staname2': ['STA1'],
            'weight': [1.0],
        })
        sr.rec_points_cr = pd.DataFrame({
            'staname': ['STA2'],
            'event_id2': ['E0'],
            'weight': [1.0],
        })

        sr.box_weighting(d_deg=1.0, obj='rec')

        expected_dense_weight = 1.0 / np.sqrt(2.0)
        receiver_weights = sr.receivers.set_index('staname')['weight']
        self.assertAlmostEqual(receiver_weights['STA0'], expected_dense_weight)
        self.assertAlmostEqual(receiver_weights['STA1'], expected_dense_weight)
        self.assertAlmostEqual(receiver_weights['STA2'], 1.0)
        self.assertTrue(np.allclose(
            sr.rec_points['weight'],
            sr.rec_points['staname'].map(receiver_weights),
        ))
        self.assertAlmostEqual(
            sr.rec_points_cs.iloc[0]['weight'], expected_dense_weight
        )
        self.assertAlmostEqual(sr.rec_points_cr.iloc[0]['weight'], 0.75)

    def test_select_by_linear_regression(self):
        sr = SrcRec('unused')
        distance = np.concatenate((np.arange(21, dtype=float), [10.0, 0.0]))
        distance_km = distance * 100.0
        distance_3d_km = distance_km + 10.0
        travel_time = 2.0 * distance + 5.0
        travel_time[10] += 100.0
        sr.rec_points = pd.DataFrame({
            'src_index': [0] * 21 + [1, 1],
            'staname': [f'STA{i:02d}' for i in range(21)] + ['STA10', 'STA00'],
            'dist_deg': distance,
            'dist_km': distance_km,
            'dist_3d_km': distance_3d_km,
            'tt': travel_time,
            'phase': 'P',
        })
        sr.rec_points_cs = pd.DataFrame({
            'src_index': [0, 0],
            'staname1': ['STA10', 'STA00'],
            'staname2': ['STA00', 'STA01'],
            'phase': ['P,cs', 'P,cs'],
        })
        sr.rec_points_cr = pd.DataFrame({
            'src_index': [0, 0],
            'src_index2': [1, 1],
            'staname': ['STA10', 'STA00'],
            'phase': ['P,cr', 'P,cr'],
        })

        with patch.object(sr, 'update') as update:
            regression_params = sr.select_by_linear_regression(
                std_multiplier=3.0,
                distance='dist_3d_km',
            )

        expected_slope, expected_intercept, expected_std = linear_regression(
            distance_3d_km, travel_time
        )
        self.assertIn('P', regression_params)
        slope, intercept, residual_std = regression_params['P']
        self.assertAlmostEqual(slope, expected_slope)
        self.assertAlmostEqual(intercept, expected_intercept)
        self.assertAlmostEqual(residual_std, expected_std)
        self.assertEqual(sr.rec_points.shape[0], 22)
        self.assertNotIn(10, sr.rec_points.index)
        self.assertEqual(sr.rec_points_cs.shape[0], 1)
        self.assertEqual(sr.rec_points_cs.iloc[0]['staname1'], 'STA00')
        self.assertEqual(sr.rec_points_cr.shape[0], 1)
        self.assertEqual(sr.rec_points_cr.iloc[0]['staname'], 'STA00')
        update.assert_called_once_with()

    def test_linear_regression_method(self):
        sr = SrcRec('unused')
        distance = np.arange(5, dtype=float)
        p_travel_time = np.array([4.0, 7.0, 10.0, 13.0, 17.0])
        sr.rec_points = pd.DataFrame({
            'dist_deg': np.concatenate((distance, distance)),
            'dist_km': np.concatenate((distance * 10.0, distance * 10.0)),
            'dist_3d_km': np.concatenate(
                (distance * 10.0 + 2.0, distance * 10.0 + 2.0)
            ),
            'tt': np.concatenate((p_travel_time, 5.0 * distance + 2.0)),
            'phase': ['P'] * 5 + ['S'] * 5,
        })
        original = sr.rec_points.copy(deep=True)

        result = sr.linear_regression(phase='P')
        expected = linear_regression(distance * 10.0 + 2.0, p_travel_time)

        for actual_value, expected_value in zip(result, expected):
            self.assertAlmostEqual(actual_value, expected_value)

        result_deg = sr.linear_regression(phase='P', distance='dist_deg')
        expected_deg = linear_regression(distance, p_travel_time)
        for actual_value, expected_value in zip(result_deg, expected_deg):
            self.assertAlmostEqual(actual_value, expected_value)
        pd.testing.assert_frame_equal(sr.rec_points, original)

    def test_linear_regression_method_calculates_distance(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'tt': [2.0, 5.0, 8.0],
            'phase': ['P', 'P', 'P'],
        })

        def add_distance():
            sr.rec_points['dist_3d_km'] = [0.0, 1.0, 2.0]

        with patch.object(sr, 'calc_distaz', side_effect=add_distance) as calc:
            slope, intercept, residual_std = sr.linear_regression()

        self.assertAlmostEqual(slope, 3.0)
        self.assertAlmostEqual(intercept, 2.0)
        self.assertAlmostEqual(residual_std, 0.0)
        calc.assert_called_once_with()

    def test_calc_distaz_calculates_source_receiver_distance(self):
        sr = SrcRec('unused')
        sr.src_points = pd.DataFrame({
            'evla': [10.0],
            'evlo': [20.0],
            'evdp': [10.0],
        })
        sr.rec_points = pd.DataFrame({
            'src_index': [0],
            'stla': [10.0],
            'stlo': [20.0],
            'stel': [1000.0],
        })

        sr.calc_distaz()

        self.assertAlmostEqual(sr.rec_points.loc[0, 'dist_deg'], 0.0)
        self.assertAlmostEqual(sr.rec_points.loc[0, 'dist_km'], 0.0)
        self.assertAlmostEqual(sr.rec_points.loc[0, 'dist_3d_km'], 11.0)

    def test_calc_weights_uses_lat_lon_order_and_normalizes(self):
        sr = SrcRec('unused')
        latitude = np.array([0.0, 60.0, 10.0])
        longitude = np.array([0.0, 10.0, 170.0])
        scale = 0.5

        weights = sr._calc_weights(latitude, longitude, scale)

        points_rad = np.deg2rad(np.column_stack((latitude, longitude)))
        distances = haversine_distances(points_rad)
        reference_distance = scale * distances.mean()
        expected = np.reciprocal(
            np.exp(-((distances / reference_distance) ** 2)).sum(axis=0)
        )
        expected /= expected.max()
        self.assertTrue(np.allclose(weights, expected))
        self.assertAlmostEqual(weights.max(), 1.0)

    def test_geo_weighting_maps_and_normalizes_all_weights(self):
        sr = SrcRec('unused')
        sr.src_points = pd.DataFrame({
            'event_id': ['E0', 'E1', 'E2'],
            'evla': [0.0, 1.0, 3.0],
            'evlo': [0.0, 2.0, 1.0],
            'evdp': [5.0, 10.0, 15.0],
            'weight': [1.0, 1.0, 1.0],
        })
        sr.rec_points = pd.DataFrame({
            'src_index': [0, 1, 2],
            'staname': ['STA0', 'STA1', 'STA2'],
            'stla': [0.0, 2.0, 1.0],
            'stlo': [0.0, 1.0, 4.0],
            'stel': [0.0, 100.0, 200.0],
            'phase': ['P', 'P', 'P'],
            'tt': [1.0, 2.0, 3.0],
            'weight': [1.0, 1.0, 1.0],
        })
        sr.rec_points_cs = pd.DataFrame({
            'src_index': [0],
            'staname1': ['STA0'],
            'stla1': [0.0],
            'stlo1': [0.0],
            'stel1': [0.0],
            'staname2': ['STA1'],
            'stla2': [2.0],
            'stlo2': [1.0],
            'stel2': [100.0],
            'phase': ['P,cs'],
            'weight': [1.0],
        })
        sr.rec_points_cr = pd.DataFrame({
            'src_index': [0],
            'src_index2': [1],
            'event_id2': ['E1'],
            'evla2': [1.0],
            'evlo2': [2.0],
            'evdp2': [10.0],
            'staname': ['STA0'],
            'stla': [0.0],
            'stlo': [0.0],
            'stel': [0.0],
            'phase': ['P,cr'],
            'weight': [1.0],
        })
        sr.update_unique_src_rec()

        sr.geo_weighting(scale=0.5, obj='both', dd_weight='multiply')

        source_weights = sr.src_points.set_index('event_id')['weight']
        receiver_weights = sr.receivers.set_index('staname')['weight']
        self.assertTrue(np.allclose(
            sr.sources['weight'], sr.sources['event_id'].map(source_weights)
        ))
        self.assertTrue(np.allclose(
            sr.rec_points['weight'],
            sr.rec_points['staname'].map(receiver_weights),
        ))
        expected_cs_weight = (
            receiver_weights['STA0'] * receiver_weights['STA1']
        )
        expected_cr_weight = receiver_weights['STA0'] * source_weights['E1']
        self.assertAlmostEqual(
            sr.rec_points_cs.iloc[0]['weight'], expected_cs_weight
        )
        self.assertAlmostEqual(
            sr.rec_points_cr.iloc[0]['weight'], expected_cr_weight
        )

        all_weights = np.concatenate((
            sr.src_points['weight'].to_numpy(),
            sr.sources['weight'].to_numpy(),
            sr.receivers['weight'].to_numpy(),
            sr.rec_points['weight'].to_numpy(),
            sr.rec_points_cs['weight'].to_numpy(),
            sr.rec_points_cr['weight'].to_numpy(),
        ))
        self.assertAlmostEqual(all_weights.max(), 1.0)
        self.assertTrue(np.all(all_weights <= 1.0))

    def test_geo_weighting_deduplicates_receiver_names(self):
        sr = SrcRec('unused')
        sr.receivers = pd.DataFrame({
            'staname': ['STA0', 'STA0', 'STA1'],
            'stla': [0.0, 0.1, 1.0],
            'stlo': [0.0, 0.1, 2.0],
            'stel': [0.0, 10.0, 20.0],
        })
        sr.rec_points = pd.DataFrame({
            'staname': ['STA0', 'STA1'],
            'weight': [1.0, 1.0],
        })

        sr.geo_weighting(scale=0.5, obj='rec')

        self.assertEqual(sr.receivers['staname'].tolist(), ['STA0', 'STA1'])
        receiver_weights = sr.receivers.set_index('staname')['weight']
        self.assertTrue(np.allclose(
            sr.rec_points['weight'],
            sr.rec_points['staname'].map(receiver_weights),
        ))
        self.assertAlmostEqual(sr.receivers['weight'].max(), 1.0)

    def test_select_by_distance_filters_double_differences(self):
        sr = SrcRec('unused')
        sr.src_points = pd.DataFrame({
            'evla': [0.0, 0.0],
            'evlo': [0.0, 0.5],
        }, index=[0, 1])
        sr.rec_points = pd.DataFrame({
            'src_index': [0, 0],
            'staname': ['ABS_IN', 'ABS_OUT'],
            'dist_deg': [0.5, 2.0],
            'dist_km': [55.6, 222.4],
            'phase': ['P', 'P'],
        })
        sr.rec_points_cs = pd.DataFrame({
            'src_index': [0],
            'staname1': ['STA1'],
            'stla1': [0.0],
            'stlo1': [0.5],
            'staname2': ['STA2'],
            'stla2': [0.0],
            'stlo2': [2.0],
            'phase': ['P,cs'],
        })
        sr.rec_points_cr = pd.DataFrame({
            'src_index': [0, 0],
            'src_index2': [1, 1],
            'event_id2': ['EVT1', 'EVT1'],
            'staname': ['STA1', 'STA2'],
            'stla': [0.0, 0.0],
            'stlo': [0.5, 0.5],
            'evla2': [0.0, 0.0],
            'evlo2': [0.2, 2.0],
            'phase': ['P,cr', 'P,cr'],
        })

        with patch.object(sr, 'update') as update:
            sr.select_by_distance(
                [0.0, 112.0],
                distance='dist_km',
            )

        self.assertEqual(sr.rec_points['staname'].tolist(), ['ABS_IN'])
        self.assertTrue(sr.rec_points_cs.empty)
        self.assertEqual(sr.rec_points_cr['staname'].tolist(), ['STA1'])
        update.assert_called_once_with()

    def test_select_by_distance_rejects_invalid_distance(self):
        sr = SrcRec('unused')

        with self.assertRaisesRegex(ValueError, 'distance'):
            sr.select_by_distance([0.0, 1.0], distance='dist_3d_km')

    def test_select_by_constant_velocity(self):
        sr = SrcRec('unused')
        distance_deg = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
        distance_km = np.deg2rad(distance_deg) * 6371.0
        reference_tt = distance_km / 10.0
        sr.rec_points = pd.DataFrame({
            'src_index': [0, 0, 0, 0, 1],
            'staname': ['STA0', 'STA1', 'STA2', 'STA3', 'STA0'],
            'dist_deg': distance_deg,
            'dist_km': distance_km,
            'tt': reference_tt + np.array([-0.9, 0.0, 1.9, 2.1, 0.0]),
            'phase': ['P'] * 5,
        })
        sr.rec_points_cs = pd.DataFrame({
            'src_index': [0, 0],
            'staname1': ['STA0', 'STA0'],
            'staname2': ['STA1', 'STA3'],
            'phase': ['P,cs', 'P,cs'],
        })
        sr.rec_points_cr = pd.DataFrame({
            'src_index': [0, 0],
            'src_index2': [1, 1],
            'staname': ['STA0', 'STA3'],
            'phase': ['P,cr', 'P,cr'],
        })

        with patch.object(sr, 'update') as update:
            sr.select_by_constant_velocity(
                velocity=10.0,
                tt_res_range=(-1.0, 2.0),
            )

        self.assertEqual(sr.rec_points.shape[0], 4)
        self.assertNotIn('STA3', sr.rec_points['staname'].values)
        self.assertEqual(sr.rec_points_cs.shape[0], 1)
        self.assertEqual(sr.rec_points_cr.shape[0], 1)
        update.assert_called_once_with()

    def test_select_by_constant_velocity_validates_parameters(self):
        sr = SrcRec('unused')

        with self.assertRaisesRegex(ValueError, 'velocity'):
            sr.select_by_constant_velocity(0.0, (-1.0, 1.0))
        with self.assertRaisesRegex(ValueError, 'tt_res_range'):
            sr.select_by_constant_velocity(1.0, (2.0, 1.0))
        for distance in ('dist_deg', 'invalid', None, ['dist_km']):
            with self.subTest(distance=distance):
                with self.assertRaisesRegex(
                    ValueError,
                    "distance must be either 'dist_km' or 'dist_3d_km'",
                ):
                    sr.select_by_constant_velocity(
                        1.0,
                        (-1.0, 1.0),
                        distance=distance,
                    )

    def test_select_by_constant_velocity_accepts_3d_distance(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'src_index': [0],
            'staname': ['STA0'],
            'dist_3d_km': [10.0],
            'tt': [2.0],
            'phase': ['P'],
        })

        with (
            patch.object(sr, 'calc_distaz') as calc_distaz,
            patch.object(sr, 'update') as update,
        ):
            sr.select_by_constant_velocity(
                velocity=10.0,
                tt_res_range=(0.0, 2.0),
                distance='dist_3d_km',
            )

        self.assertEqual(sr.rec_points.shape[0], 1)
        calc_distaz.assert_not_called()
        update.assert_called_once_with()

    def test_plot(self):
        sr = SrcRec.read(self.fname)
        original_columns = sr.src_points.columns.copy()

        figure = sr.plot()
        figure.canvas.draw()

        self.assertIsNotNone(figure)
        self.assertTrue(np.allclose(figure.get_size_inches(), (8.0, 8.0)))
        self.assertTrue(original_columns.equals(sr.src_points.columns))
        map_position = figure.axes[0].get_position()
        latitude_depth_position = figure.axes[1].get_position()
        longitude_depth_position = figure.axes[2].get_position()
        self.assertAlmostEqual(map_position.y0, latitude_depth_position.y0)
        self.assertAlmostEqual(map_position.y1, latitude_depth_position.y1)
        self.assertAlmostEqual(map_position.x0, longitude_depth_position.x0)
        self.assertAlmostEqual(map_position.x1, longitude_depth_position.x1)
        right_gap = (
            latitude_depth_position.x0 - map_position.x1
        ) * figure.get_figwidth()
        lower_gap = (
            map_position.y0 - longitude_depth_position.y1
        ) * figure.get_figheight()
        self.assertAlmostEqual(right_gap, lower_gap)
        right_depth_length = (
            latitude_depth_position.width * figure.get_figwidth()
        )
        lower_depth_length = (
            longitude_depth_position.height * figure.get_figheight()
        )
        self.assertAlmostEqual(right_depth_length, lower_depth_length)
        colorbar_position = figure.axes[3].get_position()
        self.assertAlmostEqual(
            colorbar_position.x0, latitude_depth_position.x0
        )
        self.assertAlmostEqual(
            colorbar_position.x1, latitude_depth_position.x1
        )
        self.assertAlmostEqual(
            colorbar_position.y0, longitude_depth_position.y0
        )
        self.assertAlmostEqual(
            colorbar_position.y1, longitude_depth_position.y1
        )
        self.assertEqual(
            figure.axes[1].yaxis.get_ticks_position(), "right"
        )
        self.assertEqual(
            figure.axes[1].yaxis.get_label_position(), "right"
        )
        self.assertEqual(figure.axes[0].get_aspect(), 1.0)
        self.assertEqual(figure.axes[0].get_adjustable(), "box")
        map_xlim = figure.axes[0].get_xlim()
        map_ylim = figure.axes[0].get_ylim()
        longitude_scale = map_position.width / (map_xlim[1] - map_xlim[0])
        latitude_scale = map_position.height / (map_ylim[1] - map_ylim[0])
        self.assertAlmostEqual(longitude_scale, latitude_scale)
        plt.close(figure)

    def test_write_sources_and_receivers_format_weights(self):
        sr = SrcRec("unused")
        sr.sources = pd.DataFrame({
            "event_id": ["EVENT_0", "EVENT_1"],
            "evla": [1.0, 2.0],
            "evlo": [3.0, 4.0],
            "evdp": [5.0, 6.0],
            "weight": [1.0 / 3.0, 1.0],
        })
        sr.receivers = pd.DataFrame({
            "staname": ["STA0", "STA1"],
            "stla": [1.0, 2.0],
            "stlo": [3.0, 4.0],
            "stel": [5.0, 6.0],
            "weight": [2.0 / 3.0, 1.0],
        })

        with TemporaryDirectory() as output_directory:
            source_file = join(output_directory, "sources.txt")
            receiver_file = join(output_directory, "receivers.txt")
            sr.write_sources(source_file)
            sr.write_receivers(receiver_file)

            with open(source_file) as output:
                source_weights = [
                    line.split()[-1] for line in output if line.strip()
                ]
            with open(receiver_file) as output:
                receiver_weights = [
                    line.split()[-1] for line in output if line.strip()
                ]

        self.assertEqual(source_weights, ["0.3333", "1.0000"])
        self.assertEqual(receiver_weights, ["0.6667", "1.0000"])
        self.assertEqual(sr.sources.loc[0, "weight"], 1.0 / 3.0)
        self.assertEqual(sr.receivers.loc[0, "weight"], 2.0 / 3.0)

    def test_write_roundtrip_preserves_microsecond_and_weight_precision(self):
        """Write→read roundtrip preserves microsecond origin time and 4-decimal weight precision."""
        sr = SrcRec("unused")

        origin_time_with_us = pd.Timestamp("2013-10-06 09:20:53.123456")
        src_weight = 1.0 / 3.0  # 0.3333...
        rec_weight = 2.0 / 3.0  # 0.6666...

        src_df = pd.DataFrame({
            "origin_time": [origin_time_with_us],
            "evla": [-1.7673],
            "evlo": [-0.6619],
            "evdp": [9.55],
            "mag": [2.66],
            "num_rec": [1],
            "event_id": ["EVT_0001"],
            "weight": [src_weight],
        })
        src_df.index = pd.Index([0], name="src_index")
        sr.src_points = src_df

        rec_df = pd.DataFrame({
            "src_index": [0],
            "rec_index": [0],
            "staname": ["STA0"],
            "stla": [-1.0351],
            "stlo": [-0.3383],
            "stel": [219.0],
            "phase": ["P"],
            "tt": [14.786],
            "weight": [rec_weight],
        })
        sr.rec_points = rec_df

        with TemporaryDirectory() as directory:
            output_file = join(directory, "src_rec.dat")
            sr.write(output_file)
            reread = SrcRec.read(output_file)

        # Verify microsecond precision is preserved in origin time
        self.assertEqual(
            reread.src_points["origin_time"].iloc[0],
            origin_time_with_us,
        )

        # Verify source weight is preserved to 4 decimal places
        self.assertAlmostEqual(
            reread.src_points["weight"].iloc[0],
            src_weight,
            places=4,
        )

        # Verify receiver weight is preserved to 4 decimal places
        self.assertAlmostEqual(
            reread.rec_points["weight"].iloc[0],
            rec_weight,
            places=4,
        )


        sr = SrcRec.read(self.fname, src_only=True)

        figure = sr.plot(color_by="weight")

        self.assertIsNotNone(figure)
        plt.close(figure)

    def test_plot_uses_shared_norm_for_constant_weights(self):
        sr = SrcRec.read(self.fname, src_only=True)
        sr.src_points['weight'] = 1.0

        figure = sr.plot(color_by='weight')
        figure.canvas.draw()
        source_collections = [axis.collections[0] for axis in figure.axes[:3]]

        self.assertIs(
            source_collections[0].norm, source_collections[1].norm
        )
        self.assertIs(
            source_collections[0].norm, source_collections[2].norm
        )
        reference_colors = source_collections[0].get_facecolors()
        for collection in source_collections[1:]:
            self.assertTrue(np.allclose(
                collection.get_facecolors(), reference_colors
            ))
        plt.close(figure)

    def test_plot_colors_receivers_by_weight(self):
        sr = SrcRec.read(self.fname)
        sr.geo_weighting(obj='both')

        figure = sr.plot(color_by='weight')
        figure.canvas.draw()
        source_collection = figure.axes[0].collections[0]
        receiver_collection = figure.axes[0].collections[1]
        source_index = sr.src_points['weight'].to_numpy().argmax()
        receiver_index = sr.receivers['weight'].to_numpy().argmax()

        self.assertIsNot(source_collection.norm, receiver_collection.norm)
        self.assertIs(source_collection.cmap, receiver_collection.cmap)
        self.assertTrue(np.allclose(
            source_collection.get_facecolors()[source_index],
            receiver_collection.get_facecolors()[receiver_index],
        ))
        self.assertEqual(
            source_collection.norm(sr.src_points['weight'].max()),
            receiver_collection.norm(sr.receivers['weight'].max()),
        )
        colorbar_labels = {
            axis.get_xlabel() for axis in figure.axes[3].child_axes
        }
        self.assertIn('Source weight', colorbar_labels)
        self.assertIn('Receiver weight', colorbar_labels)
        plt.close(figure)

    def test_plot_rejects_invalid_color_by(self):
        sr = SrcRec.read(self.fname, src_only=True)

        with self.assertRaisesRegex(ValueError, "color_by"):
            sr.plot(color_by="magnitude")

    def test_plot_accepts_matplotlib_scatter_options(self):
        sr = SrcRec.read(self.fname, src_only=True)

        figure = sr.plot(cmap="jet", s=12, alpha=0.5, marker="x")
        source_collection = figure.axes[0].collections[0]

        self.assertEqual(source_collection.get_cmap().name, "jet")
        self.assertEqual(source_collection.get_sizes()[0], 12)
        self.assertEqual(source_collection.get_alpha(), 0.5)
        plt.close(figure)

    def test_plot_travel_time_returns_editable_figure(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_3d_km': [0.0, 1.0, np.nan],
            'tt': [1.0, 3.0, 5.0],
        })

        figure = sr.plot_travel_time(color='red', s=12, alpha=0.5)
        axis = figure.axes[0]
        collection = axis.collections[0]
        line = axis.plot([0.0, 1.0], [1.0, 3.0])[0]

        self.assertEqual(collection.get_offsets().shape[0], 2)
        self.assertTrue(np.allclose(figure.get_size_inches(), (6.0, 4.5)))
        self.assertTrue(
            np.allclose(collection.get_facecolors()[0], to_rgba('red', 0.5))
        )
        self.assertIn(line, axis.lines)
        plt.close(figure)

    def test_plot_travel_time_calculates_missing_distance(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({'tt': [1.0, 2.0]})

        def add_distance():
            sr.rec_points['dist_3d_km'] = [0.0, 1.0]

        with patch.object(sr, 'calc_distaz', side_effect=add_distance) as calc:
            figure = sr.plot_travel_time()

        calc.assert_called_once_with()
        plt.close(figure)

    def test_plot_travel_time_uses_kilometres(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_deg': [0.0, 1.0],
            'dist_km': [0.0, 111.19],
            'tt': [1.0, 3.0],
        })

        figure = sr.plot_travel_time(distance='dist_km')
        axis = figure.axes[0]
        offsets = axis.collections[0].get_offsets()

        self.assertTrue(np.allclose(offsets[:, 0], [0.0, 111.19]))
        self.assertEqual(axis.get_xlabel(), 'Epicentral distance (km)')
        plt.close(figure)

    def test_plot_travel_time_defaults_to_3d_distance(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_deg': [1.0, 2.0],
            'dist_3d_km': [120.0, 230.0],
            'tt': [10.0, 20.0],
        })

        figure = sr.plot_travel_time()
        axis = figure.axes[0]
        offsets = axis.collections[0].get_offsets()

        self.assertTrue(np.allclose(offsets[:, 0], [120.0, 230.0]))
        self.assertEqual(
            axis.get_xlabel(), '3-D source-receiver distance (km)'
        )
        plt.close(figure)

    def test_plot_travel_time_rejects_invalid_distance(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_3d_km': [0.0, 1.0],
            'tt': [1.0, 3.0],
        })

        with self.assertRaisesRegex(ValueError, 'distance'):
            sr.plot_travel_time(distance='miles')

    def test_plot_travel_time_uses_existing_figure(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_3d_km': [0.0, 1.0],
            'tt': [1.0, 3.0],
        })
        existing_figure, axis = plt.subplots()
        existing_line = axis.plot([0.0, 1.0], [0.0, 2.0])[0]

        returned_figure = sr.plot_travel_time(
            fig=existing_figure,
            color='red',
        )

        self.assertIs(returned_figure, existing_figure)
        self.assertIn(existing_line, axis.lines)
        self.assertEqual(len(axis.collections), 1)
        plt.close(existing_figure)

    def test_plot_travel_time_inherits_y_limits(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_3d_km': [0.0, 1.0],
            'tt': [1.0, 30.0],
        })
        existing_figure, axis = plt.subplots()
        axis.set_ylim(5.0, 20.0)

        returned_figure = sr.plot_travel_time(
            fig=existing_figure,
            ylim='inherit',
        )

        self.assertIs(returned_figure, existing_figure)
        self.assertEqual(axis.get_ylim(), (5.0, 20.0))
        plt.close(existing_figure)

    def test_plot_travel_time_uses_matplotlib_color_cycle(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_3d_km': [0.0, 1.0],
            'tt': [1.0, 2.0],
        })

        figure = sr.plot_travel_time()
        first_color = figure.axes[0].collections[-1].get_facecolors()[0]
        sr.plot_travel_time(fig=figure, ylim='inherit')
        second_color = figure.axes[0].collections[-1].get_facecolors()[0]

        self.assertFalse(np.allclose(first_color, second_color))
        plt.close(figure)

    def test_plot_travel_time_y_limits(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_3d_km': [0.0, 1.0, 2.0, 3.0, 4.0],
            'tt': [1.0, 2.0, 1000.0, 4.0, 5.0],
        })

        adaptive_figure = sr.plot_travel_time()
        auto_figure = sr.plot_travel_time(ylim='auto')
        explicit_figure = sr.plot_travel_time(ylim=(0.0, 10.0))

        adaptive_limits = adaptive_figure.axes[0].get_ylim()
        self.assertLess(adaptive_limits[0], 1.0)
        self.assertGreater(adaptive_limits[1], 5.0)
        self.assertLess(adaptive_limits[1], 1000.0)
        self.assertGreater(auto_figure.axes[0].get_ylim()[1], 1000.0)
        self.assertEqual(explicit_figure.axes[0].get_ylim(), (0.0, 10.0))
        plt.close(adaptive_figure)
        plt.close(auto_figure)
        plt.close(explicit_figure)


class TestSrcRecUtils(unittest.TestCase):
    def test_linear_regression(self):
        slope, intercept, std = linear_regression(
            [0.0, 1.0, 2.0], [1.0, 3.0, 5.0]
        )
        self.assertAlmostEqual(slope, 2.0)
        self.assertAlmostEqual(intercept, 1.0)
        self.assertAlmostEqual(std, 0.0)

    def test_define_rec_cols(self):
        # Case 1: dist_in_data=False, name_net_and_sta=False
        cols, last_col = define_rec_cols(False, False)
        self.assertNotIn("dist_deg", cols)
        self.assertNotIn("netname", cols)
        self.assertEqual(last_col, 7)

        # Case 2: dist_in_data=True, name_net_and_sta=False
        cols, last_col = define_rec_cols(True, False)
        self.assertIn("dist_deg", cols)
        self.assertNotIn("netname", cols)
        self.assertEqual(last_col, 8)

        # Case 3: dist_in_data=False, name_net_and_sta=True
        cols, last_col = define_rec_cols(False, True)
        self.assertNotIn("dist_deg", cols)
        self.assertIn("netname", cols)
        self.assertEqual(last_col, 8)

        # Case 4: dist_in_data=True, name_net_and_sta=True
        cols, last_col = define_rec_cols(True, True)
        self.assertIn("dist_deg", cols)
        self.assertIn("netname", cols)
        self.assertEqual(last_col, 9)

    def test_get_rec_points_types(self):
        types = get_rec_points_types(False)
        self.assertNotIn("dist_deg", types)
        
        types = get_rec_points_types(True)
        self.assertIn("dist_deg", types)
        self.assertEqual(types["dist_deg"], float)

    def test_setup_rec_points_dd(self):
        cols, types = setup_rec_points_dd('cs')
        self.assertIn("rec_index1", cols)
        self.assertIn("rec_index2", cols)
        
        cols, types = setup_rec_points_dd('cr')
        self.assertIn("src_index", cols)
        self.assertIn("src_index2", cols)
        
        with self.assertRaises(ValueError):
            setup_rec_points_dd('invalid')

    def test_update_position(self):
        # Mock SrcRec object
        sr = MagicMock()
        
        # Setup DataFrames
        sr.sources = pd.DataFrame({
            'event_id': [1, 2],
            'evlo': [10.0, 20.0],
            'evla': [30.0, 40.0]
        })
        
        sr.receivers = pd.DataFrame({
            'staname': ['STA1', 'STA2'],
            'stlo': [100.0, 110.0],
            'stla': [50.0, 60.0]
        })
        
        sr.src_points = pd.DataFrame({
            'event_id': [1, 2],
            'evlo': [0.0, 0.0], # Old values
            'evla': [0.0, 0.0]  # Old values
        })
        
        sr.rec_points = pd.DataFrame({
            'staname': ['STA1', 'STA2'],
            'stlo': [0.0, 0.0], # Old values
            'stla': [0.0, 0.0]  # Old values
        })
        
        sr.rec_points_cs = pd.DataFrame({
            'staname1': ['STA1'],
            'staname2': ['STA2'],
            'stlo1': [0.0], 'stla1': [0.0],
            'stlo2': [0.0], 'stla2': [0.0]
        })
        
        sr.rec_points_cr = pd.DataFrame({
            'staname': ['STA1'],
            'event_id2': [2],
            'stlo': [0.0], 'stla': [0.0],
            'evlo2': [0.0], 'evla2': [0.0]
        })
        
        update_position(sr)
        
        # Check src_points updated
        self.assertEqual(sr.src_points.iloc[0]['evlo'], 10.0)
        self.assertEqual(sr.src_points.iloc[0]['evla'], 30.0)
        
        # Check rec_points updated
        self.assertEqual(sr.rec_points.iloc[0]['stlo'], 100.0)
        self.assertEqual(sr.rec_points.iloc[0]['stla'], 50.0)
        
        # Check rec_points_cs updated
        self.assertEqual(sr.rec_points_cs.iloc[0]['stlo1'], 100.0)
        self.assertEqual(sr.rec_points_cs.iloc[0]['stlo2'], 110.0)
        
        # Check rec_points_cr updated
        self.assertEqual(sr.rec_points_cr.iloc[0]['stlo'], 100.0)
        self.assertEqual(sr.rec_points_cr.iloc[0]['evlo2'], 20.0)

    def test_download_src_rec_file(self):
        with patch('urllib3.PoolManager') as mock_pool:
            mock_http = mock_pool.return_value
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.headers = {'Content-Length': '10'}
            mock_response.read.side_effect = [b'test data', b'']
            mock_http.request.return_value = mock_response
            
            data = download_src_rec_file('http://example.com/file')
            self.assertEqual(data.getvalue(), 'test data')
            
            # Test failure case
            mock_response.status = 404
            data = download_src_rec_file('http://example.com/file')
            self.assertIsNone(data)


if __name__ == '__main__':
    unittest.main()

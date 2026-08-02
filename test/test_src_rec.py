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

    def test_subcase_10(self):
        sr = SrcRec.read(self.fname)
        sr.box_weighting(0.4, 10, obj='both')

    def test_select_by_linear_regression(self):
        sr = SrcRec('unused')
        distance = np.concatenate((np.arange(21, dtype=float), [10.0, 0.0]))
        travel_time = 2.0 * distance + 5.0
        travel_time[10] += 100.0
        sr.rec_points = pd.DataFrame({
            'src_index': [0] * 21 + [1, 1],
            'staname': [f'STA{i:02d}' for i in range(21)] + ['STA10', 'STA00'],
            'dist_deg': distance,
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
                std_multiplier=3.0
            )

        expected_slope, expected_intercept = np.polyfit(
            distance, travel_time, deg=1
        )
        self.assertIn('P', regression_params)
        slope, intercept = regression_params['P']
        self.assertAlmostEqual(slope, expected_slope)
        self.assertAlmostEqual(intercept, expected_intercept)
        self.assertEqual(sr.rec_points.shape[0], 22)
        self.assertNotIn(10, sr.rec_points.index)
        self.assertEqual(sr.rec_points_cs.shape[0], 1)
        self.assertEqual(sr.rec_points_cs.iloc[0]['staname1'], 'STA00')
        self.assertEqual(sr.rec_points_cr.shape[0], 1)
        self.assertEqual(sr.rec_points_cr.iloc[0]['staname'], 'STA00')
        update.assert_called_once_with()

    def test_select_by_constant_velocity(self):
        sr = SrcRec('unused')
        distance = np.array([0.0, 1.0, 2.0, 3.0, 0.0])
        reference_tt = np.deg2rad(distance) * 6371.0 / 10.0
        sr.rec_points = pd.DataFrame({
            'src_index': [0, 0, 0, 0, 1],
            'staname': ['STA0', 'STA1', 'STA2', 'STA3', 'STA0'],
            'dist_deg': distance,
            'tt': reference_tt + np.array([-1.0, 0.0, 2.0, 2.1, 0.0]),
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
        plt.close(figure)

    def test_plot_source_only(self):
        sr = SrcRec.read(self.fname, src_only=True)

        figure = sr.plot(color_by="weight")

        self.assertIsNotNone(figure)
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
            'dist_deg': [0.0, 1.0, np.nan],
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
            sr.rec_points['dist_deg'] = [0.0, 1.0]

        with patch.object(sr, 'calc_distaz', side_effect=add_distance) as calc:
            figure = sr.plot_travel_time()

        calc.assert_called_once_with()
        plt.close(figure)

    def test_plot_travel_time_uses_existing_figure(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_deg': [0.0, 1.0],
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
            'dist_deg': [0.0, 1.0],
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

    def test_plot_travel_time_y_limits(self):
        sr = SrcRec('unused')
        sr.rec_points = pd.DataFrame({
            'dist_deg': [0.0, 1.0, 2.0, 3.0, 4.0],
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

import unittest
from pytomoatt.src_rec import SrcRec
from pytomoatt.utils.src_rec_utils import (
    define_rec_cols,
    get_rec_points_types,
    setup_rec_points_dd,
    update_position,
    download_src_rec_file
)
from os.path import dirname, join
from unittest.mock import MagicMock, patch
import pandas as pd
import io


class TestSrcRec(unittest.TestCase):
    fname: str = join(dirname(dirname(__file__)), 'examples', 'src_rec_file_eg')
    fname1: str = join(dirname(__file__), 'test_srcrec_a.dat')

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


class TestSrcRecUtils(unittest.TestCase):
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



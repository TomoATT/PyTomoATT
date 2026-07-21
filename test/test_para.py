import unittest
import os
import shutil
from ruamel.yaml import YAML
from pytomoatt.para import ATTPara
import numpy as np

yaml = YAML()

class TestATTPara(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_para_output'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        self.cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        self.yaml_content = {
            'domain': {
                'min_max_dep': [0, 100],
                'min_max_lat': [30, 40],
                'min_max_lon': [100, 110],
                'n_rtp': [11, 11, 11]
            },
            'test_section': {
                'key1': 'value1'
            }
        }
        self.fname = 'test_params.yml'
        with open(self.fname, 'w') as f:
            yaml.dump(self.yaml_content, f)

    def tearDown(self):
        os.chdir(self.cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init(self):
        para = ATTPara(self.fname)
        self.assertEqual(para.input_params['domain']['n_rtp'], [11, 11, 11])
        self.assertEqual(para.input_params['test_section']['key1'], 'value1')

    def test_init_axis(self):
        para = ATTPara(self.fname)
        dep, lat, lon, dd, dt, dp = para.init_axis()
        
        # Check shapes based on n_rtp [11, 11, 11]
        self.assertEqual(len(dep), 11)
        self.assertEqual(len(lat), 11)
        self.assertEqual(len(lon), 11)
        
        # Check values
        # Note: init_axis flips the depth array
        self.assertEqual(dep[0], 100)
        self.assertEqual(dep[-1], 0)
        self.assertEqual(lat[0], 30)
        self.assertEqual(lat[-1], 40)
        self.assertEqual(lon[0], 100)
        self.assertEqual(lon[-1], 110)

    def test_update_param(self):
        para = ATTPara(self.fname)
        
        # Test updating existing nested key
        para.update_param('domain.n_rtp', '20,20,20')
        self.assertEqual(para.input_params['domain']['n_rtp'], [20, 20, 20])
        
        # Test updating existing simple key
        para.update_param('test_section.key1', 'new_value')
        self.assertEqual(para.input_params['test_section']['key1'], 'new_value')
        
        # Test adding new key
        para.update_param('new_section.new_key', '123.45')
        self.assertEqual(para.input_params['new_section']['new_key'], 123.45)

    def test_write(self):
        para = ATTPara(self.fname)
        para.update_param('domain.n_rtp', '30,30,30')
        
        out_fname = 'out_params.yml'
        para.write(out_fname)
        
        self.assertTrue(os.path.exists(out_fname))
        
        # Verify content
        with open(out_fname, 'r') as f:
            new_params = yaml.load(f)
        self.assertEqual(new_params['domain']['n_rtp'], [30, 30, 30])

    def test_write_overwrite(self):
        para = ATTPara(self.fname)
        para.update_param('domain.n_rtp', '40,40,40')
        para.write() # Should overwrite self.fname
        
        with open(self.fname, 'r') as f:
            new_params = yaml.load(f)
        self.assertEqual(new_params['domain']['n_rtp'], [40, 40, 40])

if __name__ == '__main__':
    unittest.main()

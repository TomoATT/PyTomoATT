import unittest
from unittest.mock import patch, MagicMock
import os
import shutil
import sys
from ruamel.yaml import YAML
import h5py
import numpy as np
from pytomoatt.script import PTA

yaml = YAML()


class TestScripts(unittest.TestCase):
    def setUp(self):
        self.test_dir = 'test_script_output'
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        self.cwd = os.getcwd()
        os.chdir(self.test_dir)

    def tearDown(self):
        os.chdir(self.cwd)
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_init_pjt(self):
        project_name = 'my_project'
        with patch('builtins.input', return_value='y'):
            with patch.object(sys, 'argv', ['pta', 'init_pjt', project_name]):
                PTA()
        self.assertTrue(os.path.exists(project_name))
        self.assertTrue(os.path.exists(os.path.join(project_name, 'input_params.yml')))

    def test_setpar(self):
        project_name = 'my_project'
        with patch.object(sys, 'argv', ['pta', 'init_pjt', project_name]):
            PTA()
        
        os.chdir(project_name)
        with patch.object(sys, 'argv', ['pta', 'setpar', 'input_params.yml', 'domain.n_rtp', '10,10,10']):
            PTA()
        
        with open('input_params.yml', 'r') as f:
            params = yaml.load(f)
        self.assertEqual(params['domain']['n_rtp'], [10, 10, 10])

    def test_model2vtk(self):
        # Create a dummy model file
        project_name = 'my_project'
        with patch.object(sys, 'argv', ['pta', 'init_pjt', project_name]):
            PTA()
        os.chdir(project_name)
        
        # Create a dummy h5 model
        # The shape must match n_rtp in input_params.yml which is [10, 50, 50] by default
        with h5py.File('model.h5', 'w') as f:
            f.create_dataset('vel', data=np.zeros((10, 50, 50)))
            f.create_dataset('xi', data=np.zeros((10, 50, 50)))
            f.create_dataset('eta', data=np.zeros((10, 50, 50)))
            f.create_dataset('zeta', data=np.zeros((10, 50, 50)))

        with patch.dict(sys.modules, {'pyvista': MagicMock()}):
            with patch.object(sys, 'argv', ['pta', 'model2vtk', 'input_params.yml', '-i', 'model.h5', '-o', 'model.vtk']):
                PTA()
        
        # self.assertTrue(os.path.exists('model.vtk')) # Mocked pyvista won't write file

if __name__ == '__main__':
    unittest.main()

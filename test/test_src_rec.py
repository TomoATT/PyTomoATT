import pytest
from pytomoatt.src_rec import SrcRec
from os.path import dirname, join


class TestSrcRec:
    fname: str = join(dirname(dirname(__file__)), 'examples', 'src_rec_file_eg')

    def test_subcase_01(self):
        sr = SrcRec.read(self.fname)
        sr.select_distance([0, 1])
        assert sr.rec_points.shape[0] == 19378

    def test_subcase_02(self):
        sr = SrcRec.read(self.fname)
        sr.select_box_region([-1, 0, -1, 0])
        assert sr.rec_points.shape[0] == 671 and sr.src_points.shape[0] == 85

    def test_subcase_03(self):
        sr = SrcRec.read(self.fname)
        sr.geo_weighting(rec_weight=True)
        sr.write('test_src_rec_eg')

    def test_subcase_04(self):
        sr = SrcRec.read(self.fname)
        sr.add_noise()
        sr.add_noise(shape='uniform')

if __name__ == '__main__':
    tsr = TestSrcRec()
    tsr.test_subcase_03()


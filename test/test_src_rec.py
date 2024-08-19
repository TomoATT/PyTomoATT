import pytest
from pytomoatt.src_rec import SrcRec
from os.path import dirname, join


class TestSrcRec:
    fname: str = join(dirname(dirname(__file__)), 'examples', 'src_rec_file_eg')
    fname1: str = join(dirname(__file__), 'test_srcrec_a.dat')

    def test_subcase_01(self):
        sr = SrcRec.read(self.fname)
        sr.select_by_distance([0, 1])
        assert sr.rec_points.shape[0] == 19378

    def test_subcase_02(self):
        sr = SrcRec.read(self.fname)
        sr.select_by_box_region([-1, 0, -1, 0])
        assert sr.rec_points.shape[0] == 671 and sr.src_points.shape[0] == 85

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
        assert sr.src_points.shape[0] == 1413 and sr.rec_points.shape[0] == 11226
    
    def test_subcase_06(self):
        sr = SrcRec.read(self.fname)
        sr1 = SrcRec.read(self.fname1, dist_in_data=True)
        sr.append(sr1)
        assert sr.src_points.shape[0] == 2506 and sr._count_records() == 19926

    def test_subcase_07(self):
        sr = SrcRec.read(self.fname)
        sr.select_by_azi_gap(120)
        assert sr.src_points.shape[0] == 329 and sr.rec_points.shape[0] == 3815

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

if __name__ == '__main__':
    tsr = TestSrcRec()
    tsr.test_subcase_01()
    tsr.test_subcase_02()
    tsr.test_subcase_03()
    tsr.test_subcase_04()
    tsr.test_subcase_05()
    tsr.test_subcase_06()
    tsr.test_subcase_07()
    tsr.test_subcase_08()
    tsr.test_subcase_09()
    tsr.test_subcase_10()



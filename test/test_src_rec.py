import pytest
from pytomoatt.src_rec import SrcRec
from os.path import dirname, join


class TestSrcRec:
    fname: str = join(dirname(dirname(__file__)), 'examples', 'src_rec_file_eg')

    def test_subcase_01(self):
        self.sr = SrcRec.read(self.fname)
        self.sr.select_distance([0, 1])
        assert self.sr.rec_points.shape[0] == 19378

    def test_subcase_02(self):
        self.sr = SrcRec.read(self.fname)
        self.sr.select_box_region([-1, 0, -1, 0])
        assert self.sr.rec_points.shape[0] == 671 and self.sr.src_points.shape[0] == 85


if __name__ == '__main__':
    tsr = TestSrcRec()
    tsr.test_subcase_01()


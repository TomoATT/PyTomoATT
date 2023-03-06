from pytomoatt.src_rec import SrcRec
from subprocess import Popen


def test_from_seispy():
    s = 'wget https://osf.io/hzq2x/download -O ex-ccp.tar.gz\n'
    s += 'tar -xzf ex-ccp.tar.gz\n'
    proc = Popen(s, shell=True)
    proc.communicate()

    sr = SrcRec.from_seispy('ex-ccp/RFresult')
    sr.write('src_rec_seispy')


if __name__ == '__main__':
    # path = '/Users/xumijian/Codes/seispy-example/ex-ccp/RFresult'
    # sub_case01(path)
    pass
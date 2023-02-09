from os.path import dirname, abspath, join, exists
from os import makedirs
from shutil import rmtree, copy
import argparse
import argcomplete
import sys
from .src_rec import SrcRec


def init_project(path):
    inpara_path = join(dirname(abspath(__file__)), 'template', 'input_params.yml')
    src_rec_path = join(dirname(abspath(__file__)), 'template', 'src_rec.dat')
    if exists(path):
        rm_str = input('The {} already exists. Do you want to remove and recreate this directory? [Y/n]'.format(path))
        if rm_str.lower() == 'y':
            rmtree(path)
            makedirs(path)
        else:
            pass
    else:
        makedirs(path)
    copy(inpara_path, join(path, 'input_params.yml'))
    copy(src_rec_path, join(path, 'src_rec.dat'))


class PTA:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser(
        usage='''pta <command> [<args>]
The pta commands are:
init_pjt      Initialize a new project for TomoATT
gen_src_rec   Generate src_rec file from other format
''')
        parser.add_argument('command', help='pta commands')
        argcomplete.autocomplete(parser)
        args = parser.parse_args(sys.argv[1:2])
        getattr(self, args.command)()
    
    def __str__(self):
        return ''
    
    def init_pjt(self):
        parser = argparse.ArgumentParser(description='Initialize a new project for TomoATT')
        parser.add_argument('pjt_path', help='Path to project.')
        args = parser.parse_args(sys.argv[2:])
        init_project(args.pjt_path)

    def gen_src_rec(self):
        parser = argparse.ArgumentParser(description='Generate src_rec file from other format')
        parser.add_argument('-i', help='Path to input directory', required=True, metavar='fname')
        parser.add_argument('--seispy', help='Convert receiver function information from seispy.', action='store_true')
        parser.add_argument('-o', help='Path to output src_rec file', default='./src_rec', metavar='fname')
        args = parser.parse_args(sys.argv[2:])
        if args.seispy:
            try:
                sr = SrcRec.from_seispy(args.i)
            except Exception as e:
                print('Error in reading receiver functions')
                print('{}'.format(e))
                sys.exit(1)
            sr.write(args.o)

def main():
    PTA()

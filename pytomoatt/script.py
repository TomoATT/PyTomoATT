from os.path import dirname, abspath, join, exists
from os import makedirs
from shutil import rmtree, copy
import argparse
import argcomplete
import sys


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
init_pjt    Initialize a new project for TomoATT
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

def main():
    PTA()

from os.path import dirname, abspath, join, exists
from os import makedirs
from shutil import rmtree, copy
import argparse
import argcomplete
import sys
import h5py
from .src_rec import SrcRec
from .io.crust import CrustModel
from .para import ATTPara
from .utils import to_vtk, init_axis
from .checkerboard import Checker


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
\033[1minit_pjt\033[0m              Initialize a new project for TomoATT
\033[1mgen_src_rec\033[0m           Generate src_rec file from other format
\033[1mcreate_model\033[0m          Create model for TomoATT
\033[1mcreate_checkerboard\033[0m   Add perturbations on a model
\033[1mmodel2vtk\033[0m             Write model with h5 format to VTK format
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

    def create_model(self):
        parser = argparse.ArgumentParser(description='Create model for TomoATT from internal models: CRUST1.0')
        parser.add_argument('input_params', help='The parameter file of TomoATT, The section \"domain\" will be read.')
        parser.add_argument('-o', help='Path to output model', default='model_crust1.0_vp.h5', metavar='fname')
        parser.add_argument('-s', help='Smooth the 3D model with a Gaussian filter,' 
                            'Sigma is the standard division of the smoothing kernel, defaults to None',
                            default=None, type=float, metavar='sigma')
        args = parser.parse_args(sys.argv[2:])
        para = ATTPara(args.input_params)
        cm = CrustModel()
        cm.griddata(
            para.input_params['domain']['min_max_dep'],
            para.input_params['domain']['min_max_lat'],
            para.input_params['domain']['min_max_lon'],
            para.input_params['domain']['n_rtp'],
        )
        if args.s is not None:
            cm.smooth(args.s)
        cm.write(args.o)

    def create_checkerboard(self):
        parser = argparse.ArgumentParser(description='Add perturbations on a model')
        parser.add_argument('input_params', help='The parameter file of TomoATT, The section \"domain\" will be read.')
        parser.add_argument('-i', help='Path to input model file', required=True, metavar='fname')
        parser.add_argument('-n', help='nx, ny and nz pairs of anomalies along X, Y and Z', metavar='nx/ny/nz', required=True)
        parser.add_argument('-p', help='Amplitude of perturbations for velocity (pert_vel) and anisotropy (pert_ani)', 
                            metavar='pert_vel/pert_ani', default='0.08/0.04')
        parser.add_argument('-o', help='Path to output perturbed model', default='model_pert.h5', metavar='fname')
        parser.add_argument('-x', help='Upper and low bound for X direction', default=None, metavar='xmin/xmax')
        parser.add_argument('-y', help='Upper and low bound for Y direction', default=None, metavar='ymin/zmax')
        parser.add_argument('-z', help='Upper and low bound for Z direction', default=None, metavar='zmin/zmax')
        args = parser.parse_args(sys.argv[2:])
        para = ATTPara(args.input_params)
        cb = Checker(args.i)
        cb.init_axis(
            para.input_params['domain']['min_max_dep'],
            para.input_params['domain']['min_max_lat'],
            para.input_params['domain']['min_max_lon'],
            para.input_params['domain']['n_rtp'],
        )
        n_period = [float(v) for v in args.n.split('/')]
        pert = [float(v) for v in args.p.split('/')]
        if args.x is not None:
            lim_x = [float(v) for v in args.x.split('/')]
        else:
            lim_x = args.x
        if args.y is not None:
            lim_y = [float(v) for v in args.y.split('/')]
        else:
            lim_y = args.y
        if args.z is not None:
            lim_z = [float(v) for v in args.z.split('/')]
        else:
            lim_z = args.z
        cb.checkerboard(*n_period, *pert, lim_x=lim_x, lim_y=lim_y, lim_z=lim_z)
        cb.write(args.o)
        
    def model2vtk(self):
        parser = argparse.ArgumentParser(description='Write model with h5 format to VTK format')
        parser.add_argument('input_params', help='The parameter file of TomoATT, The section \"domain\" will be read.')
        parser.add_argument('-i', help='Path to input model file', required=True, metavar='fname')
        parser.add_argument('-o', help='Path to output VTK file', default='model.vtk', metavar='fname')
        args = parser.parse_args(sys.argv[2:])
        para = ATTPara(args.input_params)
        dep, lat, lon, _, _, _ = init_axis(
            para.input_params['domain']['min_max_dep'],
            para.input_params['domain']['min_max_lat'],
            para.input_params['domain']['min_max_lon'],
            para.input_params['domain']['n_rtp'],
        )
        with h5py.File(args.i) as model:
            to_vtk(args.o, model, dep, lat, lon)


def main():
    PTA()

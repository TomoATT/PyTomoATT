from os.path import dirname, abspath, join, exists
from os import makedirs
from shutil import rmtree, copy
import argparse
import sys
import h5py
from .src_rec import SrcRec
from .model import ATTModel
from .para import ATTPara
from .utils.common import to_vtk, init_axis, str2val
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
The pta commands include:
\033[1minit_pjt\033[0m              Initialize a new project for TomoATT
\033[1mgen_src_rec\033[0m           Generate src_rec file from other format
\033[1mcreate_model\033[0m          Create model for TomoATT
\033[1mcreate_checkerboard\033[0m   Add perturbations on a model
\033[1mmodel2vtk\033[0m             Write model with h5 format to VTK format
\033[1msetpar\033[0m                Set parameters for input_params.yml
''')
        parser.add_argument('command', help='pta commands')
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
        parser = argparse.ArgumentParser(description='Create model for TomoATT from external models: CRUST1.0 or custom model\n'
                                         'Ex1 (CRUST1.0): pta create_model -oatt_model.h5 -s5 -tvs input_params.yml\n'
                                         'Ex2 (custom model): pta create_model -m2 -iUSTClitho2.0.txt -oatt_model.h5 -s3 input_params.yml',
                                         formatter_class=argparse.RawTextHelpFormatter)
        parser.add_argument('input_params', help='The parameter file of TomoATT, The section \"domain\" will be read.')
        parser.add_argument('-m', help='Background model type. 1 for CRUST1.0, 2 for custom ASCII file, defaults to 1',
                            default=1, type=int, metavar='1|2')
        parser.add_argument('-i', help='Path to input custom model with ASCII format, only valid for -m2', metavar='fname')
        parser.add_argument('-c', help='Columns used in custom model file, order by lon/lat/dep/vel. defaults to 0/1/2/3',
                            default='0/1/2/3', metavar='ncol_lon/ncol_lat/ncol_dep/ncol_vel')
        parser.add_argument('-o', help='Path to output model, defaults to Sub_CRUST1.0_nr_nt_np.h5', default=None, metavar='fname')
        parser.add_argument('-s', help='Smooth the 3D model with a Gaussian filter,' 
                            'Sigma is the standard division of the smoothing kernel in km, defaults to None',
                            default=None, type=float, metavar='sigma')
        parser.add_argument('-t', help='Type of velocity vp or vs are available, valid for -m1', default='vp', metavar='vel_type')
        args = parser.parse_args(sys.argv[2:])
        mod = ATTModel(args.input_params)
        if args.m == 1:
            mod.grid_data_crust1(args.t)
        elif args.m == 2:
            try:
                usecols = [int(c) for c in args.c.split('/')]
            except:
                raise ValueError('Columns should be in format of ncol_lon/ncol_lat/ncol_dep/ncol_vel')
            if not exists(args.i):
                raise FileNotFoundError('No such model file of {}'.format(args.i))
            mod.grid_data_ascii(args.i, usecols=usecols)
        else:
            raise ValueError('Invalid model type of {}'.format(args.m))
        if args.s is not None:
            mod.smooth(args.s)
        mod.write(args.o)

    def create_checkerboard(self):
        parser = argparse.ArgumentParser(description='Add perturbations on a model')
        parser.add_argument('input_params', help='The parameter file of TomoATT, The section \"domain\" will be read.')
        parser.add_argument('-a', help='nx, ny, nz, and FVD of anisotropic anomalies along longitude, '
                                        'latitude and depth, and fast-velocity-direction, defaults to None', 
                            metavar='nx/ny/nz[/fvd]', default=None)
        parser.add_argument('-i', help='Path to input model file', required=True, metavar='fname')
        parser.add_argument('-n', help='nx, ny and nz of velocity anomalies along longitude, latitude and depth', metavar='nx/ny/nz', required=True)
        parser.add_argument('-o', help='Path to output perturbed model', default=None, metavar='fname')
        parser.add_argument('-p', help='Amplitude of perturbations for velocity (pert_vel) and anisotropy (pert_ani)', 
                            metavar='pert_vel/pert_ani', default='0.08/0.04')
        parser.add_argument('-x', help='Upper and low bound for longitude direction', default=None, metavar='xmin/xmax')
        parser.add_argument('-y', help='Upper and low bound for latitude direction', default=None, metavar='ymin/ymax')
        parser.add_argument('-z', help='Upper and low bound for depth direction', default=None, metavar='zmin/zmax')
        args = parser.parse_args(sys.argv[2:])
        cb = Checker(args.i, para_fname=args.input_params)
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
        if args.a is not None:
            ani_lst = args.a.split('/')
            n_ani = [float(v) for v in ani_lst[0:3]]
            if len(ani_lst) > 3:
                ani_dir = float(ani_lst[3])
            else:
                ani_dir = 45.0
            cba = cb.copy()
            cba.checkerboard(*n_ani, *pert, lim_x=lim_x, lim_y=lim_y, lim_z=lim_z, ani_dir=ani_dir)
            cb.epsilon = cba.epsilon
            cb.phi = cba.phi
            cb.xi = cba.xi
            cb.eta = cba.eta
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

    def setpar(self):
        parser = argparse.ArgumentParser(description='Set parameters for input_params.yml')
        parser.add_argument('input_params', help='The parameter file of TomoATT, The section \"domain\" will be read.')
        parser.add_argument('key', help='The key of parameter file to be set. Use \".\" to separate the keys.')
        parser.add_argument('value', help='The value of parameter file to be set.')
        parser.add_argument('-o', help='Path to output parameter file, defaults to overwrite input_params.yml', default=None, metavar='fname')
        args = parser.parse_args(sys.argv[2:])

        # Read the parameter file
        para = ATTPara(args.input_params)
        
        # Update the desired parameter
        para.update_param(args.key, args.value)

        # Write the parameter file
        para.write(fname=args.o)


def main():
    PTA()

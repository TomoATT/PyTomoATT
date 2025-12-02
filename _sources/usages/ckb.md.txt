# Create checkerboard model

We have provided a simple command line interface `pta create_checkerboard` to create a typical checkerboard model. The help message of this command is as follows:

```bash
usage: pta create_checkerboard [-h] [-a nx/ny/nz] -i fname -n nx/ny/nz [-o fname] [-p pert_vel/pert_ani] [-x xmin/xmax] [-y ymin/ymax] [-z zmin/zmax] input_params

Add perturbations on a model

positional arguments:
  input_params          The parameter file of TomoATT, The section "domain" will be read.

options:
  -h, --help            show this help message and exit
  -a nx/ny/nz           nx, ny and nz of anisotropic anomalies along longitude, latitude and depth, defaults to the
                        same as -n
  -i fname              Path to input model file
  -n nx/ny/nz           nx, ny and nz of velocity anomalies along longitude, latitude and depth
  -o fname              Path to output perturbed model
  -p pert_vel/pert_ani  Amplitude of perturbations for velocity (pert_vel) and anisotropy (pert_ani)
  -x xmin/xmax          Upper and low bound for longitude direction
  -y ymin/ymax          Upper and low bound for latitude direction
  -z zmin/zmax          Upper and low bound for depth direction
  ```

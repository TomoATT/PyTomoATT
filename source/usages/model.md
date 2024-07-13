# Create initial model

TomoATT requires a 3D initial model to start the inversion. The initial model is a `h5` file that contains the velocity and anisotropy information. The velocity and anisotropy are stored in Three datasets, `vel`, `xi`, and `eta` respectively. The shape of the datasets is `(nr, nt, nt)`, where `nr`, `nt`, and `np` are the number of grid points along the depth, latitude, and longitude, respectively. These values are defined in the `domain` section of the parameter file with key `n_rtp`. 

The range of the model is defined by the `domain` section of the parameter file. The `domain` section should contain the following keys:

- `min_max_dep`: The minimum and maximum depth of the model in km.
- `min_max_lat`: The minimum and maximum latitude of the model in degrees.
- `min_max_lon`: The minimum and maximum longitude of the model in degrees.

PyTomoATT provides a simple command line interface `pta create_model` to create a 3D initial model. The help message of this command is as follows:

```bash
usage: pta create_model [-h] [-m 1|2] [-i fname] [-c ncol_lon/ncol_lat/ncol_dep/ncol_vel] [-o fname] [-s sigma] [-t vel_type] input_params

Create model for TomoATT from external models: CRUST1.0 or custom model
Ex1 (CRUST1.0): pta create_model -oatt_model.h5 -s5 -tvs input_params.yml
Ex2 (custom model): pta create_model -m2 -imodel.txt -oatt_model.h5 -s3 input_params.yml

positional arguments:
  input_params          The parameter file of TomoATT, The section "domain" will be read.

options:
  -h, --help            show this help message and exit
  -m 1|2                Background model type. 1 for CRUST1.0, 2 for custom ASCII file, defaults to 1
  -i fname              Path to input custom model with ASCII format, only valid for -m2
  -c ncol_lon/ncol_lat/ncol_dep/ncol_vel
                        Columns used in custom model file, order by lon/lat/dep/vel. defaults to 0/1/2/3
  -o fname              Path to output model, defaults to Sub_CRUST1.0_nr_nt_np.h5
  -s sigma              Smooth the 3D model with a Gaussian filter,Sigma is the standard division of the smoothing kernel in km, defaults to None
  -t vel_type           Type of velocity vp or vs are available, valid for -m1
```

## Using internal CRUST1.0 model

PyTomoATT provide a internal CRUST1.0 model, which is a 3D model (Vp and Vs) based on the CRUST1.0 model. The model is stored in the `data` folder of the PyTomoATT package.

Using `-m1` in `pta create_model` to create a 3D model based on the CRUST1.0 model, for example:

```bash

pta create_model -m1 -oatt_model.h5 -s15 -tvs input_params.yml

```

In this example, the command will create a 3D S-wave velocity model based on the CRUST1.0 model, smooth the model with a Gaussian filter with a standard division of 15 km, and save the model to `att_model.h5`. The size of the model is defined in the `domain` section of the parameter file.

## Using custom model

PyTomoATT also supports creating a 3D model based on a custom ASCII file. The ASCII file should contain the velocity information in the format of `lon lat dep vel`. The columns used in the ASCII file can be specified by `-c` option. The order of the columns should be longitude, latitude, depth, and velocity. The default order is 0, 1, 2, and 3. The ASCII file can be specified by `-i` option. For example:

```bash

pta create_model -m2 -i model.txt -oatt_model.h5 -s20 input_params.yml

```

In this example, the command will create a 3D model based on the ASCII file `model.txt`, smooth the model with a Gaussian filter with a standard division of 20 km, and save the model to `att_model.h5`. The size of the model is defined in the `domain` section of the parameter file.
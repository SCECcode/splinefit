# BSpline fit of the community fault model (developer version)
This directory contains scripts to run a developer version of the CFM spline
fitting tool. A detailed description of what the tool does can be found in the
[documentation](../docs/README.md). These notes describe how to run the tool.

## Installation
Make sure to install the `splinefit` package. Follow the instructions on the
main page, found [here](../README.md).

You need to have `CFM5_release_2017` available. If possible, you should place
this release in `usr/local/` If you are unable to place it here, then you need
to modify the variable `cfm` in the `Makefile` or, each time, call

```
make cfm=path_to_CFM5_release_2017
```

In addition to the path the CFM, the python executable must also be specified.
This version has been test using Python 3.6.3. Set the variable `PY` to specify
python executable. The default is `PY=python3`, but you may need to change this
to `PY=python`.

## Run 

To run the tool, you specify what mesh you want to process. The simplest option
is to choose any of the 12 selected candidate faults from the pilot study.
These twelve faults are

1. PNRA-CRSF-USAV-Fontana_Seismicity_lineament-CFM1
2. GRFS-GRFZ-WEST-Garlock_fault-CFM5
3. WTRA-NCVS-VNTB-Southern_San_Cayetano_fault-steep-JHAP-CFM5
4. WTRA-ORFZ-SFNV-Northridge-Frew_fault-CFM2
5. WTRA-SSFZ-MULT-Santa_Susana_fault-CFM1
6. SAFS-SAFZ-MULT-Garnet_Hill_fault_strand-CFM4
7. SAFS-SAFZ-COAV-Southern_San_Andreas_fault-CFM4
8. WTRA-SFFS-SMMT-Santa_Monica_fault-steep-CFM5
9. WTRA-SBTS-SMMT-Santa_Monica_thrust_fault-CFM1
10. PNRA-NIRC-LABS-Newport-Inglewood_fault-dip_w_splays-split-CFM5
11. PNRA-CSTL-SJQH-San_Joaquin_Hills_fault-truncated-CFM3
12. PNRA-CEPS-LABS-Compton-Los_Alamitos_fault-CFM2

Simply pass `num` to make for the fault you want to process. For example, to
process *Southern San Cayetano fault*, use
```bash
$ make num=4
```
If it is the first time you process this fault (or you have deleted the
generated output data), then you need to run the command twice.

## Output
Assuming that the processing went fine, then you should have the following
directories and files created.
```
figures/
pydata/
part_*.vtk
part_*.msh
part_*_bspline_surf_fit.json
part_*_bspline_surf_fit.vtk
```
The figure directory contains some figures that were generated during different
steps. The pydata directory contains a dump of binary Python data that was
generated or during each processing step. This data is needed by subsequent
steps. If you want to redo a particular step, you can delete that particular
data file.

The files in the root of the output directory contains a conversion of each
surface found in the original `.ts` file to either gmsh `.msh` or legacy vtk
`.vtk`. The number, for example `part_0`, `part_1` references a specific
surface in the `.ts` file and they are ordered after how they are listed in the
source file. The `part_*_bspline_surf_fit.vtk` is the final BSpline surface fit
that has been evaluated using a certain number of grid points in each
direction.

### JSON data
The JSON data files `_part_*_bspline_surf_fit.json` contains all of the data
needed to reconstruct the BSpline surface using some other software. The fields are

| Field(s)      | Description                                                                                                                                                                      |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `U`           | Knot vectors for clamped BSplines in the u and v-directions (1D array) .                                                                                                         |
| `Px`, `Py`, `Pz` | Control points in reference coordinates. Each coordinate is a 2D array, e.g. `X[i,j]`, `i` is the index for the v-direction and `j` is the index for the u-direction in (u,v)-parameter space. |
| `real_world_Px`, `real_world_Py`, `real_world_Pz` | Control points in real-world coordinates.  | |
| `pu`, `pv`    | Polynomial degree of the BSpline in each direction (integer).                                                                                                                    |

The number of control points `n` is equal to `n = m - p - 1`, where `m` is the
number of knots, and `p` is the polynomial degree.

## Release
All of the data that is placed in the `output` directory can be compressed into
a zip archive. Use

```bash
$ make release NAME=..  VERSION=.. YEAR=.. 
```
To create the zip archive. The zip archive will be placed in the `releases`
directory under the filename `NAME_VERSION_YEAR.zip`.

Currently, only the data in the output directory is saved. Hence the settings
used to produce these results are lost. It also save all intermediate data
(such as Python binary dumps), which might be undesirable. These drawbacks will
be addressed in the near future.

## Release notes
### CFM5_pilot_bspline_beta_1.0_2018 

### PNRA-CRSF-USAV-Fontana_Seismicity_lineament-CFM1
| Compiles | Number of part(s) | Warnings |
|----------|-------------------|----------|
| Yes      | 1                 | None     |

Fit is not good because one of the boundaries has become quite curved. 

### GRFS-GRFZ-WEST-Garlock_fault-CFM5
| Compiles | Number of part(s) | Warnings |
|----------|-------------------|----------|
| Yes      | 1                 | None     |

Curvature in the original surface is not well-preserved, and one one boundary
corner is more curved away from original geometry. Some improvement to the
boundary fit is desirable.

### WTRA-NCVS-VNTB-Southern_San_Cayetano_fault-steep-JHAP-CFM5
| Compiles | Number of part(s) | Warnings                                               |
|----------|-------------------|--------------------------------------------------------|
| Yes      | 1                 | Divide by zero for point 1 in part 1 during UV mapping |

Fit is not so good because the boundary is too oscillatory in some places.

### WTRA-ORFZ-SFNV-Northridge-Frew_fault-CFM2
| Compiles | Number of part(s) | Warnings                                                                                   |
|----------|-------------------|--------------------------------------------------------------------------------------------|
| Yes      | 2                 | Divide by zero for point 30, 760 in part 0 during UV mapping, and for point 80, in part 1. |

Can only fit the first surface, which is a deformed, circular patch. Second
part cannot be fitted properly because it is a plane with a hole in it.

### WTRA-SSFZ-MULT-Santa_Susana_fault-CFM1
| Compiles | Number of part(s) | Warnings |
|----------|-------------------|----------|
| Yes      | 2                 | None     |

Part 0 (the base) is under fitted in some places, curvature of original
geometry is not respected. Fit is too curved near boundaries. Part 1 cannot be fitted (the top segment), large distortions are produced. The original geometry has many interesting features, that breaks the assumption of planarity.

### SAFS-SAFZ-MULT-Garnet_Hill_fault_strand-CFM4 

| Compiles | Number of part(s) | Warnings |
|----------|-------------------|----------|
| Yes      | 7                 | None     |

Overall, the fit is quite good. There are some parts where the curvature of the
original geometry is not respect (look at the top section for example). Many of
the problems appear to be at the boundaries.

### SAFS-SAFZ-COAV-Southern_San_Andreas_fault-CFM4 
| Compiles | Number of part(s) | Warnings |
|----------|-------------------|----------|
| Yes      | 1                 | None     |

Fit is pretty good. Hard to see errors. One of the corners is slightly off.

### WTRA-SFFS-SMMT-Santa_Monica_fault-steep-CFM5
| Compiles | Number of part(s) | Warnings                                                                                                                         |
|----------|-------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Yes      | 10                | Too few points to fit cubic polynomials. Linear ones used on some sides. Divide by zero for point 50 in part 1 during UV mapping |

Boundary fit is extremely poor for some parts. Part 3 is completely wrong, its
corners are highly stretched out. Overall impression, without Part 3 the fit
looks pretty good. There are some boundaries that do not curve correctly and
create gaps that the original geometry does not contain.


### WTRA-SBTS-SMMT-Santa_Monica_thrust_fault-CFM1
| Compiles | Number of part(s) | Warnings                                                                                                                                                              |
|----------|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Yes      | 3                 | Too few points to fit cubic polynomials. Linear ones used on some sides. Divide by zero for point (10, 90), (190, 340) (100) in part (0), (1),  (3) during UV mapping |


Boundary fit does not work so well for some parts. There are certain boundaries
that only have two points on one side, but many more on the opposite side. When
there are fewer points than polynomial degree - 1, the degree is adjusted until
the data can be interpolated. Since the same number of control points must be
used on both sides, the procedure is forced to fit a complex shape with nearly
no points, and that is impossible. An improvement here would be to insert more
points on the side with few points using some insertion method (nearest
neighbor interpolation?). 

The surface of part 0 has become inverted. Unclear what caused this artifact. The other parts are fine. Some of the structure in the original geometry is not preserved (see for example part 1).

### PNRA-NIRC-LABS-Newport-Inglewood_fault-dip_w_splays-split-CFM5
| Compiles | Number of part(s) | Warnings                                                                                                                                             |
|----------|-------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| Yes      | 9                 | Too few points to fit cubic polynomials. Linear ones used on some sides. Divide by zero for point (30),  (0, 170) in part (5), (8) during UV mapping |

Original surface contains a hole. Procedure identifies the hole as a boundary
and tries to fit a surface to the hole. Actual geometry is ignored.  To fit the
hole, the original geometry should be split into multiple patches, apply the
fitting to each one, and then stitch them together.

The other parts looks ok. The fit for the branch is slightly off.

### PNRA-CSTL-SJQH-San_Joaquin_Hills_fault-truncated-CFM3
| Compiles | Number of part(s) | Warnings                                                |
|----------|-------------------|---------------------------------------------------------|
| Yes      | 1                 | Divide by zero for point 10 in part 0 during UV mapping |

Fit looks good except for one corner edge that is not respected.

### PNRA-CEPS-LABS-Compton-Los_Alamitos_fault-CFM2

| Compiles | Number of part(s) | Warnings                                                                 |
|----------|-------------------|--------------------------------------------------------------------------|
| Yes      | 1                 | Too few points to fit cubic polynomials. Linear ones used on some sides. |

Fit looks good except for boundary curvature that is not respected.

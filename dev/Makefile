PY=python3
NAME=CFM5_pilot_bspline
VERSION=beta_1.1
YEAR=2018
cfm=/usr/local/CFM5_release_2017/obj/CFM5_preferred/native

### General options
# show_plot : Show figures that are being generated. When enabled, user
# interaction is required. Plot window must be closed to resume operation
show_plot = 0

### Options for mesh conversion
# min_elem : Minimum number of elements a triangular surface must contain to
# get processed
min_elem = 20


### Options for BSpline fitting
# bnd_degree : Polynomial degree
# residual : Desired residual to obtain. More control points are inserted until
# 	this threshold is reached
# bnd_smooth : Amount of smoothness to add in the fitting. There might be a bug
# 	in the implementation of this. Use with caution.
# bnd_refinement_ratio : Refine sides with unbalanced number of nodes compared
# 	to opposing side. Refinement inserts points at midpoints.
# surf_residual : Unused.
# surf_invert_mapping : Determine UV-mapping by inverting the transfinite
# 	interpolation mapping.
bnd_degree = 3
bnd_residual = 1.8
bnd_smooth = 0.01
bnd_refinement_ratio = 3
surf_residual = 0.1
surf_invert_mapping = 1

# List of faults to process
# Each fault is processed sequentially by calling `make num=1`
# where `num` is the index of the fault in the array.
num=1
fault1=PNRA-CRSF-USAV-Fontana_Seismicity_lineament-CFM1
fault2=GRFS-GRFZ-WEST-Garlock_fault-CFM5
fault3=WTRA-NCVS-VNTB-Southern_San_Cayetano_fault-steep-JHAP-CFM5
fault4=WTRA-ORFZ-SFNV-Northridge-Frew_fault-CFM2
fault5=WTRA-SSFZ-MULT-Santa_Susana_fault-CFM1
fault6=SAFS-SAFZ-MULT-Garnet_Hill_fault_strand-CFM4
fault7=SAFS-SAFZ-COAV-Southern_San_Andreas_fault-CFM4
fault8=WTRA-SFFS-SMMT-Santa_Monica_fault-steep-CFM5
fault9=WTRA-SBTS-SMMT-Santa_Monica_thrust_fault-CFM1
fault10=PNRA-NIRC-LABS-Newport-Inglewood_fault-dip_w_splays-split-CFM5
fault11=PNRA-CSTL-SJQH-San_Joaquin_Hills_fault-truncated-CFM3
fault12=PNRA-CEPS-LABS-Compton-Los_Alamitos_fault-CFM2
faults=$(fault1) $(fault2) $(fault3) $(fault4) $(fault5) $(fault6) $(fault7) \
       $(fault8) $(fault9) $(fault10) $(fault11) $(fault12)
fault=$(word $(num), $(faults)).msh

# Name of output directories
out=output2
figures=figures
fig=png
pydata=pydata
releasedir=releases

meshdir=$(out)/$(basename $(fault))
meshes=$(basename $(wildcard $(meshdir)/*.msh))

all: init gmsh boundary projection rotation segmentation bspline-boundary \
     bspline-surface   
fit: clean-fit bspline-boundary bspline-surface

init:
	mkdir -p $(out)
	mkdir -p $(meshdir)
	mkdir -p $(meshdir)/$(pydata)
	mkdir -p $(meshdir)/$(figures)

$(meshdir)/%.msh:
	tsurfmsh $(cfm)/$(notdir $(basename $@)).ts $(meshdir)/part.msh --min_elem=$(min_elem)

%.vtk: %.msh
	mshvtk $< $@

clean-fit:
	rm -f $(meshdir)/$(pydata)/*_bnd.p
	rm -f $(meshdir)/$(pydata)/*_surf.p

$(meshdir)/$(pydata)/%_bnd.p: $(meshdir)/%.msh
	@echo Extracting boundary data from triangular mesh: $(notdir $<)
	$(PY) python/boundary.py $< $(meshdir)/$(pydata)/$(notdir $@) \
 	$(meshdir)/$(figures)/$(basename $(notdir $@)).$(fig) \
	$(show_plot)

$(meshdir)/$(pydata)/%_proj.p: $(meshdir)/$(pydata)/%_bnd.p
	@echo Projecting boundary data onto bestfitting plane for $(notdir $<)
	$(PY) python/projection.py $< $(meshdir)/$(pydata)/$(notdir $@) \
 	$(meshdir)/$(figures)/$(basename $(notdir $@)).$(fig) \
	$(show_plot)

$(meshdir)/$(pydata)/%_rot.p: $(meshdir)/$(pydata)/%_proj.p
	@echo Rotating coordinate system for $(notdir $<)
	$(PY) python/rotation.py $< $(meshdir)/$(pydata)/$(notdir $@) \
 	$(meshdir)/$(figures)/$(basename $(notdir $@)).$(fig) \
	$(show_plot)

$(meshdir)/$(pydata)/%_seg.p: $(meshdir)/$(pydata)/%_rot.p
	@echo Detecting boundary segments for $(notdir $<)
	$(PY) python/segment.py $< $(meshdir)/$(pydata)/$(notdir $@) \
 	$(meshdir)/$(figures)/$(basename $(notdir $@)).$(fig) \
	$(show_plot)

$(meshdir)/$(pydata)/%_bspline_bnd.p: $(meshdir)/$(pydata)/%_seg.p
	@echo Fitting BSpline curve through boundary for $(notdir $<)
	$(PY) python/bspline-boundary.py $< $(meshdir)/$(pydata)/$(notdir $@) \
	$(bnd_degree) $(bnd_residual) $(bnd_smooth) $(bnd_refinement_ratio)\
 	$(meshdir)/$(figures)/$(basename $(notdir $@)).$(fig) \
	$(show_plot)

$(meshdir)/$(pydata)/%_bspline_surf.p: $(meshdir)/$(pydata)/%_bspline_bnd.p
	@echo Fitting BSpline surface for $(notdir $<)
	$(PY) python/bspline-surface.py $< $(meshdir)/$(pydata)/$(notdir $@) \
	$(surf_invert_mapping)\
 	$(meshdir)/$(basename $(notdir $@))_fit.vtk \
 	$(meshdir)/$(basename $(notdir $@))_fit.json \
 	$(meshdir)/$(figures)/$(basename $(notdir $@))_uv.$(fig) \
	$(show_plot)
	      
gmsh: init $(meshdir)/$(fault)

vtk: $(foreach mesh,$(meshes),$(mesh).vtk)

boundary: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_bnd.p)
projection: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_proj.p)
rotation: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_rot.p)
segmentation: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_seg.p)
bspline-boundary: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_bspline_bnd.p)
bspline-surface: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_bspline_surf.p)

release:
	mkdir -p $(releasedir)
	mkdir -p $(NAME)_$(VERSION)_$(YEAR)
	cp -r $(out)/* $(NAME)_$(VERSION)_$(YEAR)
	zip  $(releasedir)/$(NAME)_$(VERSION)_$(YEAR).zip -r $(NAME)_$(VERSION)_$(YEAR)  
	rm -r $(NAME)_$(VERSION)_$(YEAR)

clean:
	rm -rf $(meshdir)/*

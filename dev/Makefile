PY=python3

# List of faults to process
# Each fault is processed sequentially by calling `make num=1`
# where `num` is the index of the fault in the array.

# Name of output directories
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
	$(bnd_degree) $(bnd_residual) $(bnd_smooth) $(bnd_refinement_ratio) \
	$(bnd_regularization) \
	$(use_mapping) \
 	$(meshdir)/$(figures)/$(basename $(notdir $@)).$(fig) \
	$(show_plot)

$(meshdir)/$(pydata)/%_bspline_surf.p: $(meshdir)/$(pydata)/%_bspline_bnd.p
	@echo Fitting BSpline surface for $(notdir $<)
	$(PY) python/bspline-surface.py $< $(meshdir)/$(pydata)/$(notdir $@) \
	$(surf_invert_mapping) $(surf_smooth) \
	$(bnd_regularization) \
	$(use_mapping) \
 	$(meshdir)/$(basename $(notdir $@))_fit.vtk \
 	$(meshdir)/$(basename $(notdir $@))_fit.json \
 	$(meshdir)/$(figures)/$(basename $(notdir $@))_uv.$(fig) \
	$(show_plot)

$(meshdir)/$(pydata)/%_error.p: $(meshdir)/$(pydata)/%_bspline_surf.p
	@echo Computing error for surface for $(notdir $<)
	$(PY) python/error.py $< $(meshdir)/$(pydata)/$(notdir $@) \
 	$(meshdir)/$(basename $(notdir $@)).vtk 
	      
gmsh: init $(meshdir)/$(fault)

vtk: $(foreach mesh,$(meshes),$(mesh).vtk)

boundary: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_bnd.p)
projection: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_proj.p)
rotation: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_rot.p)
segmentation: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_seg.p)
bspline-boundary: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_bspline_bnd.p)
bspline-surface: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_bspline_surf.p)
error: $(foreach mesh,$(meshes),$(meshdir)/$(pydata)/$(notdir $(mesh))_error.p)

release:
	mkdir -p $(releasedir)
	mkdir -p $(NAME)_$(VERSION)_$(YEAR)
	cp -r $(out)/* $(NAME)_$(VERSION)_$(YEAR)
	zip  $(releasedir)/$(NAME)_$(VERSION)_$(YEAR).zip -r $(NAME)_$(VERSION)_$(YEAR)  
	rm -r $(NAME)_$(VERSION)_$(YEAR)

clean:
	rm -rf $(meshdir)/*


PY=python3
cfm=/usr/local/CFM5_release_2017/obj/CFM5_preferred/native
out=meshes
data=data
figures=figures
# p: polynomial degree
# sb : smoothing parameter for boundary curve fitting
p=3
sb=0.15

fault1=PNRA-CRSF-USAV-Fontana_Seismicity_lineament-CFM1
fault2=GRFS-GRFZ-WEST-Garlock_fault-CFM5
fault3=WTRA-NCVS-VNTB-Southern_San_Cayetano_fault-steep-JHAP-CFM5
fault4=WTRA-ORFZ-SFNV-Northridge-Frew_fault-CFM2
fault5=WTRA-SSFZ-MULT-Santa_Susana_fault-CFM1
fault6a=SAFS-SAFZ-MULT-Garnet_Hill_fault_strand-CFM4
fault6b=SAFS-SAFZ-COAV-Southern_San_Andreas_fault-CFM4
fault7a=WTRA-SFFS-SMMT-Santa_Monica_fault-steep-CFM5                     
fault7b=WTRA-SBTS-SMMT-Santa_Monica_thrust_fault-CFM1                    
fault8a=PNRA-NIRC-LABS-Newport-Inglewood_fault-dip_w_splays-split-CFM5
fault8b=PNRA-CSTL-SJQH-San_Joaquin_Hills_fault-truncated-CFM3            
fault8c=PNRA-CEPS-LABS-Compton-Los_Alamitos_fault-CFM2                   
fault=$(fault5)

test: test-init test-convert test-boundary test-projection test-rotation \
      test-segment test-bspline-boundary


define init
	@mkdir -p $(data)/$(1)
	@mkdir -p $(figures)/$(1)
endef

define mesh 
	mkdir -p $(out)/$(1)
	tsurfmsh $(cfm)/$(2).ts $(out)/$(1)/$(2).msh
endef

define boundary 
	$(PY) boundary.py $(out)/$(1)/$(2).msh $(data)/$(1)/$(2)_boundary.p \
		$(figures)/$(1)/$(2)_boundary.png
endef

define projection 
	$(PY) projection.py $(data)/$(1)/$(2)_boundary.p $(data)/$(1)/$(2)_projection.p \
		$(figures)/$(1)/$(2)_projection.png
endef

define rotation
	$(PY) rotation.py $(data)/$(1)/$(2)_projection.p \
		$(data)/$(1)/$(2)_rotation.p \
		$(figures)/$(1)/$(2)_rotation.png
endef

define quad
	$(PY) quad.py $(data)/$(1)/$(2)_rotation.p \
		$(data)/$(1)/$(2)_quad.p \
		$(figures)/$(1)/$(2)_quad.png
endef

define segment
	$(PY) segment.py $(data)/$(1)/$(2)_rotation.p \
		$(data)/$(1)/$(2)_segment.p \
		$(figures)/$(1)/$(2)_segment.png
endef

define bspline-boundary
	$(PY) bspline-boundary.py $(data)/$(1)/$(2)_segment.p \
		$(data)/$(1)/$(2)_bspline-boundary.p \
		$(3) $(4) \
		$(figures)/$(1)/$(2)_bspline-boundary.png
endef

test-init:
	$(call init,test)

test-convert:
	$(call mesh,test,$(fault))

test-boundary:
	$(call boundary,test,$(fault))

test-projection:
	$(call projection,test,$(fault))

test-rotation:
	$(call rotation,test,$(fault))

test-quad:
	$(call quad,test,$(fault))

test-segment:
	$(call segment,test,$(fault))

test-bspline-boundary:
	$(call bspline-boundary,test,$(fault),$(p),$(sm))

init-easy:
	$(call init,easy)
convert-easy:
	$(call mesh,easy,$(fault1))
	$(call mesh,easy,$(fault2))
	$(call mesh,easy,$(fault3))

boundary-easy:
	$(call boundary,easy,$(fault1))
	$(call boundary,easy,$(fault2))
	$(call boundary,easy,$(fault3))

projection-easy:
	$(call projection,easy,$(fault1))
	$(call projection,easy,$(fault2))
	$(call projection,easy,$(fault3))

rotation-easy:
	$(call rotation,easy,$(fault1))
	$(call rotation,easy,$(fault2))
	$(call rotation,easy,$(fault3))

segment-easy:
	$(call segment,easy,$(fault1))
	$(call segment,easy,$(fault2))
	$(call segment,easy,$(fault3))

bspline-boundary-easy:
	#$(call bspline-boundary,easy,$(fault1),3,0.1)
	#$(call bspline-boundary,easy,$(fault2),3,0.1)
	$(call bspline-boundary,easy,$(fault3),3,0.15)

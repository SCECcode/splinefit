PY=python3
cfm=/usr/local/CFM5_release_2017/obj/CFM5_preferred/native
out=meshes
data=data
figures=figures

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
fault=$(fault1)

all: test-convert test-boundary test-projection test-rotation
define mesh 
	mkdir -p $(out)/$(1)
	tsurfmsh $(cfm)/$(2).ts $(out)/$(1)/$(2).msh
endef

define boundary 
	@mkdir -p $(data)/$(1)
	@mkdir -p $(figures)/$(1)
	$(PY) boundary.py $(out)/$(1)/$(2).msh $(data)/$(1)/$(2)_boundary.msh \
		$(figures)/$(1)/$(2)_boundary.png
endef

define projection 
	@mkdir -p $(data)/$(1)
	@mkdir -p $(figures)/$(1)
	$(PY) projection.py $(data)/$(1)/$(2)_boundary.msh $(data)/$(1)/$(2)_projection.msh \
		$(figures)/$(1)/$(2)_projection.png
endef

define rotation
	@mkdir -p $(data)/$(1)
	@mkdir -p $(figures)/$(1)
	$(PY) rotation.py $(data)/$(1)/$(2)_projection.p \
		$(data)/$(1)/$(2)_rotation.p \
		$(figures)/$(1)/$(2)_rotation.png
endef


# Select all "easy" faults and convert to .msh files
test-convert:
	$(call mesh,test,$(fault))

test-boundary:
	$(call boundary,test,$(fault))

test-projection:
	$(call projection,test,$(fault))

test-rotation:
	$(call rotation,test,$(fault))

convert-easy:
	$(call mesh,easy1,$(fault1))
	$(call mesh,easy2,$(fault2))
	$(call mesh,easy3,$(fault3))

convert-moderate-difficulty:
	$(call mesh,medium1,$(fault4))
	$(call mesh,medium2,$(fault5))
	$(call mesh,medium3,$(fault6))
	$(call mesh,medium3,$(fault6b))

convert-difficult:
	$(call mesh,hard1,$(fault7a))
	$(call mesh,hard1,$(fault7b))
	$(call mesh,hard2,$(fault8a))
	$(call mesh,hard2,$(fault8b))
	$(call mesh,hard2,$(fault8c))

boundary-easy:
	$(call boundary,easy1,$(fault1))
	$(call boundary,easy2,$(fault2))
	$(call boundary,easy3,$(fault3))

boundary-moderate-difficulty:
	$(call boundary,easy1,$(fault1))
	$(call boundary,easy2,$(fault2))
	$(call boundary,easy3,$(fault3))

projection-easy:
	$(call projection,easy1,$(fault1))
	$(call projection,easy2,$(fault2))
	$(call projection,easy3,$(fault3))

rotation-easy:
	$(call rotation,easy1,$(fault1))
	$(call rotation,easy2,$(fault2))
	$(call rotation,easy3,$(fault3))

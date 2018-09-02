cfm=/usr/local/CFM5_release_2017/obj/CFM5_preferred/native
out=meshes

all: convert-easy convert-moderate-difficulty convert-difficult

define mesh =
	mkdir -p $(out)/$(1)
	tsurfmsh $(cfm)/$(2).ts $(out)/$(1)/$(2).msh
endef


# Select all "easy" faults and convert to .msh files
convert-easy:
	$(call mesh,easy1,PNRA-CRSF-USAV-Fontana_Seismicity_lineament-CFM1)
	$(call mesh,easy2,GRFS-GRFZ-WEST-Garlock_fault-CFM5)
	$(call mesh,easy3,WTRA-NCVS-VNTB-Southern_San_Cayetano_fault-steep-JHAP-CFM5)

convert-moderate-difficulty:
	$(call mesh,medium1,WTRA-ORFZ-SFNV-Northridge-Frew_fault-CFM2)
	$(call mesh,medium2,WTRA-SSFZ-MULT-Santa_Susana_fault-CFM1)
	$(call mesh,medium3,SAFS-SAFZ-MULT-Garnet_Hill_fault_strand-CFM4)
	$(call mesh,medium3,SAFS-SAFZ-COAV-Southern_San_Andreas_fault-CFM4)

convert-difficult:
	$(call mesh,hard1,WTRA-SFFS-SMMT-Santa_Monica_fault-steep-CFM5)  
	$(call mesh,hard1,WTRA-SBTS-SMMT-Santa_Monica_thrust_fault-CFM1)
	$(call mesh,hard2,PNRA-NIRC-LABS-Newport-Inglewood_fault-dip_w_splays-split-CFM5)
	$(call mesh,hard2,PNRA-CSTL-SJQH-San_Joaquin_Hills_fault-truncated-CFM3)
	$(call mesh,hard2,PNRA-CEPS-LABS-Compton-Los_Alamitos_fault-CFM2)

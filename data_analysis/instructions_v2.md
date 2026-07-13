# Updated instructions
Keep in mind that the first set of instructions are in the file "instructions.md".

# Data
The /home/cephandrius/Desktop/git/Lexi-BU/lexi_data_analysis/data/downloaded_themis_c_data/ has the themis_c data in cdf format. These are standard CDF files from NASA that are ISTP compliant!

## Moments data: thc_l2s_mom_20250316193000_20250316211459_cdaweb.cdf
    -  Relevant keys: 
        - thc_peem_epoch: Epoch of observation (Unit: Datetime in UTC)
        - thc_peem_flux: Electron particle flux vector (Unit: particles / (cm^2 * s))
        - thc_peim_flux: Ion particle flux vector (Unit: particles / (cm^2 * s))
        - thc_peem_velocity_gse: Electron velocity vector in GSE coordinates (Unit: km/s)
        - thc_peim_velocity_gse: Ion velocity vector in GSE coordinates (Unit: km/s)

## Magnetic Field Data: thc_l2s_fgm_20250316193001_20250316211457_cdaweb.cdf
    - Relevant keys:
        - thc_fgs_epoch: Epoch of observation (Unit: Datetime in UTC)
        - thc_fgs_gse: Magnetic field vector in GSE coordinates (Unit: nT, in GSE coordinates)

## Position Data: thc_or_ssc_20250316193000_20250316211500_cdaweb.cdf
    - Relevant keys:
        - Epoch: Epoch of observation (Unit: Datetime in UTC)
        - XYZ_GSE: Position vector in GSE coordinates (Unit: Earth radii, in GSE coordinates)


Units: (IMPORTANT)
Distance: Earth radius (convert it to km by multiplying it with the Earth radius)
Velocity: km/sec
Density: cm^(-3)
Magnetic field: nT

Based on these informations, modify the file /home/cephandrius/Desktop/git/Lexi-BU/lexi_data_analysis/data_analysis/planar_propagation_analysis.py

# Instructions:
1. Use the magnetic field data from here to compute the propagation time using MVA.
2. Use the ion or electron velocity to compute the ballistic propagation time.
3. Remember to convert units as required.
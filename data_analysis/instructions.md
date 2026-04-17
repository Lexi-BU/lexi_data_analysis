# NOTE: DO NOT modify any existing python files. Create new files or folders as needed.

# Background Details:
1. LEXI data file is"../data/line_profile_data/bg_corrected/from_l2/line_profile_fit_parameters_bg_corrected_1min.csv"
2. The key for the LEXI count data is "background_corrected_total_hist_counts".
3. LEXI spacecraft position data is in the file
   "../data/lexi_themis_analysis/lexi_themis_c_analysis_lexi_spacecraft.csv" with following keys:
   lexi_sc_pos_gse_x, lexi_sc_pos_gse_y, lexi_sc_pos_gse_z

# TO DO:
1. Get the THEMIS-C data, including magnetic field and plasma data starting from 2025-03-16 19:30:00
   to 2025-03-16 21:15:00. Use the file "get_themis_electron_params.py" as a sample to create a program to
   download the relevant data.
2. For themis data, use the THC_L2_MOM data for density as well as computing flux from the vector.
2. If Themis-C data already exists, read the data from the file instead of downloading it again.
3. Get the LEXI count data using the provided CSV file, focusing on the
   "background_corrected_total_hist_counts" key. Convert it to counts per second by dividing by 60.

# Planar propagation
1. Assume that the solar wind has planar propagation.
2. Using the THEMIS-C position, and the value of the magnetic field and plasma data, determine the
   propagation delay (between the solar wind observed at THEMIS-C and the corresponding response observed at LEXI) using the mean variance analysis.
3. This delay will be in and around the ballistic propagation delay (distance between THEMIS-C and LEXI divided by the solar wind speed).

# Plotting
1. Plot the LEXI count data (original and aligned) and THEMIS-C flux data.
2. In the aligned LEXI count data, color the points based on the delay determined from the mean variance
   analysis, to visualize how the delay changes over time.
3. Do not display the figure. Close it after saving it to the correct folder (../figures/lexi_themis_c_analysis/planar_propagation/).
4. For the x-axis, only plot it between the time_range defined above.
5. The lexi data is at a cadence of 1-minute. Please preserve that cadence.
6. Make all the plots scatter plot.
7. add a separate subplot to display the time delay.
# b_value_japan

With this repository, you can recreate all the results of the article.

Software needed:
- Python>=3.10
- Python packages: rft1d, seismostats

Files needed:
The only additional file that is needed is an earthquake catalog of japan. This catalog should be a cvs with the following collumns (case sensitive): time, latitude, longitude, depth, magnitude, event_type.
The csv file is read as dataframe (df). Some notes:
- time: includes data and time, has to be such that this works: pd.to_datetime(df["time"], format="mixed")
- event_type: only the earthquakes that are 'earthquake' are considered
- latitude/ longitude: in degree with decimal points (not minutes/seconds)
- depth: in km

Recreate:
Run the following scripts, in order. They are designed to be run with a slurm workload manager. The way to run each script is commented at the top of each document.
- 0_parameters.py (set the parameters that will be used later) 
- 1_prepare_catalogs.py (filter the catalogs to the given buffer)
- 2_map.py (estimate maps with different length scales)
- 2_map_full.py (for a chosen length-scale, estimate the full grid of b-values)
- 2a_synthetic_shuffle.py & 3a_synthetic_shuffle_retain_depth (only needed to recreate suynthetic tests)
- 3_main_plots (jupyternotebook that recreates all plots shown in the article)
- 3_robustness_analysis (jupyternotebook that recreates the plots of the robustness analysis in the supplement)
- 3_synthetic_analysis (jupyternotebook that recreates the plots of the synthetixc tests in the supplement)


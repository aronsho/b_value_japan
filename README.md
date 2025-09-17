# b_value_japan

With this repository, you can recreate all the results of the article.

Software needed:
- Python>=3.10
- Python packages: 

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


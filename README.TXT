The following explains, plot-by-plot, how to reproduce the figures presented in the paper.

Figure 1
a. 'plot_1a.py' takes the data from 'fig1a_data.csv' and plots it as '1a_fig.svg'. 
e. The file 'dynamics.py' contains a section titled '# Growth lag tradeoff with inactive ribosomes' which produces the plot, 'Growth-lag_ribosomes.svg'
f. The file 'dynamics.py' contains a section titled '# Growth lag tradeoff with inactive enzymes' which produces the plot, 'Growth-lag_enzymes.svg'

Figure 2a,b and Figure 3a,b
The file dynamics.py contains a section titled '# This code simulates actual dynamics' wherein dynamics of either population abundance or nutrient concentrations can be obtained by changing the plotting to 'pop' or 'res' on line 246. On line 193, set c['r'] = 1 for Figure 2 and c['r'] = 100 for Figure 3.

Figure 2c,d and Figure 3c,d
The files 'invasibility_community_root_1.py' and 'invasibility_community_root_100.py' in the folder 'figure2_community' produce the plots for Figures 2 and 3 respectively. 'community_root_fr_1.svg' and 'community_root_phi_1.svg' correspond to 2c and 2d, while 'community_root_fr_100.svg' and 'community_root_phi_100.svg' correspond to 3c and 3d.

Figure 2e,f and Figure 3e,f
The file 'pairwise_invasion_final.py' produce the pairwise invasibility plots. On line 105, set c['r'] = 1 for Figure 2 and c['r'] = 100 for Figure 3. 'Invasionplot_f_r_1.svg' and 'Invasionplot_phi_R0max_1.svg' correspond to 2e and 2f, while 'Invasionplot_f_r_100.svg' and 'Invasionplot_phi_R0max_100.svg' correspond to 3e and 3f.

Figure 2g
In the folder '02-mutation-figure', 'evolutionary_dynamics_trajectories.py' runs stepwise mutation analyses and saves the outputs to 'trait_trajectory_{i}.csv'. The file called 'adaptive_dynamics_trajectories.py' in the outer folder produces the prediction by adaptive dynamics, which is copied into this folder as 'adaptive_dynamics_trajectory.csv'. 'trajectoryplotter.py' produces 'Phase_space_trajectories.svg' which is Fig 2g.

Figure 3g
The same description as above, but in the folder '02-mutation-figure-2'. Here, the file 'traj_debug.py' plots single trajectories rather than all of them, and the output is 'fig3_f.svg'.

Figure 4a,b
The folder '02-heatmap-species' contains the code for Figure 4. The pipeline is as follows:
1. 'invasibility_community_root.py' runs simulations of community dynamics with an array of strains, and writes to 'final_{i}.csv'
2. To ensure dynamics are run to completion, 'cleanup.py' picks the endpoints of these simulations, excludes low-abundance strains and runs dynamics till steady-state.
3. 'numerical_ESS_invasion.py' numerically finds the Evolutionarily Stable Strategy (ESS) for a given environment, and invades the final community with this ESS point. This ensures that the final strain is closest to 'optimal' as possible.
4. 'plot_heatmap.py' plots the final results of this exercise as two heatmaps - these are 'fr_combined_heatmap.svg' (Fig 4a) and 'phi_R0max_combined_heatmap.svg' (Fig 4b).

Figure 5b
In the folder 'last_figure_new', 'evolutionary_dynamics_trajectories.py' runs the analysis and produces evolved strains. These are plotted along with the data for Fig 1a by 'bothinonefig.py'.

Figure S1
The file 'dynamics.py' has a section titled '# Inactive fraction of ribosomes' that produces the output 'inact_fraction.pdf'.

Figure S2
This is the output 'Heatmap_branching.svg' of 'trajectoryplotter.py' in the folder '02-mutation-figure-2'.

Figure S3
In the folder '02-heatmap-species', 'plot_heatmap.py' contains a section that plots the final strains from Figure 4 into 'physiology.svg'.
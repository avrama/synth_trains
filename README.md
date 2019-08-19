# synth_trains
creation of synthetic spike trains
1. spike_trains.py 

  - creates uncorrelated spike trains using several statistical distributions
  - will create trains for every cell type specified in the cell_type_dictionary
  - num_cells: number of trains
  - mean_isi: mean inter-spike interval
  - interburst,intraburst: parameters for creating bursty trains
  - freq_dependence: between 0 and 1, allows for sinusoidal modulation of mean frequency
  
2. corr_trains

  - creates a single set of spike trains with user specified distribution and correlation
  - command line specification of mean frequency, statistical distribution, correlation type and value
  
3. spike_train_utils:

  - functions used by spike_trains and corr_trains

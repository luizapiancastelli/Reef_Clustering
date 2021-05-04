# Reef_Clustering: model-based clustering proportion data of reef species abundances

This reposity contains code to fit the Dirichlet mixture model for clustering compositional data introduced by Piancastelli, Friel, Vercelloni, Mengersen and Mira (preprint ArXiv: ) via MCMC.

## Data sets

The data sets in this work are made available in the \Data folder. It consists of the proportions of four species (Algae, Hard Corals, Soft Corals and Sand) at various reef locations of the Great Barrier Reef in 2012, 2014, 2016 and 2017. License These data are shared under the Creative Commons license “Attribution-NonCommercialShareAlike 4.0 International”. A condition of the use of this data is that it is appropriately cited as given below, attributed and derived datasets are shared under similar terms. 

González-Rivero, Manuel, Rodriguez-Ramirez, Alberto, Beijbom, Oscar, Dalton, Peter, Kennedy, Emma V., Neal, Benjamin P., Vercelloni, Julie, Bongaerts, Pim, Ganase, Anjani, Bryant, Dominic E.P., Brown, Kristen, Kim, Catherine, Radice, Veronica Z., Lopez-Marcano, Sebastian, Dove, Sophie, Bailhache, Christophe, Beyer, Hawthorne L., and Hoegh-Guldberg, Ove(2019). Seaview Survey Photo-quadrat and Image Classification Dataset. The University of Queensland. Data Collection.

## Code

## Simulating data and running MCMC
High performance code for data simulation and MCMC are implemented in Python, making use of the `numba` library that translates Python functions to optimized machine code (\url{http://numba.pydata.org/}). A step by step description with simulated data example is given in the Jupyter notebook `Simulation_and_MCMC.ipynb`. The main functions are those contained in `Main.py`.

## Post-processing 
We handle possible label-switching and visualization in `R` using the libraries `label.switching` and `ggplot2`. Some basic summaries are contained in `chain_processing.R` with guidelines on the `.Rmd` file `post_process.Rmd`.


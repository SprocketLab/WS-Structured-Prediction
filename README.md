# Experiment code for Lifting Weak Supervision To Structured Prediction

## Finite metric spaces: learning to rank
The script that reproduces our ranking experiments can be found at `code/run.sh`. This script calls `main.py` with various parameter sweeps and seeds. The resulting logs can be processed into plots using `notebooks/plots_ranking.ipynb`. 

## Riemannian manifolds: Hyperbolic regression
The Hyperbolic regression experiments can be reproduced by running the notebook `notebooks/hyperbolic_ws_sp.ipynb`. 

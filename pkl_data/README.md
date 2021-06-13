DOI: 10.5281/zenodo.3970145

# (OLD) Generating and plotting of figures for "Learning as filtering" manuscript.

  To plot, provide one of the following figure keys (str) as -f argument:
      fig1d, fig2a, fig2b, fig2c, fig2d, fig2e, fig3, fig4,
      figS1, figS2, figS3, figS4, figS5

  To generate new figure-data, use the production keys (str) as -p arguments:
      fig1d, fig2_dim, fig2_beta, fig2_dim_pf, fig2_beta_pf, fig2_eta,
      fig2d, fig2e, fig3, fig4, figS4, figS5

# Command line example for generating fig1d data and plot:
    python main.py -f fig1d -p fig1d

  The plot is saved as ./figures/fig1d.pdf
  The data (a pandas data frame) is stored as ./pkl_data/fig1d/fig1d.pkl

# Further details:
    This file contains 2 simulation environments, one for the biological &
    one for the performance oriented simulations. Parameters are set in
    three layers. Lower layers have priority.
    1. default parameters apply to all simulations
    2. simulation type parameters apply either to bio- or performance sims
    3. for each figure, specpfic parameters can be selected

    Plotting parameters (labels, line color ect) must be tuned directly in
    the function "plt_manuscript_figures" in the file "./util/util.py"

# Simple Binning Search Program

This code has been adapted from Keith Bechtol's original simple binning code program to be more modular and ready-to-use for future searches.

[Sidney Mau](https://github.com/SidneyMau)

## Configuration and use

The `config.yaml` file will point the different parts of the code to the data set over which the search will be performed and the directories where the results will be saved. Once this has been set, then `farm_simple.py` will use `search_algorithm` to perform the simple binnning search over the given data set, storing the results in `results_dir/`. Then, `make_list.py` will compile the data stored in `results_dir/` into a candidate list, `results.csv`. Finally, running `render_plots.py` will use `diagnostic_plots.py` to create diagnostic plots for each candidate with sigma > 5.5 and will save these plots in `save_dir/`.

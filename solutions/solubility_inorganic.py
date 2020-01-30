import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

DF = pd.read_csv(
    '../data/aqueous_solubility_inorganic_temperatures.csv', index_col=0)

FORMULAS = DF['Formula']

TEMPERATURES = [int(i) for i in list(DF.columns[1:])]


def compounds_indexes(compounds_list):
    """Returns the indexes of the compounds formulas passed as a list
    of strings

    Parameters
    ----------
    compounds_list : list of strings
        List of compounds formulas (strings)

    Returns
    -------
    array
        Array with indexes

    Raises
    ------
    ValueError
        Not a valid formula (not in the dataframe)
    """
    if any(FORMULAS.isin(compounds_list)):
        idx_formulas = DF.index[FORMULAS.isin(compounds_list)].values
    else:
        raise ValueError('Input not valid')

    return idx_formulas


def _plot_params(plot_size=(10, 8)):
    """Plot parameters.

    Parameters
    ----------
    plot_size : tuple, optional
        Figure size, by default (10, 8)

    Returns
    -------
    Matplolib objects
        Figure and axis
    """
    fig, axarr = plt.subplots(figsize=plot_size, nrows=1, ncols=1)
    ax = axarr

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(1.5)

    ax.grid(b=True, axis='both', which='major',
            linestyle='--', linewidth=2.0, zorder=-1)
    ax.minorticks_on()
    ax.grid(b=True, which='minor', axis='both', linestyle=':', linewidth=1.0)
    ax.tick_params(axis='both', labelsize=16,
                   length=6, which='major', width=1.5)
    ax.set_xlabel('Temperature / °C', size=18)
    ax.set_ylabel('Solubility in mass percentage / %)', size=18)
    ax.set_title('Aqueous solubility in mass percentage', size=18)

    return fig, ax


def plot(compounds_list, colors=plt.cm.Dark2, interpolation=True,
         plot_size=(10, 8)):
    """Plot of the data

    Parameters
    ----------
    compounds_list : list of strings
        List of salts formulas (strings)
    colors : Matplotlib colormap, optional
        colormap, by default plt.cm.Dark2
    interpolation : bool, optional
        If a curve build with interpolation must be plotted, by default True
        If less than 5 five data points are available, a linear interpolation
        will be done. Otherwise, a cubic one will be plotted.
    plot_size : tuple, optional
        Figure size, by default (10, 8)
    """
    _plot_params(plot_size=plot_size)
    idx = compounds_indexes(compounds_list)

    ax = plt.gca()
    colormap = colors
    ax.set_prop_cycle(plt.cycler(
        'color', colormap(np.linspace(0, 1, len(idx)))))

    # possible markers
    markers = []
    for m in Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass

    # ignores point and pixel markers
    for index, mark in zip(idx, markers[2:]):

        # displaying chemical formulas in a proper way
        label_formula = re.sub("([0-9])", "_\\1", DF.iloc[index, 0])
        label_formula = '$\mathregular{'+label_formula+'}$'

        ax.scatter(TEMPERATURES,
                   DF.iloc[index, 1:].to_list(),
                   marker=mark,
                   s=100,
                   zorder=2,
                   label=label_formula)

        if interpolation:
            y = DF.iloc[index, 1:].dropna().values
            x = [int(i) for i in DF.iloc[index, 1:].dropna().index]

            if len(y) < 2:
                ax.plot(x, y)  # to maintain the lines and markers colors
            elif len(y) < 5:
                f = interp1d(x, y, kind='linear')
                temp_new = np.arange(min(x), max(x), 0.1)
                ax.plot(temp_new, f(temp_new))
            else:
                f = interp1d(x, y, kind='cubic')
                temp_new = np.linspace(min(x), max(x), num=100)
                ax.plot(temp_new, f(temp_new))

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    plt.show()

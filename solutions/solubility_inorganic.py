import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d
from chempy import Substance
import pathlib
import time

DF = pd.read_csv(
    '../data/aqueous_solubility_inorganic_temperatures.csv', index_col=0)

DF_PHASE_CHANGE = pd.read_csv(
    '../data/aqueous_solubility_inorganic_temperatures_phase_changes_STRINGS.csv', index_col=0)

FORMULAS = DF['Formula']

TEMPERATURES = [int(i) for i in list(DF.columns[1:])]

M_water = 18.015  # g/mol

_ANIONS = {'acetate': 'C2H3O2',
           'arsenate': '[^H][\d]AsO4',
           'hydrogen arsenate': 'HAsO4',
           'dihydrogen arsenate': 'H2AsO4',
           'bicarbonate': 'HCO3',
           'bromide': 'Br\d?$',
           'bromate': 'BrO3',
           'carbonate': '[^H]CO3',
           'chloride': 'Cl\d?$',
           'hypochlorite': 'ClO$',
           'chlorite': 'ClO2',
           'chlorate': 'ClO3',
           'perchlorate': 'ClO4',
           'chromate': 'CrO4',
           'dichromate': 'Cr2O7',
           'cyanide': '[^Fe][^S]CN',
           'fluoride': '[^B]F\d?$',
           'ferrocyanide': '[4]Fe\(CN\)6',
           'ferricyanide': '[3]Fe\(CN\)6',
           'formate': 'CHO2',
           'hydroxide': 'OH',
           'iodide': 'I\d?$',
           'iodate': 'IO3',
           'periodate': 'IO4',
           'nitrite': 'NO2',
           'nitrate': 'NO3',
           'oxide': '(?<!B|C|S|N|P|I|H|W)(?<!C[0-9]|S[0-9]|H[0-9]|B[0-9]|P[0-9])(?<!As|Cl|Se|Br|Mo|Mn)(?<![0-9a-z]Cr)(?<![0-9a-z]Cr\d)O\d?$',
           'oxalate': 'C2O4',
           'permanganate': 'MnO4',
           'molybdate': 'MoO4',
           'phosphate': '[^H][\d]PO4',
           'hydrogenphosphate': 'HPO4',
           'dihydrogenphosphate': 'H2PO4',
           'phosphite': '[^H]PO3',
           'hydrogenphosphite': 'HPO3',
           'pyrophosphate': 'P2O7',
           'selenite': 'SeO3',
           'selenate': 'SeO4',
           'sulfide': 'S\d?$',
           'sulfite': '[^H]SO3',
           'sulfate': '[^H]SO4',
           'hydrogen sulfate': 'HSO4',
           'borate': 'BO3',
           'tetrafluoroborate': 'BF4',
           'tetraborate': 'B4O7',
           'thiocyanate': 'SCN',
           'thiosulfate': 'S2O3',
           'disulfite': 'S2O5',
           'peroxydisulfate': 'S2O8',
           'tungstate': 'WO4',
           }

_GROUPS = {'group1': ['Li', 'Na', 'K', 'Rb', 'Cs', 'Fr'],
           'group2': ['Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'],
           'transition_3d': ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn'],
           'transition_4d': ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd'],
           'transition_5d': ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'],
           'lanthanides': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
           'actinides': ['Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'],
           'halides': [_ANIONS[key] for key in ['bromide', 'chloride', 'fluoride', 'iodide']],
           }


def show_filters():
    anions_filters = list(_ANIONS.keys())
    groups_filters = list(_GROUPS.keys())

    print('Anions filters: ', anions_filters)
    print()
    print('Groups filters:', groups_filters)


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


def molar_mass_solute(idx_formula):
    """Calculates the molar mass for a chemical formula in a dataframe.

    Parameters
    ----------
    idx_formula : int
        Integer representing an index in a dataframe

    Returns
    -------
    float
        Molar mass for a chemical formula in a dataframe
    """
    return Substance.from_formula(DF.iloc[idx_formula, 0]).mass


def conversion(idx_formula, unit='mass percentage'):
    """Conversion function. Converts the mass percentage from the database
    to other units.

    Parameters
    ----------
    idx_formula : int
        Integer representing an index in a dataframe
    unit : str, optional
        Desired unit, by default 'mass percentage'
        'solubility' means solubility in grams of solute per 100 g of water
        'molality' and 'mole percentage' are self explanatory.

    Returns
    -------
    pandas series
        Converted values.

    Raises
    ------
    ValueError
        Error raised when the unit passed is not valid.
    """
    data_mass_percentage = DF.iloc[idx_formula, 1:]

    if unit == 'mass percentage':
        result = data_mass_percentage
    elif unit == 'solubility':
        result = data_mass_percentage / (1 - data_mass_percentage / 100)
    elif unit == 'molality':
        result = (10 * data_mass_percentage) / \
            (molar_mass_solute(idx_formula) * (1 - data_mass_percentage / 100))
    elif unit == 'mole percentage':
        result = (data_mass_percentage/100 / molar_mass_solute(idx_formula)) / \
            ((data_mass_percentage / 100 / molar_mass_solute(idx_formula)) +
                (1 - data_mass_percentage / 100) / M_water) * 100
    else:
        raise ValueError('Unit not valid')

    return result.astype(float).round(decimals=3)


def df_subset(dataframe, mask, unit='mass percentage'):
    """Creates a subset of the dataframe based on a given mask.

    Parameters
    ----------
    dataframe : pandas dataframe
        dataframe
    mask : str
        mask to be applied
    unit : str, optional
        Desired unit, by default 'mass percentage'
        'solubility' means solubility in grams of solute per 100 g of water
        'molality' and 'mole percentage' are self explanatory.

    Returns
    -------
    pandas dataframe
        subset of the given dataframe
    """
    if mask in _ANIONS:
        df = dataframe[dataframe['Formula'].str.contains(_ANIONS[mask])]
    elif mask in _GROUPS:
        df = dataframe[dataframe['Formula'].str.contains(
            '|'.join(_GROUPS[mask]))]
    else:
        df = dataframe[dataframe['Formula'].str.contains(
            mask)]

    if unit == 'mass percentage':
        df = df
    else:
        for i in df.index:
            df[:].loc[i, '0':] = conversion(i, unit=unit).values

    return df


def _plot_params(plot_size=(10, 8), size=16, unit='mass percentage'):
    """Plot parameters.

    Parameters
    ----------
    plot_size : tuple, optional
        Figure size, by default (10, 8)
    size : int, optional
        Reference for labels sizes, by default 16
    unit : str, optional
        Desired unit, by default 'mass percentage'
        'solubility' means solubility in grams of solute per 100 g of water
        'molality' and 'mole percentage' are self explanatory.

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
    ax.tick_params(axis='both', labelsize=size,
                   length=6, which='major', width=1.5)
    ax.set_xlabel('Temperature / °C', size=size+2)

    if unit == 'mass percentage':
        ax.set_ylabel('Solubility in mass percentage / %', size=size+2)
        ax.set_title('Aqueous solubility in mass percentage', size=size+2)

    elif unit == 'solubility':
        ax.set_ylabel(
            'Solubility in grams of solute per 100 g of water', size=size+2)
        ax.set_title(
            'Aqueous solubility in grams of solute per 100 g of water', size=size+2)

    elif unit == 'molality':
        ax.set_ylabel('Solubility in molality / (mol/kg water)', size=size+2)
        ax.set_title('Aqueous solubility in molality', size=size+2)

    elif unit == 'mole percentage':
        ax.set_ylabel(
            'Solubility in mole percentage / %', size=size+2)
        ax.set_title('Aqueous solubility in mole percentage', size=size+2)

    else:
        raise ValueError('Unit not valid')

    return fig, ax


def plot(compounds_list, colors=plt.cm.Dark2, interpolation=False,
         plot_size=(10, 8), unit='mass percentage', save_fig=False):
    """Plot of the data

    Parameters
    ----------
    compounds_list : list of strings
        List of salts formulas (strings)
    colors : Matplotlib colormap, optional
        colormap, by default plt.cm.Dark2
    interpolation : bool, optional
        If a curve build with linear interpolation must be plotted, by default False
    plot_size : tuple, optional
        Figure size, by default (10, 8)
    unit : str, optional
        Desired unit, by default 'mass percentage'
        'solubility' means solubility in grams of solute per 100 g of water
        'molality' and 'mole percentage' are self explanatory.
    save_fig : bool, optional
        if the plot should be saved or not, by default False
    """
    _plot_params(plot_size=plot_size, unit=unit)
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
                   conversion(index, unit=unit).to_list(),
                   marker=mark,
                   s=100,
                   zorder=2,
                   label=label_formula)

        if interpolation:
            y = conversion(index, unit=unit).dropna().values
            x = [int(i) for i in conversion(index, unit=unit).dropna().index]

            if len(y) < 2:
                ax.plot(x, y)  # to maintain the lines and markers colors
            else:
                f = interp1d(x, y, kind='linear')
                temp_new = np.arange(min(x), max(x), 0.1)
                ax.plot(temp_new, f(temp_new))

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=14)

    if save_fig:
        pathlib.Path('saves').mkdir(parents=False, exist_ok=True)
        tme = time.localtime()
        time_string = time.strftime("%Y%m%d_%H:%M:%S", tme)
        plt.savefig('saves/{0}_{1}.png'.format(unit, time_string),
                    bbox_inches='tight')

    plt.show()

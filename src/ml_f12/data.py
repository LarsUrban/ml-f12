"""Data handling for the ML-F12 project.

All data are stored in a single table (pandas.DataFrame) with one row per
sample. A sample is one molecule in one orbital basis set.

Equation numbers in the comments refer to Section 3.3 of

    L. Urban, Formulation and implementation of accurate and efficient
    low-scaling methods for explicitly correlated F12 theory,
    PhD thesis, LMU Munich (2026).
    https://edoc.ub.uni-muenchen.de/36780/

All energies are in Hartree, given as signed contributions to the energy.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

# Metadata. Describe a sample but are not used as model input.
META_COLUMNS = [
    "molecule",
    "basis",
    "aux_basis",
    "cabs_basis",
    "n_atoms",
    "n_electrons",
    "e_hf",  # total HF energy, for bookkeeping only
    "gamma",  # geminal exponent
]

# Conventional MP2 correlation energy in the same basis
MP2_COLUMNS = [
    "e_mp2_direct",
    "e_mp2_exchange",
]

# Cheap F12 terms, all below O(M^4)
F12_TERM_COLUMNS = [
    "e_cabs_singles",  # CABS singles correction
    "e_v_fg_direct",  # Eq. 3.169, direct
    "e_v_fg_exchange",  # Eq. 3.169, exchange
    "e_v_fg3e_exchange",  # Eq. 3.172
    "e_x_ff_direct",  # Eq. 3.173, direct
    "e_x_ff_exchange",  # Eq. 3.173, exchange
    "e_x_ff3e_exchange",  # Eq. 3.176
    "e_b_comm_direct",  # Eq. 3.220
    "e_b_comm_exchange",  # Eq. 3.224
    "e_b_fk_direct",  # Eq. 3.221 (= 3.222)
    "e_b_fk_exchange",  # Eq. 3.225
    "e_b_k_exchange",  # Eq. 3.226
    "e_b_fo_exchange",  # Eq. 3.227
    "e_b_ofo_exchange",  # Eq. 3.228
]

# Target. Full MP2-F12 correction without CABS singles.
TARGET_COLUMN = "e_f12"

FEATURE_GROUPS: dict[str, list[str]] = {
    "mp2": MP2_COLUMNS,
    "f12": F12_TERM_COLUMNS,
}


def feature_columns(
    df: pd.DataFrame, groups: tuple[str, ...] = ("mp2", "f12", "ao")
) -> list[str]:
    """Return the feature column names for the selected groups, in fixed order."""
    columns: list[str] = []
    for group in groups:
        if group == "ao":
            names = [str(c) for c in df.columns]
            columns += sorted(n for n in names if n.startswith("ao_"))
        else:
            columns += FEATURE_GROUPS[group]
    return columns


def split_xy(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return the feature matrix X and the target vector y."""
    X = df[columns].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def load_fermions_dir(directory: str | Path) -> pd.DataFrame:
    """Read all FermiONs++ JSON files in a directory into one table."""
    rows = []
    for path in sorted(Path(directory).glob("*.json")):
        with open(path) as fh:
            rows.append(json.load(fh))
    return pd.DataFrame(rows)

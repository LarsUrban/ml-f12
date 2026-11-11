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

import numpy as np
import pandas as pd

# Metadata. Describe a sample but are not used as model input.
META_COLUMNS = [
    "molecule",
    "basis",
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
    "e_cabs_direct",  # CABS singles
    "e_cabs_exchange",
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

# ---------------------------------------------------------
# Adapted from:
# https://github.com/molML/MoleculeACE/blob/main/MoleculeACE/benchmark/cliffs.py
# Stripped down to Levenshtein-only similarity (no Tanimoto / scaffold / MMP).
# ---------------------------------------------------------

"""
Activity-cliff identification using normalized SMILES Levenshtein similarity.

    - ActivityCliffs:           Class that computes cliff compounds.
    - find_fc():                Fold change between two activities.
    - get_fc():                 Pairwise fold-change matrix.
    - get_levenshtein_matrix(): Pairwise normalized Levenshtein similarity.
    - levenshtein_similarity(): Binary similarity matrix at a threshold.
    - is_cliff():               Two-molecule cliff check.

A cliff pair is one where SMILES-string Levenshtein similarity >= `similarity`
AND fold change in activity > `potency_fold`.
"""

from typing import List, Union
import numpy as np
from Levenshtein import distance as levenshtein
from tqdm import tqdm


class ActivityCliffs:
    """Activity cliff class that computes cliff compounds."""

    def __init__(self, smiles: List[str], bioactivity: Union[List[float], np.array]):
        self.smiles = smiles
        self.bioactivity = list(bioactivity) if type(bioactivity) is not list else bioactivity
        self.cliffs = None

    def find_cliffs(self, similarity: float = 0.9, potency_fold: float = 10):
        """Compute activity cliffs using Levenshtein similarity only.

        :param similarity: threshold on normalized SMILES Levenshtein similarity
        :param potency_fold: threshold on fold change in bioactivity
        :return: square binary matrix; entry (i, j) = 1 iff (i, j) is a cliff pair
        """
        sim = levenshtein_similarity(self.smiles, similarity)
        fc = (get_fc(self.bioactivity) > potency_fold).astype(int)
        self.cliffs = np.logical_and(sim == 1, fc == 1).astype(int)
        return self.cliffs

    def get_cliff_molecules(self, return_smiles: bool = True, **kwargs):
        """Return molecules that participate in at least one cliff pair.

        :param return_smiles: if True return SMILES strings, else a 0/1 list
        :param kwargs: forwarded to find_cliffs()
        """
        if self.cliffs is None:
            self.find_cliffs(**kwargs)

        if return_smiles:
            return [self.smiles[i] for i in np.where((sum(self.cliffs) > 0).astype(int))[0]]
        else:
            return list((sum(self.cliffs) > 0).astype(int))

    def __repr__(self):
        return "Activity cliffs"


def find_fc(a: float, b: float):
    """Fold change between two activities."""
    return max([a, b]) / min([a, b])


def get_fc(bioactivity: List[float]):
    """Pairwise fold-change matrix."""
    act_len = len(bioactivity)
    m = np.zeros([act_len, act_len])
    for i in range(act_len):
        for j in range(i, act_len):
            m[i, j] = find_fc(bioactivity[i], bioactivity[j])
    m = m + m.T - np.diag(np.diag(m))
    np.fill_diagonal(m, 0)
    return m


def get_levenshtein_matrix(smiles: List[str], normalize: bool = True,
                           hide: bool = False, top_n: int = None):
    """Pairwise SMILES Levenshtein similarity matrix.

    Similarity is `1 - lev(s_i, s_j) / max(|s_i|, |s_j|)` when normalize=True,
    otherwise `1 - lev(s_i, s_j)` (raw distance, kept for backwards compat with
    the original MoleculeACE signature).
    """
    smi_len = len(smiles)
    m = np.zeros([smi_len, smi_len])
    for i in tqdm(range(smi_len if top_n is None else top_n), disable=hide):
        for j in range(i, smi_len):
            if normalize:
                m[i, j] = levenshtein(smiles[i], smiles[j]) / max(len(smiles[i]), len(smiles[j]))
            else:
                m[i, j] = levenshtein(smiles[i], smiles[j])
    m = m + m.T - np.diag(np.diag(m))
    m = 1 - m
    np.fill_diagonal(m, 0)
    return m


def levenshtein_similarity(smiles: List[str], similarity: float = 0.9,
                           hide: bool = False):
    """Binary matrix: entry (i, j) = 1 iff Levenshtein similarity >= `similarity`."""
    return (get_levenshtein_matrix(smiles, hide=hide) >= similarity).astype(int)


def is_cliff(smiles1, smiles2, y1, y2, similarity: float = 0.9, potency_fold: float = 10):
    """Check whether two molecules form a cliff pair."""
    sim = levenshtein_similarity([smiles1, smiles2], similarity=similarity, hide=True)[0][1]
    fc = get_fc([y1, y2])[0][1]
    return sim == 1 and fc >= potency_fold

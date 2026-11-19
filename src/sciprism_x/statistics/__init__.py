"""Paired inference and multiple-testing correction."""

from .bootstrap import BootstrapResult, paired_bootstrap
from .multiple_testing import HolmResult, holm_bonferroni

__all__ = ["BootstrapResult", "HolmResult", "holm_bonferroni", "paired_bootstrap"]

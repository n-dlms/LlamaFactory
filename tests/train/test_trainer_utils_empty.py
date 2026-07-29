"""Test for empty dataset edge case in trainer_utils loss functions."""

import torch
import pytest
from llamafactory.train.trainer_utils import _dft_cross_entropy, _eaft_cross_entropy


class TestEmptyDataset:
    def test_dft_cross_entropy_all_ignore_index(self):
        """If all tokens are IGNORE_INDEX (-100), return 0.0 without division by zero."""
        source = torch.randn(2, 100, 10)  # batch=2, seq=100, vocab=10
        target = torch.full((2, 100), -100, dtype=torch.long)

        result = _dft_cross_entropy(source, target, ignore_index=-100)

        assert torch.is_tensor(result)
        assert result.item() == 0.0
        assert result.device == source.device
        assert result.dtype == source.dtype

    def test_eaft_cross_entropy_all_ignore_index(self):
        """Same guard for EAFT loss."""
        source = torch.randn(2, 100, 10)
        target = torch.full((2, 100), -100, dtype=torch.long)

        result = _eaft_cross_entropy(source, target, ignore_index=-100)

        assert result.item() == 0.0

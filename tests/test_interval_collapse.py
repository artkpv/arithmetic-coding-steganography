"""
Unit tests for interval collapse detection in arithmetic coding.
"""

import pytest

from stego_arith_coding.core import ArithmeticSteganography


class TestIntervalCollapse:
    """Test cases for detecting and handling interval collapse."""

    def test_narrow_interval_detection(self):
        """Test that narrow intervals are properly detected and raise errors."""
        stego = ArithmeticSteganography(model="gpt-3.5-turbo-1106")

        # Test various narrow intervals that should fail
        test_cases = [
            # (interval, expected_error_pattern)
            (
                [0, 2],
                "Interval too narrow",
            ),  # Range = 2, too narrow for multiple tokens
            ([100, 103], "Interval too narrow"),  # Range = 3
            (
                [2147483647, 2147483650],
                "Interval too narrow",
            ),  # Range = 3 (from actual logs)
        ]

        # Use a simple context
        context = "The weather is"

        for interval, error_pattern in test_cases:
            with pytest.raises(ValueError, match=error_pattern):
                stego._compute_interval_division(context, interval, precision=16)

    def test_wide_interval_success(self):
        """Test that sufficiently wide intervals work correctly."""
        stego = ArithmeticSteganography(model="gpt-3.5-turbo-1106")

        # Test cases with sufficiently wide intervals
        test_cases = [
            [0, 65536],  # 2^16, standard precision
            [0, 4294967296],  # 2^32
            [1000, 10000],  # Range = 9000
        ]

        context = "The weather is"

        for interval in test_cases:
            # Should not raise an error
            cum_probs, tokens = stego._compute_interval_division(
                context, interval, precision=16
            )

            # Basic validation
            assert len(tokens) > 0, f"No tokens returned for interval {interval}"
            assert len(cum_probs) == len(
                tokens
            ), "Mismatch between tokens and probabilities"
            assert (
                cum_probs[-1] == interval[1]
            ), "Last cumulative probability should equal interval top"

    def test_interval_bounds_validation(self):
        """Test that invalid intervals are rejected."""
        stego = ArithmeticSteganography(model="gpt-3.5-turbo-1106")

        context = "Test"

        # Test collapsed interval (top <= bottom)
        with pytest.raises(ValueError, match="Interval has collapsed"):
            stego._compute_interval_division(context, [100, 100], precision=16)

        # Test inverted interval
        with pytest.raises(ValueError, match="Interval has collapsed"):
            stego._compute_interval_division(context, [100, 50], precision=16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


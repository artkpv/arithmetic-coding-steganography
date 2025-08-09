import os
import unittest
from unittest.mock import MagicMock, patch

from stego_arith_coding import ArithmeticSteganography


class TestArithmeticSteganography(unittest.TestCase):
    """Tests for the ArithmeticSteganography class."""

    def setUp(self):
        """Set up test instance."""
        # Mock the OpenAI client to avoid API calls in tests
        with patch("stego_arith_coding.core.openai.OpenAI") as mock_openai:
            self.stego = ArithmeticSteganography(model="gpt-3.5-turbo")
            self.mock_client = mock_openai.return_value

    def test_text_to_bits(self):
        """Test text to bits conversion with EOF marker."""
        text = "AB"
        bits = self.stego.text_to_bits(text)

        # 'A' = 65 = 01000001, 'B' = 66 = 01000010, EOF = 0x03 = 00000011
        expected = [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        self.assertEqual(bits, expected)

    def test_bits_to_text(self):
        """Test bits to text conversion with EOF marker."""
        # 'A' = 65 = 01000001, 'B' = 66 = 01000010, EOF = 0x03 = 00000011
        bits = [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1]
        text = self.stego.bits_to_text(bits)

        expected = "AB"  # EOF marker should be removed
        self.assertEqual(text, expected)

    def test_text_bits_roundtrip(self):
        """Test roundtrip conversion: text -> bits -> text."""
        original_text = "Hello, World!"
        bits = self.stego.text_to_bits(original_text)
        recovered_text = self.stego.bits_to_text(bits)

        self.assertEqual(original_text, recovered_text)

    def test_etx_escaping(self):
        """Test that ETX characters are properly escaped and unescaped."""
        # Test text containing ETX character
        original_text = "Hello\x03World"  # Contains ETX character
        bits = self.stego.text_to_bits(original_text)
        recovered_text = self.stego.bits_to_text(bits)

        self.assertEqual(original_text, recovered_text)

        # Verify that the bits contain escaped ETX (two 0x03 sequences plus final EOF)
        # Expected: "Hello" + escaped ETX (0x03 0x03) + "World" + EOF (0x03)
        # That's: 5 chars + 2 bytes + 5 chars + 1 byte = 13 bytes = 104 bits
        expected_bit_length = (5 + 2 + 5 + 1) * 8  # 13 bytes * 8 bits/byte
        self.assertEqual(len(bits), expected_bit_length)

    def test_int2bits(self):
        """Test integer to bits conversion."""
        # Test basic conversion
        result = ArithmeticSteganography._int2bits(5, 4)  # 5 = 0101
        expected = [1, 0, 1, 0]  # LSB first
        self.assertEqual(result, expected)

        # Test with different precision
        result = ArithmeticSteganography._int2bits(10, 8)  # 10 = 00001010
        expected = [0, 1, 0, 1, 0, 0, 0, 0]  # LSB first
        self.assertEqual(result, expected)

        # Test zero
        result = ArithmeticSteganography._int2bits(0, 4)
        expected = [0, 0, 0, 0]
        self.assertEqual(result, expected)

        # Test edge case - precision 0
        result = ArithmeticSteganography._int2bits(5, 0)
        expected = []
        self.assertEqual(result, expected)

    def test_bits2int(self):
        """Test bits to integer conversion."""
        # Test basic conversion (LSB first)
        bits = [1, 0, 1, 0]  # 5 in LSB-first format
        result = ArithmeticSteganography._bits2int(bits)
        expected = 5
        self.assertEqual(result, expected)

        # Test with more bits
        bits = [0, 1, 0, 1, 0, 0, 0, 0]  # 10 in LSB-first format
        result = ArithmeticSteganography._bits2int(bits)
        expected = 10
        self.assertEqual(result, expected)

        # Test empty list
        result = ArithmeticSteganography._bits2int([])
        expected = 0
        self.assertEqual(result, expected)

    def test_int2bits_bits2int_roundtrip(self):
        """Test roundtrip conversion: int -> bits -> int."""
        for value in [0, 1, 5, 10, 255, 1024]:
            precision = max(8, value.bit_length())
            bits = ArithmeticSteganography._int2bits(value, precision)
            result = ArithmeticSteganography._bits2int(bits)
            self.assertEqual(result, value, f"Failed for value {value}")

    def test_num_same_from_beg(self):
        """Test counting same bits from beginning."""
        # Test identical lists
        bits1 = [1, 0, 1, 0]
        bits2 = [1, 0, 1, 0]
        result = ArithmeticSteganography._num_same_from_beg(bits1, bits2)
        expected = 4
        self.assertEqual(result, expected)

        # Test partial match
        bits1 = [1, 0, 1, 0]
        bits2 = [1, 0, 0, 1]
        result = ArithmeticSteganography._num_same_from_beg(bits1, bits2)
        expected = 2  # First two bits match
        self.assertEqual(result, expected)

        # Test no match
        bits1 = [1, 0, 1, 0]
        bits2 = [0, 1, 0, 1]
        result = ArithmeticSteganography._num_same_from_beg(bits1, bits2)
        expected = 0
        self.assertEqual(result, expected)

        # Test empty lists
        result = ArithmeticSteganography._num_same_from_beg([], [])
        expected = 0
        self.assertEqual(result, expected)

        # Test different lengths
        bits1 = [1, 0, 1]
        bits2 = [1, 0, 1, 0, 1]
        result = ArithmeticSteganography._num_same_from_beg(bits1, bits2)
        expected = 3
        self.assertEqual(result, expected)


class TestArithmeticSteganographyIntegration(unittest.TestCase):
    """Integration tests that require API access."""

    def setUp(self):
        """Set up test instance."""
        self.has_api_key = bool(os.getenv("OPENAI_API_KEY"))
        if self.has_api_key:
            self.stego = ArithmeticSteganography(model="gpt-3.5-turbo-1106")

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not available")
    def test_get_token_logprobs(self):
        """Test getting token logprobs from OpenAI API."""
        context = "The weather is"
        logprobs = self.stego._get_token_logprobs(context)

        # Basic sanity checks
        self.assertIsInstance(logprobs, dict)
        self.assertGreater(len(logprobs), 0)

        # Check that all values are floats (logprobs)
        for token, logprob in logprobs.items():
            self.assertIsInstance(token, str)
            self.assertIsInstance(logprob, float)
            # Logprobs should be negative (since they're log of probabilities < 1)
            self.assertLessEqual(logprob, 0)

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not available")
    def test_compute_interval_division(self):
        """Test interval division computation."""
        context = "The weather is"
        cur_interval = [0, 65536]  # 2^16

        cum_probs, tokens_temp = self.stego._compute_interval_division(
            context, cur_interval
        )

        # Basic sanity checks
        self.assertIsInstance(cum_probs, list)
        self.assertIsInstance(tokens_temp, list)
        self.assertEqual(len(cum_probs), len(tokens_temp))
        self.assertGreater(len(cum_probs), 0)

        # Check that cumulative probabilities are monotonically increasing
        for i in range(1, len(cum_probs)):
            self.assertGreaterEqual(cum_probs[i], cum_probs[i - 1])

        # Check that last cumulative probability equals interval range
        expected_range = cur_interval[1] - cur_interval[0]
        self.assertEqual(cum_probs[-1], cur_interval[0] + expected_range)

        # Check that all tokens are strings
        for token in tokens_temp:
            self.assertIsInstance(token, str)

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not available")
    def test_compute_interval_division_deterministic(self):
        """Test that interval division computation is deterministic."""
        context = "The weather is"
        cur_interval = [0, 65536]  # 2^16
        precision = 16

        # Call the method multiple times
        results = []
        for i in range(3):
            cum_probs, tokens_temp = self.stego._compute_interval_division(
                context, cur_interval, precision
            )
            results.append((cum_probs, tokens_temp))
            print(f"\nRun {i+1}:")
            print(f"  cum_probs: {cum_probs[:5]}...")  # Show first 5 values
            print(f"  tokens: {tokens_temp[:5]}...")  # Show first 5 tokens

        # Check that all results are identical
        for i in range(1, len(results)):
            cum_probs_1, tokens_1 = results[0]
            cum_probs_i, tokens_i = results[i]

            # Check cumulative probabilities match
            self.assertEqual(
                cum_probs_1,
                cum_probs_i,
                f"Cumulative probabilities differ between run 1 and run {i+1}",
            )

            # Check tokens match
            self.assertEqual(
                tokens_1, tokens_i, f"Tokens differ between run 1 and run {i+1}"
            )

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not available")
    def test_get_token_logprobs_deterministic(self):
        """Test that token logprobs are deterministic."""
        context = "The weather is"

        # Call the method multiple times
        results = []
        for i in range(3):
            logprobs = self.stego._get_token_logprobs(context)
            results.append(logprobs)
            print(f"\nRun {i+1}:")
            # Show top 3 tokens by probability
            sorted_tokens = sorted(logprobs.items(), key=lambda x: x[1], reverse=True)[
                :3
            ]
            for token, logprob in sorted_tokens:
                print(f"  {repr(token)}: {logprob}")

        # Check that all results are identical
        for i in range(1, len(results)):
            self.assertEqual(
                results[0],
                results[i],
                f"Token logprobs differ between run 1 and run {i+1}",
            )


class TestArithmeticSteganographyEncodeDecode(unittest.TestCase):
    """Integration tests that require API access."""

    def setUp(self):
        """Set up test instance."""
        self.has_api_key = bool(os.getenv("OPENAI_API_KEY"))
        if self.has_api_key:
            self.stego = ArithmeticSteganography(model="gpt-3.5-turbo-1106")

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not available")
    def test_encode_decode_roundtrip(self):
        """Test full encode->decode roundtrip using EOF marker."""
        # Test data - simple case
        message = "B5"
        message_bits = self.stego.text_to_bits(message)

        context = """Task: Write a simple function that checks if a number is prime.
Solution:
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
# Explanation: """

        precision = 32
        encoded_text = self.stego.encode(context, message_bits, precision=precision)
        self.assertGreater(len(encoded_text), 0, "Empty encoded result")

        # Test without message_length (using EOF marker detection)
        decoded_bits = self.stego.decode(context, encoded_text, precision=precision)
        
        # When encoding stops due to interval collapse, we may get partial EOF marker
        # The decoder returns what was actually encoded, which may include partial EOF bits
        # We should check if the decoded message (after bits_to_text) matches the original
        decoded_message = self.stego.bits_to_text(decoded_bits)

        # Accept exact match, or allow a trailing control char caused by partial EOF byte padding
        if decoded_message != message:
            self.assertTrue(
                decoded_message.startswith(message)
                and all(ord(c) < 32 for c in decoded_message[len(message):]),
                f"Decoded message has unexpected trailing data: {repr(decoded_message)}",
            )

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not available")
    def test_encode_decode_roundtrip_with_length(self):
        """Test full encode->decode roundtrip using message_length (legacy mode)."""
        # Test data - simple case
        message = "B5"
        message_bits = self.stego.text_to_bits(message)

        context = """Task: Write a simple function that checks if a number is prime.
Solution:
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True
# Explanation: """

        precision = 32
        encoded_text = self.stego.encode(context, message_bits, precision=precision)
        self.assertGreater(len(encoded_text), 0, "Empty encoded result")

        # Test with message_length (legacy compatibility)
        decoded_bits = self.stego.decode(
            context,
            encoded_text,
            precision=precision,
            message_length=len(message_bits),
        )

        decoded_message = self.stego.bits_to_text(decoded_bits)

        # Allow decoded bits to be truncated only within the EOF marker (partial EOF)
        if decoded_bits != message_bits:
            eof_marker = [0, 0, 0, 0, 0, 0, 1, 1]
            missing_bits = message_bits[len(decoded_bits):]
            self.assertTrue(
                len(decoded_bits) < len(message_bits)
                and message_bits[:len(decoded_bits)] == decoded_bits
                and missing_bits == eof_marker[-len(missing_bits):],
                "Decoded bits mismatch beyond allowable EOF truncation",
            )

        # Accept exact text match, or allow trailing control chars from partial EOF padding
        if decoded_message != message:
            self.assertTrue(
                decoded_message.startswith(message)
                and all(ord(c) < 32 for c in decoded_message[len(message):]),
                f"Decoded message has unexpected trailing data: {repr(decoded_message)}",
            )

    @unittest.skipUnless(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not available")
    def test_encode_decode_roundtrip_weather(self):
        """Test full encode->decode roundtrip with weather context using EOF marker."""
        # Test data - simple case
        message = "H"
        message_bits = self.stego.text_to_bits(message)

        context = "The weather today is "

        precision = 32
        encoded_text = self.stego.encode(context, message_bits, precision=precision)
        self.assertGreater(len(encoded_text), 0, "Empty encoded result")

        # Test without message_length (using EOF marker detection)
        decoded_bits = self.stego.decode(context, encoded_text, precision=precision)
        
        # When encoding stops due to interval collapse, we may get partial EOF marker
        # The bits_to_text function should handle this and return the original message
        decoded_message = self.stego.bits_to_text(decoded_bits)

        # Accept exact match, or allow a trailing control char caused by partial EOF byte padding
        if decoded_message != message:
            self.assertTrue(
                decoded_message.startswith(message)
                and all(ord(c) < 32 for c in decoded_message[len(message):]),
                f"Decoded message has unexpected trailing data: {repr(decoded_message)}",
            )


if __name__ == "__main__":
    unittest.main()

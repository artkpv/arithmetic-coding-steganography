#!/usr/bin/env python3
"""Test bit conversion to understand the issue."""

import unittest
from stego_arith_coding.core import ArithmeticSteganography


class TestBitConversion(unittest.TestCase):
    """Test bit conversion functions."""
    
    def setUp(self):
        self.stego = ArithmeticSteganography()
    
    def test_bits2int_int2bits(self):
        """Test bit to int conversion and back."""
        # Test with some values
        test_values = [0, 1, 42, 255, 1024, 65535]
        
        for val in test_values:
            bits = self.stego._int2bits(val, 16)
            reconstructed = self.stego._bits2int(bits)
            self.assertEqual(val, reconstructed, f"Failed for value {val}")
            
    def test_message_bits_interpretation(self):
        """Test how message bits are interpreted."""
        # Test message "B5"
        # 'B' = 0x42 = 66 = 01000010
        # '5' = 0x35 = 53 = 00110101
        message_bits = [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1]
        
        # Test with precision 32
        precision = 32
        i = 0
        
        # Get first window of bits
        message_window = message_bits[i:i+precision]
        if i+precision > len(message_bits):
            message_window = message_window + [0]*(i+precision-len(message_bits))
        
        print(f"\nFirst window (padded to {precision} bits):")
        print(f"MSB first: {message_window}")
        print(f"As string: {''.join(map(str, message_window))}")
        
        # Convert as in reference (reverse for LSB-first)
        message_idx_ref = self.stego._bits2int(list(reversed(message_window)))
        print(f"\nReference approach (bits2int(reversed)):")
        print(f"Value: {message_idx_ref}")
        print(f"As fraction of 2^32: {message_idx_ref / (2**32):.6f}")
        
        # Direct MSB interpretation
        message_value = 0
        for bit_idx, bit in enumerate(message_window):
            message_value += bit * (2 ** (precision - 1 - bit_idx))
        print(f"\nDirect MSB interpretation:")
        print(f"Value: {message_value}")  
        print(f"As fraction of 2^32: {message_value / (2**32):.6f}")
        
        # They should be the same!
        self.assertEqual(message_idx_ref, message_value, "Two approaches should give same result")
        
    def test_text_to_bits_format(self):
        """Test the format of text_to_bits output with EOF marker."""
        text = "B5"
        bits = self.stego.text_to_bits(text)
        
        print(f"\ntext_to_bits('{text}') = {bits}")
        print(f"Length: {len(bits)}")
        print(f"As string: {''.join(map(str, bits))}")
        
        # Check individual characters
        print("\nCharacter breakdown:")
        print(f"'B' = {ord('B')} = 0x{ord('B'):02x} = {format(ord('B'), '08b')}")
        print(f"'5' = {ord('5')} = 0x{ord('5'):02x} = {format(ord('5'), '08b')}")
        print(f"EOF = 0x03 = {format(0x03, '08b')}")
        
        # Verify the bits match including EOF marker
        expected_bits = []
        for char in text:
            char_bits = format(ord(char), '08b')
            expected_bits.extend([int(bit) for bit in char_bits])
        # Add EOF marker
        expected_bits.extend([0, 0, 0, 0, 0, 0, 1, 1])  # 0x03
        
        self.assertEqual(bits, expected_bits, "text_to_bits should produce MSB-first bits with EOF marker")


if __name__ == "__main__":
    unittest.main()
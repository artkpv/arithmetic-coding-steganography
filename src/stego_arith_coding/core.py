"""
Core arithmetic steganography implementation.

This module contains the main ArithmeticSteganography class that handles
encoding and decoding of secret messages using arithmetic coding.
"""

import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import openai
import tiktoken

from .config import ArithmeticSteganographyConfig


class ArithmeticSteganography:
    """Main class for arithmetic coding steganography operations."""

    def __init__(
        self,
        model: str = "gpt-3.5-turbo-1106",
        openai_api_key: Optional[str] = None,
        config: Optional[ArithmeticSteganographyConfig] = None,
    ):
        """Initialize the arithmetic steganography system.

        Args:
            model: The OpenAI model to use for token probability generation
            openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable
            config: Configuration object. If None, uses default configuration with provided model
        """
        # Initialize config
        if config is None:
            self.config = ArithmeticSteganographyConfig(model=model)
        else:
            self.config = config
            # Override model if explicitly provided (not using default)
            if model != "gpt-3.5-turbo-1106":
                self.config.model = model
        
        # Set up OpenAI client and tokenizer based on config
        self.model = self.config.model
        self._openai_api_key = openai_api_key
        self.openai_client = None
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.cached_system_fingerprint = None
        
        # Use cache directory from config
        self.cache_dir = self.config.cache_dir
        
        # Set up logger
        self.logger = logging.getLogger("stego_arith_coding.core")

    def _generate_cache_key(self, **params) -> str:
        """Generate a cache key based on request parameters."""
        # Create a hash from the serialized parameters
        cache_str = json.dumps(params, sort_keys=True)
        cache_hash = hashlib.sha256(cache_str.encode()).hexdigest()
        return cache_hash

    def _load_from_cache(self, cache_key: str) -> Optional[Dict[str, float]]:
        """Load cached response if it exists."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    self.logger.debug(f"Loaded from cache: {cache_key}")
                    return cached_data['token_logprobs']
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Error loading cache {cache_key}: {e}")
        return None

    def _save_to_cache(self, cache_key: str, token_logprobs: Dict[str, float], system_fingerprint: str):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_data = {
            "token_logprobs": token_logprobs,
            "system_fingerprint": system_fingerprint,
            "cached_at": datetime.datetime.utcnow().isoformat() + "Z"
        }
        try:
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            self.logger.debug(f"Saved to cache: {cache_key}")
        except Exception as e:
            self.logger.warning(f"Error saving cache {cache_key}: {e}")

    def _get_token_logprobs(self, context: str) -> Dict[str, float]:
        """Get logprobs of next tokens for the given context.

        Args:
            context: The text context to get next token logprobs for

        Returns:
            Dictionary containing token logprobs as key-value pairs
        """
        # Prepare API call parameters using config
        user_message = f"Continue this text:\n\n{context}"
        api_params = self.config.get_api_params_with_model()
        api_params["messages"] = [
            {
                "role": "user",
                "content": user_message,
            }
        ]
        
        # Generate cache key from API parameters
        cache_key = self._generate_cache_key(**api_params)
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            return cached_result

        max_retries = self.config.max_retries

        if self.openai_client is None:
            self.openai_client = openai.OpenAI(
                api_key=self._openai_api_key or os.getenv("OPENAI_API_KEY")
            )

        for attempt in range(max_retries):
            try:
                self.logger.debug(f'OpenAI API call: {user_message=}')
                response = self.openai_client.chat.completions.create(**api_params)

                # Check and cache system fingerprint for consistency
                if self.cached_system_fingerprint is None:
                    self.cached_system_fingerprint = response.system_fingerprint
                    self.logger.debug(
                        f"Cached system fingerprint: {self.cached_system_fingerprint}"
                    )
                elif self.cached_system_fingerprint != response.system_fingerprint:
                    self.logger.warning(
                        f"System fingerprint mismatch on attempt {attempt + 1}! Expected: {self.cached_system_fingerprint}, Got: {response.system_fingerprint}"
                    )
                    if attempt < max_retries - 1:
                        self.logger.info(
                            f"Retrying API call (attempt {attempt + 2}/{max_retries})"
                        )
                        continue
                    else:
                        raise ValueError(
                            f"System fingerprint mismatch after {max_retries} attempts. Expected: {self.cached_system_fingerprint}, Got: {response.system_fingerprint}"
                        )

                if (
                    response.choices[0].logprobs
                    and response.choices[0].logprobs.content
                    and response.choices[0].logprobs.content[0].top_logprobs
                ):
                    token_logprobs = {
                        tlp.token: tlp.logprob
                        for tlp in response.choices[0].logprobs.content[0].top_logprobs
                    }
                    # Save successful response to cache
                    self._save_to_cache(cache_key, token_logprobs, response.system_fingerprint)
                    return token_logprobs
                else:
                    raise ValueError(
                        f"Failed to get logprobs from API response: {response.choices}"
                    )

            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(
                        f"API call failed on attempt {attempt + 1}, retrying: {str(e)}"
                    )
                    continue
                else:
                    raise Exception(
                        f"Failed to get token logprobs after {max_retries} attempts: {str(e)}"
                    )

                # response = self.openai_client.completions.create(
                #     model=self.model,
                #     prompt=context,
                #     max_tokens=1,
                #     logprobs=20,  # Maximum allowed by OpenAI
                #     temperature=1.5,
                #     seed=42,
                #     n=1,
                #     suffix=" continues..."
                # )
                #
                # if response.choices[0].logprobs and response.choices[0].logprobs.top_logprobs:
                #     # The legacy API returns logprobs differently
                #     logprobs_dict = response.choices[0].logprobs.top_logprobs[0]
                #     token_logprobs = {}
                #
                #     for token, logprob in logprobs_dict.items():
                #         token_logprobs[token] = logprob  # Keep as logprob
                #
                #     return token_logprobs
                # else:
                #     raise ValueError(f"Failed to get logprobs from API response: {response.choices}")

    def _compute_interval_division(
        self, context: str, cur_interval: List[int], precision: int = 16
    ) -> Tuple[List[int], List[str]]:
        """Compute cumulative probabilities for current interval division.

        Args:
            context: The text context to get next token logprobs for
            cur_interval: Current interval [bottom, top) as [int, int]
            precision: Precision for arithmetic coding (default: 16)

        Returns:
            Tuple of (cum_probs, tokens_temp) where:
            - cum_probs: List of cumulative probabilities adjusted to current interval
            - tokens_temp: List of corresponding tokens in same order
        """
        # Get token logprobs from LLM for current context
        token_logprobs = self._get_token_logprobs(context)
        self.logger.debug(f'{token_logprobs=}')

        # Convert logprobs to probabilities with additional temperature scaling
        # Apply extra temperature to flatten the distribution since OpenAI caps at 2.0
        extra_temperature = self.config.extra_temperature  # Additional flattening on top of API temperature
        tokens = list(token_logprobs.keys())
        probs = [math.exp(logprob / extra_temperature) for logprob in token_logprobs.values()]

        # Sort by probability (descending)
        sorted_pairs = sorted(zip(probs, tokens), reverse=True)
        sorted_probs = [p for p, _ in sorted_pairs]
        sorted_tokens = [t for _, t in sorted_pairs]

        # Normalize probabilities
        total_prob = sum(sorted_probs)
        assert total_prob > 0
        sorted_probs = [p / total_prob for p in sorted_probs]

        # Cutoff low probabilities that would be rounded to 0
        cur_int_range = cur_interval[1] - cur_interval[0]
        if cur_int_range <= 0:
            raise ValueError("Interval has collapsed")
        cur_threshold = self.config.min_prob_threshold_divisor / float(cur_int_range)
        k = min(len(sorted_probs), self.config.topk_limit)  # topk equivalent

        # Find cutoff point
        for idx in range(len(sorted_probs)):
            if sorted_probs[idx] < cur_threshold:
                k = min(max(2, idx), self.config.topk_limit)
                break

        # Keep only top k probabilities
        probs_temp = sorted_probs[:k]
        tokens_temp = sorted_tokens[:k]

        # Filter out overlapping tokens to ensure deterministic decoding
        # When tokens overlap (one is a prefix of another), keep the more probable one
        filtered_indices = []

        for i in range(len(tokens_temp)):
            should_keep = True
            token_i = tokens_temp[i]

            # Check against all more probable tokens (those with lower indices)
            for j in range(i):
                token_j = tokens_temp[j]
                # If a more probable token overlaps with this one, skip this one
                if token_i.startswith(token_j) or token_j.startswith(token_i):
                    should_keep = False
                    break

            if should_keep:
                filtered_indices.append(i)

        # Keep only non-overlapping tokens
        tokens_temp = [tokens_temp[i] for i in filtered_indices]
        probs_temp = [probs_temp[i] for i in filtered_indices]

        # Ensure we have at least 1 token for encoding
        if len(tokens_temp) < 1:
            raise ValueError(
                f"Not enough non-overlapping tokens for encoding: only {len(tokens_temp)} tokens available"
            )

        # Check if interval is too narrow to properly represent probabilities
        # We need at least 2 units per token to ensure proper interval subdivision
        min_required_range = len(tokens_temp) * 2
        if cur_int_range < min_required_range:
            raise ValueError(
                f"Interval too narrow for encoding: range={cur_int_range}, "
                f"but need at least {min_required_range} for {len(tokens_temp)} tokens"
            )

        # Rescale to correct range (fix overflow by using float arithmetic)
        probs_sum = sum(probs_temp)
        assert probs_sum > 0
        probs_temp_scaled = [p / probs_sum * float(cur_int_range) for p in probs_temp]

        # Round probabilities to integers given precision
        # Ensure minimum probability to avoid zero-width intervals after splitting
        probs_temp_int = [max(self.config.min_interval_width, round(p)) for p in probs_temp_scaled]
        cum_probs = []
        cumsum = 0
        for p in probs_temp_int:
            cumsum += p
            cum_probs.append(cumsum)

        # Remove any elements from the bottom if rounding caused total prob to be too large
        if cum_probs[-1] > cur_int_range:
            # Scale down proportionally
            scale_factor = float(cur_int_range) / cum_probs[-1]
            cum_probs = [int(cp * scale_factor) for cp in cum_probs]

        # Add any mass to the top if removing/rounding causes total prob to be too small
        if cum_probs[-1] < cur_int_range:
            cum_probs[-1] = cur_int_range

        # Convert to position in range
        cum_probs = [cp + cur_interval[0] for cp in cum_probs]

        return cum_probs, tokens_temp

    @staticmethod
    def _int2bits(val: int, precision: int) -> List[int]:
        """Convert integer to list of bits (LSB first)."""
        if precision == 0:
            return []
        binary_str = format(val, f"0{precision}b")
        return [int(bit) for bit in reversed(binary_str)]

    @staticmethod
    def _bits2int(bits: List[int]) -> int:
        """Convert list of bits (LSB first) to integer."""
        val = 0
        for bit_idx, bit in enumerate(bits):
            val += bit * (2**bit_idx)
        return val

    @staticmethod
    def _num_same_from_beg(bits1: List[int], bits2: List[int]) -> int:
        """Count number of same bits from beginning of two bit lists."""
        count = 0
        for b1, b2 in zip(bits1, bits2):
            if b1 == b2:
                count += 1
            else:
                break
        return count

    def encode(self, context: str, message_bits: List[int], precision: int = 16, artifacts_dir: Optional[str] = None) -> str:
        """Encode a secret message into text using arithmetic coding steganography.

        Args:
            context: The initial context to start encoding from
            message_bits: List of bits to encode (as integers 0 or 1)
            precision: Precision for arithmetic coding (default: 16)

        Returns:
            The generated text containing the encoded message
        """
        self.logger.info(f"Starting encoding with {len(message_bits)} message bits")
        self.logger.debug(f"Message bits: {''.join(map(str, message_bits))}")
        
        # Output logging for final results
        self.logger.info(f"Encoding {len(message_bits)} bits")

        # Initialize artifacts collection if requested
        artifacts = None
        if artifacts_dir:
            artifacts_path = Path(artifacts_dir)
            artifacts_path.mkdir(parents=True, exist_ok=True)
            artifacts = {
                "mode": "encode",
                "precision": precision,
                "context": context,
                "message_bits": list(message_bits),
                "steps": []
            }

        # Initialize arithmetic coding parameters
        max_val = 2**precision
        cur_interval = [0, max_val]  # bottom inclusive, top exclusive

        # Start with the given context
        current_text = context
        generated_text = ""
        generated_tokens = []

        i = 0  # bit index
        step = 0

        while i < len(message_bits):
            step += 1
            self.logger.debug(f"Encode step {step}: Processing bit index {i}")
            self.logger.debug(f"Current interval: {cur_interval}")
            cur_interval_before = cur_interval.copy()
            cur_interval_before = cur_interval.copy()

            # Use extracted common logic for interval division
            try:
                cum_probs, tokens_temp = self._compute_interval_division(
                    current_text, cur_interval, precision
                )
            except ValueError as e:
                # Interval has collapsed - stop encoding
                self.logger.warning(f"Failed to compute interval, stopping encoding: {e}")
                break

            self.logger.debug(f"Cumulative probabilities: {cum_probs}")
            # Get selected index based on binary fraction from message bits
            # Take remaining bits, up to precision
            remaining_bits = len(message_bits) - i
            bits_to_use = min(remaining_bits, precision)
            current_message_bits = message_bits[i : i + bits_to_use]

            # Pad with zeros to reach precision for display purposes
            display_bits = current_message_bits + [0] * (precision - bits_to_use)
            self.logger.debug(f"Current message bits: {display_bits}")

            # Convert bits to integer following the reference implementation
            # The reference uses bits2int(reversed(message_bits)) where bits2int expects LSB-first
            # Since message_bits are MSB-first, we need to reverse them
            # Then pad with zeros if we have fewer bits than precision
            padded_message_bits = current_message_bits + [0] * (precision - bits_to_use)
            message_idx = self._bits2int(padded_message_bits[::-1])

            self.logger.debug(f"Message bits raw value: {message_idx}")

            # Find selection (cum_probs are already adjusted to current interval)
            selection = 0
            for idx, cp in enumerate(cum_probs):
                if cp > message_idx:
                    selection = idx
                    break

            # Select the token
            selected_token = (
                tokens_temp[selection]
                if selection < len(tokens_temp)
                else tokens_temp[0]
            )

            self.logger.debug(
                f"Selected token: {repr(selected_token)} (selection: {selection})"
            )

            # Calculate new range as ints
            new_int_bottom = (
                cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            )
            new_int_top = cum_probs[selection]

            self.logger.debug(f"New range: [{new_int_bottom}, {new_int_top})")

            # Convert range to bits and find common prefix
            new_int_bottom_bits_inc = list(
                reversed(self._int2bits(new_int_bottom, precision))
            )
            new_int_top_bits_inc = list(
                reversed(self._int2bits(new_int_top - 1, precision))
            )

            # Consume most significant bits which are now fixed and update interval
            num_bits_encoded = self._num_same_from_beg(
                new_int_bottom_bits_inc, new_int_top_bits_inc
            )

            self.logger.debug(f"Bits encoded this step: {num_bits_encoded}")
            encoded_bits = []
            if num_bits_encoded > 0:
                encoded_bits = new_int_bottom_bits_inc[:num_bits_encoded]
                self.logger.debug(f"Encoded bits: {encoded_bits}")

            i += num_bits_encoded

            # Update interval for next iteration
            new_int_bottom_bits = (
                new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
            )
            new_int_top_bits = (
                new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded
            )

            cur_interval[0] = self._bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = self._bits2int(reversed(new_int_top_bits)) + 1

            self.logger.debug(f"Updated interval: {cur_interval}")

            # Record step data for visualization
            if artifacts is not None:
                artifacts["steps"].append({
                    "step": step,
                    "cur_interval_before": cur_interval_before,
                    "cum_probs": cum_probs,
                    "tokens": tokens_temp,
                    "message_idx": message_idx,
                    "selection": selection,
                    "selected_token": selected_token,
                    "new_range": [new_int_bottom, new_int_top],
                    "num_bits_emitted": num_bits_encoded,
                    "emitted_bits": encoded_bits,
                    "cur_interval_after": cur_interval.copy()
                })

            if cur_interval[1] <= cur_interval[0]:
                raise ValueError(
                    f"Interval collapsed: cur_interval={cur_interval} at step {step}. "
                    "Check scaling/rounding logic for possible error."
                )

            # Append the token as a string to maintain synchronization
            generated_text += selected_token
            generated_tokens += [selected_token]
            current_text += selected_token  # Update the full context for next iteration

            self.logger.debug(f"Generated text: {repr(generated_text)}")

        self.logger.debug(f"Generated tokens: {repr(generated_tokens)}")
        
        self.logger.info("Encoding completed")
        # Persist artifacts data if enabled
        if artifacts is not None:
            artifacts["generated_text"] = generated_text
            try:
                with open(artifacts_path / "encoding-coding-data.json", "w", encoding="utf-8") as f:
                    json.dump(artifacts, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to write coding_data.json: {e}")
        return generated_text

    def decode(
        self,
        context: str,
        encoded_text: str,
        precision: int = 16,
        message_length: Optional[int] = None,
        artifacts_dir: Optional[str] = None,
    ) -> List[int]:
        """Decode a secret message from encoded text using arithmetic coding steganography.

        Args:
            context: The initial context that was used during encoding
            encoded_text: The encoded text containing the hidden message
            precision: Precision for arithmetic coding (default: 16)
            message_length: Expected length of message in bits (optional, for compatibility only - EOF marker detection is preferred)

        Returns:
            List of decoded bits (as integers 0 or 1) without the EOF marker
        """
        self.logger.info(f"Starting decoding with {len(encoded_text)} character encoded text")
        self.logger.debug(f"Expected message_length: {message_length}")
        
        # Output logging for final results
        self.logger.info("Starting decoding")

        # Initialize artifacts collection if requested
        artifacts = None
        if artifacts_dir:
            artifacts_path = Path(artifacts_dir)
            artifacts_path.mkdir(parents=True, exist_ok=True)
            artifacts = {
                "mode": "decode",
                "precision": precision,
                "context": context,
                "encoded_text": encoded_text,
                "message_length": message_length,
                "steps": []
            }

        # Initialize arithmetic coding parameters (same as encoding)
        max_val = 2**precision
        cur_interval = [0, max_val]  # bottom inclusive, top exclusive

        # Initialize message accumulator
        message_bits = []

        # Initialize current text with the initial context
        current_text = context

        # We'll track position in the encoded text
        encoded_text_pos = 0
        step = 0

        def _has_eof_marker(bits: List[int]) -> bool:
            """Check if the decoded bits end with an EOF marker."""
            BITS_PER_BYTE = 8
            EOF_BITS = [0, 0, 0, 0, 0, 0, 1, 1]  # Binary representation of 0x03
            
            if len(bits) < BITS_PER_BYTE:
                return False
                
            # Check if the last byte is EOF marker
            return bits[-BITS_PER_BYTE:] == EOF_BITS
            
        while encoded_text_pos < len(encoded_text) and (
            message_length is None or len(message_bits) < message_length
        ):
            step += 1
            self.logger.debug(
                f"Decode step {step}: Position {encoded_text_pos}, decoded {len(message_bits)} bits so far"
            )
            self.logger.debug(f"Current interval: {cur_interval}")
            cur_interval_before = cur_interval.copy()

            # Re-tokenize the current accumulated text to get the probability distribution
            try:
                cum_probs, tokens_temp = self._compute_interval_division(
                    current_text, cur_interval, precision
                )
            except ValueError:
                # Interval has collapsed - stop decoding
                self.logger.warning("Interval collapsed, stopping decoding")
                break

            self.logger.debug(f"Cumulative probabilities: {cum_probs}")
            # Find the longest matching token from our candidates at the current position
            # This ensures proper tokenization alignment between encoding and decoding
            current_token = None
            selection = -1
            max_token_length = 0

            self.logger.debug(f"Looking for token match at position {encoded_text_pos}")
            self.logger.debug(f"Remaining text: {repr(encoded_text[encoded_text_pos:])}")

            for i, candidate_token in enumerate(tokens_temp):
                # Check if this candidate token appears at the current position in encoded_text
                if (
                    encoded_text[encoded_text_pos:].startswith(candidate_token)
                    and len(candidate_token) > max_token_length
                ):
                    current_token = candidate_token
                    selection = i
                    max_token_length = len(candidate_token)

            if current_token is None:
                # No candidate token matches - this shouldn't happen with proper synchronization
                # Skip one character and continue
                self.logger.warning("No matching token found, skipping character")
                if encoded_text_pos < len(encoded_text):
                    current_text += encoded_text[encoded_text_pos]
                    encoded_text_pos += 1
                continue

            self.logger.debug(
                f"Found token: {repr(current_token)} (selection: {selection})"
            )

            # Update position in encoded text
            encoded_text_pos += len(current_token)

            # Calculate new range as ints (same logic as reference)
            new_int_bottom = (
                cum_probs[selection - 1] if selection > 0 else cur_interval[0]
            )
            new_int_top = cum_probs[selection]

            self.logger.debug(f"New range: [{new_int_bottom}, {new_int_top})")

            # Convert range to bits and find common prefix
            new_int_bottom_bits_inc = list(
                reversed(self._int2bits(new_int_bottom, precision))
            )
            new_int_top_bits_inc = list(
                reversed(self._int2bits(new_int_top - 1, precision))
            )

            # Emit most significant bits which are now fixed
            num_bits_encoded = self._num_same_from_beg(
                new_int_bottom_bits_inc, new_int_top_bits_inc
            )

            self.logger.debug(f"Bits to decode this step: {num_bits_encoded}")

            # Always use the normal bit extraction logic - don't treat last token specially
            new_bits = []
            if num_bits_encoded > 0:
                # Normal case - emit the common bits
                new_bits = new_int_bottom_bits_inc[:num_bits_encoded]
                self.logger.debug(f"Decoded bits: {new_bits}")
                # If we have a message length limit, don't decode more bits than needed
                if message_length is not None:
                    remaining_bits = message_length - len(message_bits)
                    if remaining_bits > 0:
                        new_bits = new_bits[:remaining_bits]
                        message_bits.extend(new_bits)
                else:
                    message_bits.extend(new_bits)

            # Check for EOF marker only when we have complete bytes
            BITS_PER_BYTE = 8
            if len(message_bits) % BITS_PER_BYTE == 0 and _has_eof_marker(message_bits):
                self.logger.info("EOF marker detected, stopping decoding")
                # Remove the EOF marker from the message bits
                message_bits = message_bits[:-BITS_PER_BYTE]  # Remove last 8 bits (EOF marker)
                break

            # Update interval for next iteration
            new_int_bottom_bits = (
                new_int_bottom_bits_inc[num_bits_encoded:] + [0] * num_bits_encoded
            )
            new_int_top_bits = (
                new_int_top_bits_inc[num_bits_encoded:] + [1] * num_bits_encoded
            )

            cur_interval[0] = self._bits2int(reversed(new_int_bottom_bits))
            cur_interval[1] = self._bits2int(reversed(new_int_top_bits)) + 1

            # Update current_text by appending the token string (just like in encoding)
            current_text += current_token

            # Record step data for visualization
            if artifacts is not None:
                artifacts["steps"].append({
                    "step": step,
                    "cur_interval_before": cur_interval_before,
                    "cum_probs": cum_probs,
                    "tokens": tokens_temp,
                    "selection": selection,
                    "selected_token": current_token,
                    "new_range": [new_int_bottom, new_int_top],
                    "num_bits_emitted": num_bits_encoded,
                    "emitted_bits": new_bits,
                    "cur_interval_after": cur_interval.copy()
                })

        self.logger.debug(f"Final decoded bits: {message_bits}")
        self.logger.debug(f"Decoded bits as string: {''.join(map(str, message_bits))}")
        
        self.logger.info("Decoding completed")
        # Persist artifacts data if enabled
        if artifacts is not None:
            artifacts["decoded_bits"] = list(message_bits)
            try:
                with open(artifacts_path / "decoding-coding-data.json", "w", encoding="utf-8") as f:
                    json.dump(artifacts, f, indent=2)
            except Exception as e:
                self.logger.warning(f"Failed to write coding_data.json: {e}")
        return message_bits

    def text_to_bits(self, text: str) -> List[int]:
        """Convert text to a list of bits with EOF marker.

        Args:
            text: The text to convert

        Returns:
            List of bits (as integers 0 or 1) with EOF marker appended
        """
        BITS_PER_BYTE = 8
        EOF_MARKER = 0x03  # ETX (End of Text)
        EOF_BITS = [0, 0, 0, 0, 0, 0, 1, 1]  # Binary representation of 0x03
        
        bits = []
        for char in text:
            # Escape any existing ETX characters by doubling them
            if ord(char) == EOF_MARKER:
                # Add escaped ETX (0x03 0x03)
                bits.extend(EOF_BITS)
                bits.extend(EOF_BITS)
            else:
                # Convert character to BITS_PER_BYTE-bit representation
                char_bits = format(ord(char), f"0{BITS_PER_BYTE}b")
                bits.extend([int(bit) for bit in char_bits])
        
        # Append EOF marker (ETX = 0x03 = 00000011)
        bits.extend(EOF_BITS)
        return bits

    def bits_to_text(self, bits: List[int]) -> str:
        """Convert a list of bits back to text, stopping at EOF marker.

        Args:
            bits: List of bits (as integers 0 or 1)

        Returns:
            The decoded text (without EOF marker)
        """
        BITS_PER_BYTE = 8
        EOF_MARKER = 0x03  # ETX (End of Text)
        
        if len(bits) % BITS_PER_BYTE != 0:
            # Pad with zeros to make it divisible by BITS_PER_BYTE
            bits = bits + [0] * (BITS_PER_BYTE - len(bits) % BITS_PER_BYTE)

        text = ""
        i = 0
        while i < len(bits):
            byte_bits = bits[i : i + BITS_PER_BYTE]
            if len(byte_bits) < BITS_PER_BYTE:
                break
                
            byte_value = sum(bit * (2 ** (7 - j)) for j, bit in enumerate(byte_bits))
            
            if byte_value == EOF_MARKER:  # Found ETX
                # Check if this is an escaped ETX (0x03 0x03)
                if i + BITS_PER_BYTE < len(bits):
                    next_byte_bits = bits[i + BITS_PER_BYTE : i + 2 * BITS_PER_BYTE]
                    if len(next_byte_bits) == BITS_PER_BYTE:
                        next_byte_value = sum(bit * (2 ** (7 - j)) for j, bit in enumerate(next_byte_bits))
                        if next_byte_value == EOF_MARKER:
                            # This is escaped ETX, add one ETX to text and skip both bytes
                            text += chr(EOF_MARKER)
                            i += 2 * BITS_PER_BYTE
                            continue
                
                # This is the EOF marker, stop decoding
                break
            
            if byte_value > 0:  # Skip null bytes
                text += chr(byte_value)
                
            i += BITS_PER_BYTE

        return text

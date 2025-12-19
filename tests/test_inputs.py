import numpy as np
import pytest
from julax.inputs import preprocess_text_inputs

# The beginning-of-sequence token ID is consistently 0 for all test cases.
BOS_TOKEN_ID = 0

# List of test cases for parameterization. Each case is defined with pytest.param
# for better readability and separate tracking in test results.
# Each parameter tuple contains: (tokens, expected_mask, expected_position_ids)
TEST_CASES = [
    pytest.param(
        # Input: A single sequence starting with BOS.
        np.array([[BOS_TOKEN_ID, 1, 2, 3]]),
        # Expected mask: A standard 4x4 causal mask.
        np.tril(np.ones((4, 4), dtype=bool))[None, :, :],
        # Expected positions: Standard increasing positions.
        np.array([[0, 1, 2, 3]]),
        id="single_sequence_with_bos",
    ),
    pytest.param(
        # Input: A single sequence with no BOS tokens.
        np.array([[1, 2, 3, 4]]),
        # Expected mask: A standard 4x4 causal mask.
        np.tril(np.ones((4, 4), dtype=bool))[None, :, :],
        # Expected positions: Standard increasing positions.
        np.array([[0, 1, 2, 3]]),
        id="no_bos_tokens",
    ),
    pytest.param(
        # Input: Two packed sequences: [BOS, 1] and [BOS, 2, 3].
        np.array([[BOS_TOKEN_ID, 1, BOS_TOKEN_ID, 2, 3]]),
        # Expected mask: A block-causal mask for segments (0,1) and (2,3,4).
        np.array(
            [
                [
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                ]
            ],
            dtype=bool,
        ),
        # Expected positions: Positions reset at the second BOS token.
        np.array([[0, 1, 0, 1, 2]]),
        id="packed_sequences",
    ),
    pytest.param(
        # Input: Two packed sequences where the first doesn't start with BOS: [1, 2] and [BOS, 3].
        np.array([[1, 2, BOS_TOKEN_ID, 3]]),
        # Expected mask: A block-causal mask for segments (0,1) and (2,3).
        np.array(
            [[[1, 0, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]], dtype=bool
        ),
        # Expected positions: Positions reset at the BOS token.
        np.array([[0, 1, 0, 1]]),
        id="sequence_without_initial_bos",
    ),
    pytest.param(
        # Input: A batch of 2 sequences with mixed packing.
        np.array(
            [
                [
                    BOS_TOKEN_ID,
                    1,
                    BOS_TOKEN_ID,
                    3,
                    4,
                ],  # Packed: [BOS, 1] and [BOS, 3, 4]
                [1, 2, 3, 4, 5],  # Not packed
            ]
        ),
        # Expected mask: A batch of masks, one block-causal and one standard-causal.
        np.array(
            [
                [
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                ],  # Block-causal
                np.tril(np.ones((5, 5), dtype=bool)),  # Standard causal
            ],
            dtype=bool,
        ),
        # Expected positions: A batch of positions, one resetting and one standard.
        np.array([[0, 1, 0, 1, 2], [0, 1, 2, 3, 4]]),
        id="batch_of_sequences",
    ),
]


@pytest.mark.parametrize("tokens, expected_mask, expected_position_ids", TEST_CASES)
def test_preprocess_text_inputs(tokens, expected_mask, expected_position_ids):
    """
    Tests the preprocess_text_inputs function with various parameterized scenarios
    covering single sequences, packed sequences, and batched inputs.
    """
    # Run the function being tested.
    result = preprocess_text_inputs(tokens, BOS_TOKEN_ID)

    # Assert that the generated mask and position_ids match the expected outputs.
    np.testing.assert_array_equal(result["mask"], expected_mask)
    np.testing.assert_array_equal(result["position_ids"], expected_position_ids)

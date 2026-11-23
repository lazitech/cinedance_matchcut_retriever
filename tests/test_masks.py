import numpy as np

from matchcut_retriever.masks import rle_decode, rle_encode


def test_rle_round_trip_edge_shapes():
    masks = [
        np.zeros((0, 7), dtype=bool),
        np.zeros((3, 5), dtype=bool),
        np.ones((3, 5), dtype=bool),
        np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
    ]
    for mask in masks:
        assert np.array_equal(rle_decode(rle_encode(mask)), mask)

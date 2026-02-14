"""Unit tests for pipeline.dedup."""

import numpy as np
import pytest

from pipeline.dedup import PhashLRU, dhash


def test_dhash_deterministic():
    pytest.importorskip("cv2", reason="cv2 required for dhash")
    img = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
    h1 = dhash(img)
    h2 = dhash(img)
    assert h1 == h2


def test_dhash_different_images():
    pytest.importorskip("cv2", reason="cv2 required for dhash")
    img1 = np.zeros((64, 64), dtype=np.uint8)
    img2 = np.ones((64, 64), dtype=np.uint8) * 255
    h1 = dhash(img1)
    h2 = dhash(img2)
    assert h1 != h2


def test_phash_lru_near_duplicate():
    lru = PhashLRU(max_size=100, hamming_max=6)
    h = 0x123456789ABCDEF0
    assert not lru.near_duplicate(h)
    lru.add(h)
    assert lru.near_duplicate(h)
    # Same hash is duplicate
    assert lru.near_duplicate(h)


def test_phash_lru_eviction():
    lru = PhashLRU(max_size=3, hamming_max=0)
    lru.add(1)
    lru.add(2)
    lru.add(3)
    assert lru.near_duplicate(1)
    lru.add(4)
    # 1 should be evicted (FIFO)
    assert not lru.near_duplicate(1)
    assert lru.near_duplicate(2)
    assert lru.near_duplicate(3)
    assert lru.near_duplicate(4)

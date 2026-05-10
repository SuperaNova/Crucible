import hashlib
import sys
from pathlib import Path

import numpy as np


DEFAULT_NPY_PATH = Path("data/pixel_art/sprites.npy")
DEFAULT_OUT_PATH = Path("data/sprites_clean.npy")


class Smelter:
    """
    Filters raw sprite data from sprites.npy into a clean, deduplicated
    set of item sprites suitable for use in the Crucible pipeline.

    Heuristics applied in order:
      1. Alpha coverage  — rejects nearly-transparent sprites
      2. Std deviation   — rejects near-monochrome sprites
      3. Color uniqueness — rejects sprites with too few distinct colors
      4. Saturation      — rejects flat/greyscale sprites
      5. Deduplication   — removes exact pixel-level duplicates via MD5
    """

    # --- Thresholds ---
    ALPHA_THRESHOLD = 30        # Pixel alpha value considered "opaque"
    ALPHA_MIN_COVERAGE = 0.10   # Minimum fraction of opaque pixels
    STD_MIN = 15.0              # Minimum RGB standard deviation
    COLOR_BINS = 20             # Bin size for color bucket quantization
    COLOR_MIN_UNIQUE = 6        # Minimum number of unique color buckets
    SATURATION_MIN = 10.0       # Minimum mean (max - min) channel spread

    def smelt(self, npy_path: Path = DEFAULT_NPY_PATH) -> np.ndarray:
        """
        Full pipeline: load -> filter -> deduplicate -> return clean array.

        Args:
            npy_path: Path to sprites.npy from the Kaggle dataset.

        Returns:
            Clean ndarray of shape (N, H, W, C).
        """
        npy_path = Path(npy_path)
        if not npy_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{npy_path}'.\n"
                "Run 'python setup.py' first to download and prepare the data."
            )

        print(f"[Smelter] Loading sprites from '{npy_path}' ...")
        raw = np.load(str(npy_path))
        print(f"[Smelter] Raw sprites loaded: {len(raw)}")

        mask = np.array([self._is_valid(s) for s in raw])
        filtered = raw[mask]
        print(f"[Smelter] After heuristic filters: {len(filtered)}"
              f" (removed {len(raw) - len(filtered)})")

        clean = self._deduplicate(filtered)
        print(f"[Smelter] After deduplication: {len(clean)}"
              f" (removed {len(filtered) - len(clean)} duplicates)")

        return clean

    # ------------------------------------------------------------------
    # Internal filters
    # ------------------------------------------------------------------

    def _is_valid(self, sprite: np.ndarray) -> bool:
        """Runs all heuristic checks; returns True if the sprite passes."""
        return (
            self._alpha_check(sprite)
            and self._std_check(sprite)
            and self._color_uniqueness_check(sprite)
            and self._saturation_check(sprite)
        )

    def _alpha_check(self, sprite: np.ndarray) -> bool:
        """Reject sprites where fewer than 10% of pixels are opaque."""
        if sprite.shape[-1] != 4:
            return True  # No alpha channel — treat as fully opaque
        opaque_fraction = (sprite[..., 3] > self.ALPHA_THRESHOLD).mean()
        return opaque_fraction >= self.ALPHA_MIN_COVERAGE

    def _std_check(self, sprite: np.ndarray) -> bool:
        """Reject near-monochrome sprites with very low RGB variance."""
        rgb = sprite[..., :3].astype(np.float32)
        return float(rgb.std()) >= self.STD_MIN

    def _color_uniqueness_check(self, sprite: np.ndarray) -> bool:
        """
        Reject sprites with fewer than COLOR_MIN_UNIQUE distinct color
        buckets when the RGB space is divided into bins of size COLOR_BINS.
        """
        rgb = sprite[..., :3].astype(np.float32)
        pixels = rgb.reshape(-1, 3)
        buckets = (pixels / self.COLOR_BINS).astype(int)
        n_unique = len(np.unique(buckets, axis=0))
        return n_unique >= self.COLOR_MIN_UNIQUE

    def _saturation_check(self, sprite: np.ndarray) -> bool:
        """Reject flat/greyscale sprites with low per-pixel channel spread."""
        rgb = sprite[..., :3].astype(np.float32)
        pixels = rgb.reshape(-1, 3)
        spread = (pixels.max(axis=1) - pixels.min(axis=1)).mean()
        return float(spread) >= self.SATURATION_MIN

    def _deduplicate(self, sprites: np.ndarray) -> np.ndarray:
        """
        Remove exact duplicates by hashing the RGB bytes of each sprite.
        Keeps the first occurrence of each unique sprite.
        """
        seen: set = set()
        unique_indices = []
        for i, sprite in enumerate(sprites):
            digest = hashlib.md5(sprite[..., :3].tobytes()).hexdigest()
            if digest not in seen:
                seen.add(digest)
                unique_indices.append(i)
        return sprites[unique_indices]


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    npy_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NPY_PATH
    smelter = Smelter()
    try:
        clean = smelter.smelt(npy_path)
        print(f"[Smelter] Done. {len(clean)} sprites ready.")
    except FileNotFoundError as e:
        print(f"[Smelter] ERROR: {e}")
        sys.exit(1)

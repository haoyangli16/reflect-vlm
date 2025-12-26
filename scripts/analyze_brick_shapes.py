#!/usr/bin/env python3
"""
Analyze and describe brick shapes from voxel representations.

This script explores how to generate meaningful symbolic descriptions
of bricks based on their voxel structure, for use in memory/learning systems.
"""

import argparse
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
import json

from roboworld.envs.generator import (
    generate_xml,
    Board,
    Brick,
    Nail,
    Peg,
    get_color_name,
    COLORS,
    VOXEL_SIZE,
)
from roboworld.envs.asset_path_utils import full_path_for


@dataclass
class SlotDescription:
    """Description of a slot/channel in a brick."""

    direction: str  # "lengthwise", "widthwise", "vertical"
    depth: int  # in voxels
    width: int  # in voxels
    position: str  # "left", "center", "right" or "top", "bottom"
    is_through: bool  # Does it go all the way through?


@dataclass
class HoleDescription:
    """Description of a hole for nail/peg insertion."""

    hole_type: str  # "nail_hole", "peg_hole", "channel"
    shape: Optional[str]  # For peg holes: "arch", "circle", etc.
    position: Tuple[int, int]  # (x, y) position on brick
    depth: int  # in voxels


@dataclass
class BrickDescription:
    """Complete symbolic description of a brick."""

    brick_id: int
    brick_name: str
    brick_type: str  # "block", "nail", "peg", "board"
    color: str

    # Dimensions
    length: int  # voxels (longest horizontal dim)
    width: int  # voxels (shorter horizontal dim)
    height: int  # voxels (vertical dim)
    volume: int  # total solid voxels

    # Shape characteristics
    aspect_ratio: str  # "elongated", "square", "tall", "flat"
    main_axis: str  # "x" or "y"
    is_symmetric: bool

    # Features
    slots: List[SlotDescription] = field(default_factory=list)
    holes: List[HoleDescription] = field(default_factory=list)
    has_carved_channels: bool = False

    # Relationships
    blocks_ids: List[int] = field(default_factory=list)  # IDs of bricks this one blocks
    blocked_by_ids: List[int] = field(default_factory=list)  # IDs that block this one

    def to_natural_language(self) -> str:
        """Generate natural language description."""
        parts = []

        # Basic description
        parts.append(f"A {self.color} {self.brick_type}")

        # Dimensions
        dims = f"({self.length}×{self.width}×{self.height} voxels)"
        parts.append(dims)

        # Aspect ratio
        parts.append(f"that is {self.aspect_ratio}")

        # Slots
        if self.slots:
            slot_desc = []
            for slot in self.slots:
                through_str = "through-" if slot.is_through else ""
                slot_desc.append(
                    f"a {through_str}slot running {slot.direction} "
                    f"(depth={slot.depth}, width={slot.width})"
                )
            parts.append("with " + ", ".join(slot_desc))

        # Holes
        if self.holes:
            hole_desc = []
            for hole in self.holes:
                if hole.shape:
                    hole_desc.append(f"a {hole.shape}-shaped hole")
                else:
                    hole_desc.append(f"a {hole.hole_type}")
            parts.append("containing " + ", ".join(hole_desc))

        # Carved channels
        if self.has_carved_channels:
            parts.append("with carved channels from intersecting pieces")

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.brick_id,
            "name": self.brick_name,
            "type": self.brick_type,
            "color": self.color,
            "dimensions": {
                "length": self.length,
                "width": self.width,
                "height": self.height,
                "volume": self.volume,
            },
            "shape": {
                "aspect_ratio": self.aspect_ratio,
                "main_axis": self.main_axis,
                "is_symmetric": self.is_symmetric,
            },
            "features": {
                "slots": [vars(s) for s in self.slots],
                "holes": [vars(h) for h in self.holes],
                "has_carved_channels": self.has_carved_channels,
            },
            "relationships": {
                "blocks": self.blocks_ids,
                "blocked_by": self.blocked_by_ids,
            },
            "natural_language": self.to_natural_language(),
        }


def analyze_voxels(voxels: np.ndarray) -> Dict[str, Any]:
    """Analyze a voxel array to extract shape features."""
    features = {}

    # Basic dimensions
    size = voxels.shape
    features["size"] = size
    features["volume"] = np.sum(voxels > 0)
    features["empty_volume"] = np.sum(voxels == 0)
    features["hole_volume"] = np.sum(voxels < 0)

    # Determine main axis (longest horizontal dimension)
    if size[0] >= size[1]:
        features["main_axis"] = "x"
        features["length"] = size[0]
        features["width"] = size[1]
    else:
        features["main_axis"] = "y"
        features["length"] = size[1]
        features["width"] = size[0]
    features["height"] = size[2]

    # Aspect ratio
    max_dim = max(size[0], size[1])
    min_dim = min(size[0], size[1])
    ratio = max_dim / max(1, min_dim)
    if ratio > 2.5:
        features["aspect_ratio"] = "elongated"
    elif ratio < 1.5:
        if size[2] > max_dim:
            features["aspect_ratio"] = "tall"
        elif size[2] < min_dim / 2:
            features["aspect_ratio"] = "flat"
        else:
            features["aspect_ratio"] = "square"
    else:
        features["aspect_ratio"] = "rectangular"

    # Symmetry check (along main axis)
    if features["main_axis"] == "x":
        flipped = np.flip(voxels, axis=0)
    else:
        flipped = np.flip(voxels, axis=1)
    features["is_symmetric"] = np.array_equal(voxels, flipped)

    # Detect slots (carved channels)
    features["slots"] = detect_slots(voxels)

    # Detect holes (negative values)
    features["holes"] = detect_holes(voxels)

    # Check for carved channels (empty regions that aren't holes)
    features["has_carved_channels"] = features["empty_volume"] > 0

    return features


def detect_slots(voxels: np.ndarray) -> List[Dict]:
    """Detect slots/channels in the voxel array."""
    slots = []
    size = voxels.shape

    # Check for lengthwise slots (along x-axis)
    for y in range(size[1]):
        for z in range(size[2]):
            if voxels[0, y, z] == 0 and voxels[-1, y, z] == 0:
                # Potential through-slot
                empty_count = np.sum(voxels[:, y, z] == 0)
                if empty_count == size[0]:
                    slots.append(
                        {
                            "direction": "lengthwise",
                            "depth": size[0],
                            "width": 1,
                            "position": f"y={y},z={z}",
                            "is_through": True,
                        }
                    )

    # Check for widthwise slots (along y-axis)
    for x in range(size[0]):
        for z in range(size[2]):
            if voxels[x, 0, z] == 0 and voxels[x, -1, z] == 0:
                empty_count = np.sum(voxels[x, :, z] == 0)
                if empty_count == size[1]:
                    slots.append(
                        {
                            "direction": "widthwise",
                            "depth": size[1],
                            "width": 1,
                            "position": f"x={x},z={z}",
                            "is_through": True,
                        }
                    )

    # Group adjacent slots
    # (simplified - just count unique positions)

    return slots[:5]  # Limit to prevent explosion


def detect_holes(voxels: np.ndarray) -> List[Dict]:
    """Detect holes (negative voxel values) in the array."""
    holes = []

    # Find positions with negative values (holes)
    hole_positions = np.argwhere(voxels < 0)

    if len(hole_positions) > 0:
        # Get unique hole values (different hole types)
        unique_holes = np.unique(voxels[voxels < 0])

        for hole_val in unique_holes:
            positions = np.argwhere(voxels == hole_val)
            center = positions.mean(axis=0).astype(int)
            depth = len(positions)

            holes.append(
                {
                    "hole_type": "insertion_hole",
                    "hole_id": int(hole_val),
                    "position": tuple(center[:2].tolist()),
                    "depth": depth,
                    "shape": None,  # Would need mesh info
                }
            )

    return holes


def analyze_brick(brick: Brick, brick_id: int, dependencies: set) -> BrickDescription:
    """Create a complete description of a brick."""
    # Get color
    color = get_color_name(brick.rgba[:3]) if brick.rgba is not None else "unknown"

    # Determine type
    if isinstance(brick, Nail):
        brick_type = "nail"
    elif isinstance(brick, Peg):
        brick_type = "peg"
    elif brick.name == "brick_1":
        brick_type = "board"
    else:
        brick_type = "block"

    # Analyze voxels
    features = analyze_voxels(brick.voxels)

    # Build description
    desc = BrickDescription(
        brick_id=brick_id,
        brick_name=brick.name,
        brick_type=brick_type,
        color=color,
        length=features["length"],
        width=features["width"],
        height=features["height"],
        volume=features["volume"],
        aspect_ratio=features["aspect_ratio"],
        main_axis=features["main_axis"],
        is_symmetric=features["is_symmetric"],
        has_carved_channels=features["has_carved_channels"],
    )

    # Add slots
    for slot in features["slots"]:
        desc.slots.append(
            SlotDescription(
                direction=slot["direction"],
                depth=slot["depth"],
                width=slot["width"],
                position=slot["position"],
                is_through=slot["is_through"],
            )
        )

    # Add holes
    for hole in features["holes"]:
        desc.holes.append(
            HoleDescription(
                hole_type=hole["hole_type"],
                shape=hole.get("shape"),
                position=hole["position"],
                depth=hole["depth"],
            )
        )

    # Add dependency relationships
    for blocker, blocked in dependencies:
        if blocker == brick_id:
            desc.blocks_ids.append(blocked)
        if blocked == brick_id:
            desc.blocked_by_ids.append(blocker)

    return desc


def analyze_board(board: Board, dependencies: set) -> List[BrickDescription]:
    """Analyze all bricks in a board."""
    descriptions = []

    for i, brick in enumerate(board.bricks):
        brick_id = i + 1  # 1-indexed
        desc = analyze_brick(brick, brick_id, dependencies)
        descriptions.append(desc)

    return descriptions


def main():
    parser = argparse.ArgumentParser(description="Analyze brick shapes from voxel data")
    parser.add_argument("--seed", type=int, default=1000000, help="Environment seed")
    parser.add_argument("--n_seeds", type=int, default=5, help="Number of seeds to analyze")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    args = parser.parse_args()

    all_results = []

    for seed in range(args.seed, args.seed + args.n_seeds):
        print(f"\n{'=' * 60}")
        print(f"Analyzing seed {seed}")
        print(f"{'=' * 60}")

        # Generate board
        np.random.seed(seed)
        xml, info = generate_xml(seed)

        # Since we can't access the Board directly from generate_xml,
        # we need to regenerate it
        from roboworld.envs.generator import generate_board

        np.random.seed(seed)
        import random

        random.seed(seed)
        board = generate_board()

        # Get dependencies
        dependencies = info["dependencies"]

        # Analyze all bricks
        descriptions = analyze_board(board, dependencies)

        result = {
            "seed": seed,
            "n_bricks": len(descriptions),
            "dependencies": list(dependencies),
            "bricks": [d.to_dict() for d in descriptions],
        }
        all_results.append(result)

        # Print summary
        print(f"\nFound {len(descriptions)} bricks:")
        for desc in descriptions:
            print(f"\n  {desc.brick_name} ({desc.color} {desc.brick_type}):")
            print(f"    Dimensions: {desc.length}×{desc.width}×{desc.height}")
            print(f"    Aspect: {desc.aspect_ratio}, Main axis: {desc.main_axis}")
            print(f"    Slots: {len(desc.slots)}, Holes: {len(desc.holes)}")
            print(f"    Blocks: {desc.blocks_ids}, Blocked by: {desc.blocked_by_ids}")
            print(f"    Description: {desc.to_natural_language()}")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Print cross-seed analysis
    print(f"\n{'=' * 60}")
    print("Cross-Seed Analysis")
    print(f"{'=' * 60}")

    # Collect all dimension combinations
    dim_combos = {}
    for result in all_results:
        for brick in result["bricks"]:
            if brick["type"] not in ["board"]:
                dims = (
                    brick["dimensions"]["length"],
                    brick["dimensions"]["width"],
                    brick["dimensions"]["height"],
                )
                if dims not in dim_combos:
                    dim_combos[dims] = 0
                dim_combos[dims] += 1

    print("\nMost common brick dimensions (L×W×H):")
    for dims, count in sorted(dim_combos.items(), key=lambda x: -x[1])[:10]:
        print(f"  {dims[0]}×{dims[1]}×{dims[2]}: {count} occurrences")


if __name__ == "__main__":
    main()

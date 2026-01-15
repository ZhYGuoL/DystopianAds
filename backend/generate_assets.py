#!/usr/bin/env python3
"""
Generate placeholder ad assets for the demo.
Run: python generate_assets.py
"""
import os
from pathlib import Path

import cv2
import numpy as np

ASSETS_DIR = Path(__file__).parent / "assets"


def generate_can_asset(name: str, color: tuple[int, int, int], size: int = 120):
    """Generate a simple can-shaped placeholder asset."""
    # Create RGBA image
    height = size * 2
    width = size
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Draw can body
    margin = 10
    can_color = (*color, 255)

    # Main body
    cv2.rectangle(
        img,
        (margin, margin + 20),
        (width - margin, height - margin - 20),
        can_color,
        -1,
    )

    # Top ellipse
    cv2.ellipse(
        img,
        (width // 2, margin + 20),
        (width // 2 - margin, 15),
        0, 0, 360,
        (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50), 255),
        -1,
    )

    # Bottom ellipse
    cv2.ellipse(
        img,
        (width // 2, height - margin - 20),
        (width // 2 - margin, 15),
        0, 0, 180,
        (max(0, color[0] - 30), max(0, color[1] - 30), max(0, color[2] - 30), 255),
        -1,
    )

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = name.upper()
    text_size = cv2.getTextSize(text, font, 0.5, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height // 2 + text_size[1] // 2

    # White text with shadow
    cv2.putText(img, text, (text_x + 1, text_y + 1), font, 0.5, (0, 0, 0, 255), 2)
    cv2.putText(img, text, (text_x, text_y), font, 0.5, (255, 255, 255, 255), 2)

    return img


def main():
    """Generate all placeholder assets."""
    ASSETS_DIR.mkdir(exist_ok=True)

    assets = {
        "coke": (34, 34, 200),      # Red (BGR)
        "pepsi": (200, 100, 50),     # Blue
        "sprite": (80, 180, 80),     # Green
        "fanta": (40, 140, 255),     # Orange
    }

    for name, color in assets.items():
        img = generate_can_asset(name, color)
        path = ASSETS_DIR / f"{name}.png"
        cv2.imwrite(str(path), img)
        print(f"Created: {path}")

    print(f"\nGenerated {len(assets)} placeholder assets in {ASSETS_DIR}")


if __name__ == "__main__":
    main()

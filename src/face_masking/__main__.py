"""Command-line entry point."""
import argparse
from pathlib import Path
from .pipeline import process_dataset


def main():
    parser = argparse.ArgumentParser(description="Run face masking on an image dataset")
    parser.add_argument("--data", type=Path, required=True, help="Directory containing images.csv")
    parser.add_argument("--weights", default="yolo11n-pose.pt", help="YOLO pose model weights")
    parser.add_argument("--display", action="store_true", help="Show each result interactively")
    args = parser.parse_args()
    process_dataset(args.data, display=args.display, pose_weights=args.weights)


if __name__ == "__main__":
    main()

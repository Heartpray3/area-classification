"""
preprocess.py

Entry-point script to generate ML-ready parquet features from GeoJSON files.
This wrapper keeps the project interface simple while reusing the core
implementation in preprocess_features.py.
"""

import argparse

from preprocess_features import preprocess_geojson


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/test parquet features from GeoJSON.")
    parser.add_argument("--train", default="train.geojson", help="Path to training GeoJSON file.")
    parser.add_argument("--test", default="test.geojson", help="Path to test GeoJSON file.")
    parser.add_argument("--train-out", default="train_features.parquet", help="Output train parquet path.")
    parser.add_argument("--test-out", default="test_features.parquet", help="Output test parquet path.")
    args = parser.parse_args()

    train_df = preprocess_geojson(args.train, is_train=True)
    train_df.to_parquet(args.train_out, index=False)
    print(f"Saved: {args.train_out}  shape={train_df.shape}")

    test_df = preprocess_geojson(args.test, is_train=False)
    test_df.to_parquet(args.test_out, index=False)
    print(f"Saved: {args.test_out}  shape={test_df.shape}")


if __name__ == "__main__":
    main()

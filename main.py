import argparse
import os
import pandas as pd

from crop_classification import classification, processing, stac, visualization


def command_search(args):
    items = stac.search_sentinel2(
        start_date=args.start_date,
        end_date=args.end_date,
        max_cloud_cover=args.max_cloud,
    )
    print(f"Found {len(items)} Sentinel-2 scenes")
    if items:
        print("Example scene assets:")
        for item in items[:1]:
            print(item.id, list(item.assets.keys()))


def command_composite(args):
    files = processing.generate_monthly_composites(
        args.month,
        output_dir=args.output_dir,
        target_resolution=args.resolution,
    )
    if files:
        print("Created composites:")
        for path in files:
            print(f" - {path}")
    else:
        print("No composites created. Check the raw blob paths and month selection.")


def command_train(args):
    stack, feature_names, profile = processing.load_composite_stack(args.input_dir)
    mask = processing.load_raster_stack(args.mask_path)[0][0]
    X, y = processing.extract_training_samples(stack, mask)
    model = classification.train_random_forest(X, y)
    os.makedirs(os.path.dirname(args.output_model) or ".", exist_ok=True)
    classification.save_model(model, args.output_model)
    print(f"Saved trained model to {args.output_model}")
    print(f"Training samples: {X.shape[0]}, features: {X.shape[1]}")
    print(f"OOB score: {model.oob_score_:.4f}")


def command_predict(args):
    model = classification.load_model(args.model)
    stack, profile = processing.load_raster_stack(args.stack_path)
    predicted = classification.predict_stack(model, stack)
    profile.update(dtype="int16", count=1)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    visualization.save_raster(predicted, profile, args.output_path)
    print(f"Saved prediction map to {args.output_path}")


def command_evaluate(args):
    model = classification.load_model(args.model)
    checkpoint = classification.load_checkpoint(args.checkpoint)
    X_test = checkpoint.get("X_test") or checkpoint.get("X_te")
    y_test = checkpoint.get("y_test") or checkpoint.get("y")
    if X_test is None or y_test is None:
        raise ValueError("Checkpoint file must contain X_test and y_test arrays")
    metrics = classification.evaluate_classifier(model, X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.4%}")
    print(metrics["classification_report"])


def command_visualize(args):
    lut = pd.read_csv(args.lut_path)
    mapping = dict(zip(lut[args.id_column], lut[args.name_column]))
    with pd.option_context("mode.use_inf_as_na", True):
        class_map = processing.load_raster_stack(args.input_path)[0][0].astype(int)
    visualization.plot_classification_map(
        class_map,
        code_to_name=mapping,
        output_path=args.output_path,
        title=args.title,
    )
    print(f"Saved visualization to {args.output_path}")


def make_parser():
    parser = argparse.ArgumentParser(description="Crop classification pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    search_parser = subparsers.add_parser(
        "search", help="Search Sentinel-2 scenes in Copernicus STAC"
    )
    search_parser.add_argument("--start-date", required=True)
    search_parser.add_argument("--end-date", required=True)
    search_parser.add_argument("--max-cloud", type=int, default=20)
    search_parser.set_defaults(func=command_search)

    composite_parser = subparsers.add_parser(
        "composite", help="Generate monthly composites from raw scenes"
    )
    composite_parser.add_argument("--month", required=True)
    composite_parser.add_argument("--output-dir", default="composites")
    composite_parser.add_argument("--resolution", type=int, default=20)
    composite_parser.set_defaults(func=command_composite)

    train_parser = subparsers.add_parser(
        "train", help="Train a Random Forest classifier"
    )
    train_parser.add_argument("--input-dir", default="composites")
    train_parser.add_argument("--mask-path", required=True)
    train_parser.add_argument(
        "--output-model", default="checkpoints/rf_model_final.joblib"
    )
    train_parser.set_defaults(func=command_train)

    predict_parser = subparsers.add_parser(
        "predict", help="Predict a classification map from a feature stack"
    )
    predict_parser.add_argument("--model", required=True)
    predict_parser.add_argument("--stack-path", required=True)
    predict_parser.add_argument(
        "--output-path", default="results/classification_map.tif"
    )
    predict_parser.set_defaults(func=command_predict)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a trained classifier against a checkpoint"
    )
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.set_defaults(func=command_evaluate)

    visualize_parser = subparsers.add_parser(
        "visualize", help="Create a classification visualization"
    )
    visualize_parser.add_argument("--input-path", required=True)
    visualize_parser.add_argument("--lut-path", required=True)
    visualize_parser.add_argument(
        "--output-path", default="results/classification_map.png"
    )
    visualize_parser.add_argument("--id-column", default="class_id")
    visualize_parser.add_argument("--name-column", default="class_name")
    visualize_parser.add_argument("--title", default="Classification Map")
    visualize_parser.set_defaults(func=command_visualize)

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

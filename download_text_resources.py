#!/usr/bin/env python3

import argparse
from pathlib import Path

import nltk
from datasets import load_dataset


DEFAULT_GLUE_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
DEFAULT_NLTK_PACKAGES = [
    "punkt",
    "punkt_tab",
    "wordnet",
    "sentiwordnet",
    "omw-1.4",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
]


def download_glue_tasks(glue_dir: Path, tasks) -> None:
    glue_dir.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        print(f"[GLUE] Downloading {task}")
        dataset = load_dataset("glue", task)
        save_path = glue_dir / task
        dataset.save_to_disk(str(save_path))
        print(f"[GLUE] Saved to {save_path}")


def download_nltk_packages(nltk_dir: Path, packages) -> None:
    nltk_dir.mkdir(parents=True, exist_ok=True)
    for package in packages:
        print(f"[NLTK] Downloading {package}")
        nltk.download(package, download_dir=str(nltk_dir))
    print(f"[NLTK] Saved to {nltk_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download local GLUE datasets and NLTK resources used by this project."
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["glue", "nltk"],
        default=["glue", "nltk"],
        help="Text resources to download.",
    )
    parser.add_argument(
        "--glue_dir",
        type=str,
        default="./glue_local",
        help="Directory for GLUE datasets.",
    )
    parser.add_argument(
        "--glue_tasks",
        nargs="+",
        default=DEFAULT_GLUE_TASKS,
        help="GLUE tasks to download.",
    )
    parser.add_argument(
        "--nltk_dir",
        type=str,
        default="./nltk_data",
        help="Directory for NLTK resources.",
    )
    parser.add_argument(
        "--nltk_packages",
        nargs="+",
        default=DEFAULT_NLTK_PACKAGES,
        help="NLTK packages to download.",
    )
    args = parser.parse_args()

    if "glue" in args.targets:
        download_glue_tasks(Path(args.glue_dir).resolve(), args.glue_tasks)

    if "nltk" in args.targets:
        download_nltk_packages(Path(args.nltk_dir).resolve(), args.nltk_packages)

    print("[Done] Requested text resources downloaded successfully.")


if __name__ == "__main__":
    main()

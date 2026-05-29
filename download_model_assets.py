#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


ASSET_PRESETS = {
    "flan-t5-base": {
        "type": "single_repo",
        "repo_id": "google/flan-t5-base",
        "target_dir": Path("local_models") / "google" / "flan-t5-base",
        "required_files": ["config.json"],
    },
    "glue-loras": {
        "type": "multi_repo",
        "target_root": Path("best_LoRA"),
        "repos": [
            "Daxuxu36/T5-COLA-LoRA",
            "Daxuxu36/T5-SST2-LoRA",
            "Daxuxu36/T5-RTE-LoRA",
            "Daxuxu36/T5-QNLI-LoRA",
            "Daxuxu36/T5-QQP-LoRA",
            "Daxuxu36/T5-MRPC-LoRA",
            "Daxuxu36/T5-MNLI-LoRA",
        ],
        "required_files": ["adapter_config.json", "adapter_model.safetensors"],
    },
}


def ensure_required_files(target_dir: Path, required_files) -> None:
    missing = [str(target_dir / name) for name in required_files if not (target_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Download incomplete for {target_dir}. Missing files: {missing}")


def download_repo(repo_id: str, target_dir: Path, token: str | None = None) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Download] {repo_id}")
    print(f"[Save To] {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        token=token,
    )


def download_single_repo(repo_id: str, target_dir: Path, required_files, token: str | None = None) -> None:
    download_repo(repo_id, target_dir, token=token)
    ensure_required_files(target_dir, required_files)
    print(f"[Done] {repo_id}")


def download_asset(asset_name: str, output_root: Path, token: str | None = None) -> None:
    preset = ASSET_PRESETS[asset_name]

    if preset["type"] == "single_repo":
        target_dir = output_root / preset["target_dir"]
        download_single_repo(
            repo_id=preset["repo_id"],
            target_dir=target_dir,
            required_files=preset["required_files"],
            token=token,
        )
        return

    if preset["type"] == "multi_repo":
        target_root = output_root / preset["target_root"]
        target_root.mkdir(parents=True, exist_ok=True)
        failures = []

        for repo_id in preset["repos"]:
            target_dir = target_root / repo_id.split("/")[-1]
            try:
                download_single_repo(
                    repo_id=repo_id,
                    target_dir=target_dir,
                    required_files=preset["required_files"],
                    token=token,
                )
            except Exception as exc:
                print(f"[Failed] {repo_id}: {exc}")
                failures.append(repo_id)

        if failures:
            raise RuntimeError(f"Some repositories failed: {failures}")
        return

    raise ValueError(f"Unsupported asset preset: {asset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Hugging Face model assets used by this project."
    )
    parser.add_argument(
        "--assets",
        nargs="+",
        choices=sorted(ASSET_PRESETS.keys()),
        default=["flan-t5-base", "glue-loras"],
        help="Assets to download.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=".",
        help="Project root where asset directories will be created.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face token. You can also set HF_TOKEN in the environment.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print("Assets to download:")
    for asset in args.assets:
        print(f" - {asset}")

    try:
        for asset in args.assets:
            download_asset(asset, output_root=output_root, token=args.token)
    except Exception as exc:
        print(f"[Error] {exc}")
        sys.exit(1)

    print("[Done] All requested assets downloaded successfully.")


if __name__ == "__main__":
    main()

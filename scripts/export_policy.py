#!/usr/bin/env python3
"""Expose one trained step in the existing openarm-pipeline/policy release layout."""
import argparse
import dataclasses
import json
from pathlib import Path
import shutil

from post_train import Settings
from post_training.configuration import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Completed numeric training-step directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    settings = Settings.read(args.config)
    config = build(settings, "export")
    model = config.model
    if model.flash_chunk_size != 8 or model.flash_buffer_size != 2 or not model.flash_enabled:
        raise ValueError("Existing policy server requires the FLASH C8/N2 training recipe")
    if settings.raw["data"]["state_dim"] != 11 or settings.raw["data"]["action_dim"] != 20:
        raise ValueError("Existing NEXTAGE client requires state11/action20")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    stats = checkpoint / "assets/dataset/norm_stats.json"
    contract = checkpoint / "assets/post_training_contract.json"
    if not (checkpoint / "params/_METADATA").is_file() or not stats.is_file():
        raise ValueError("Missing committed parameters or normalization assets")
    expected = config.data.runtime_contract
    if not contract.is_file() or json.loads(contract.read_text()) != json.loads(json.dumps(expected)):
        raise ValueError("Training config does not match this checkpoint's saved contract")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "params").symlink_to(checkpoint / "params", target_is_directory=True)
    shutil.copy2(stats, output / "norm_stats.json")
    shutil.copy2(contract, output / "post_training_contract.json")
    (output / "model_config.json").write_text(json.dumps({
        "model_fields": dataclasses.asdict(model),
        "task_instruction": settings.raw["labels"]["task_instruction"],
    }, indent=2))
    print(f"Exported {output}; params links to {checkpoint} (keep the source checkpoint)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build a vLLM-loadable copy of FunAudioLLM/Fun-ASR-MLT-Nano-2512.

The official MLT repo only ships the native FunASR checkpoint (`model.pt` + `config.yaml`),
which vLLM cannot load. Its architecture is identical to Fun-ASR-Nano-2512 (the two
`config.yaml` differ only by `use_low_frame_rate`, which vLLM hard-codes), and FunAudioLLM
publishes a vLLM packaging of that model (`Fun-ASR-Nano-2512-vllm`). So the MLT package is:
    - the MLT weights, converted from `model.pt` to `model.safetensors`
    - config, tokenizer and preprocessor files borrowed from `Fun-ASR-Nano-2512-vllm`

Every converted tensor is checked against the key names and shapes of the official vLLM
package, so a layout mismatch fails here instead of at serving time. The script is
idempotent: it exits early when the output already holds a converted model.
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import get_safetensors_metadata, hf_hub_download
from safetensors.torch import save_file

MLT_REPO = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
VLLM_REPO = "FunAudioLLM/Fun-ASR-Nano-2512-vllm"
# Non-weight files of the vLLM packaging (they carry the `<|AUDIO|>` token and FunASR processor)
VLLM_FILES = ["config.json", "generation_config.json", "preprocessor_config.json",
              "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "multilingual.tiktoken"]


def expected_layout() -> dict:
    """
    Read tensor names and shapes of the official vLLM package from its safetensors header.

    Returns:
        dict: Mapping of tensor name to shape, without downloading the weights
    """
    metadata = get_safetensors_metadata(VLLM_REPO)
    layout = {}
    for file_metadata in metadata.files_metadata.values():
        for name, info in file_metadata.tensors.items():
            layout[name] = list(info.shape)
    return layout


def convert(output_dir: Path) -> None:
    """
    Convert the MLT checkpoint and assemble a vLLM model directory.

    Args:
        output_dir (Path): Directory that receives the vLLM-loadable model
    """
    weights_path = output_dir / "model.safetensors"
    if weights_path.exists() and (output_dir / "config.json").exists():
        print(f"{output_dir} already contains a converted model, skipping")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    for filename in VLLM_FILES:
        shutil.copy(hf_hub_download(VLLM_REPO, filename), output_dir / filename)

    print(f"Downloading {MLT_REPO}/model.pt ...")
    checkpoint = torch.load(hf_hub_download(MLT_REPO, "model.pt"),
                            map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise SystemExit("checkpoint contains non-tensor state-dict values")

    # The MLT weights must fit the exact layout vLLM's FunASR implementation loads
    layout = expected_layout()
    missing = sorted(layout.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - layout.keys())
    mismatched = sorted(name for name in layout.keys() & state_dict.keys()
                        if list(state_dict[name].shape) != layout[name])
    if missing or unexpected or mismatched:
        raise SystemExit(f"layout mismatch with {VLLM_REPO}: missing={missing[:10]} "
                         f"unexpected={unexpected[:10]} shape={mismatched[:10]}")

    # safetensors refuses tensors that share storage (e.g. tied embeddings)
    state_dict = {name: tensor.contiguous().clone() for name, tensor in state_dict.items()}
    # Write under a temporary name so an interrupted run is never mistaken for a finished one
    tmp_path = weights_path.with_suffix(".tmp")
    save_file(state_dict, tmp_path)
    tmp_path.rename(weights_path)
    (output_dir / "MODEL_PROVENANCE.json").write_text(json.dumps({
        "weights": f"{MLT_REPO}/model.pt",
        "config_and_tokenizer": VLLM_REPO,
        "tensor_count": len(state_dict),
    }, indent=2))
    print(f"Wrote {len(state_dict)} tensors to {weights_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("output_dir", type=Path, help="Output model directory")
    convert(parser.parse_args().output_dir)


if __name__ == "__main__":
    main()

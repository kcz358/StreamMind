import sys

from huggingface_hub import HfApi


token = sys.stdin.readline().strip()
if not token:
    raise SystemExit("missing HF token on stdin")

HfApi(token=token).upload_file(
    path_or_fileobj=(
        "/data/kaichen/StreamMind/"
        "p16_4b_stage2_train_all_20260715-121029/"
        "checkpoint-417994/model.safetensors"
    ),
    path_in_repo="model.safetensors",
    repo_id="Mage-VL/Mage-VL-StreamMind",
    repo_type="model",
    commit_message="Update StreamMind weights",
)

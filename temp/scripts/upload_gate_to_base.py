import os

from huggingface_hub import CommitOperationAdd, HfApi


root = "/data/v-kaichen/StreamMind/temp/mage_vl_streammind_release"
api = HfApi(token=os.environ["HF_TOKEN"])
commit = api.create_commit(
    repo_id="Mage-VL/Mage-VL-Base",
    repo_type="model",
    commit_message="Add optional StreamMind gate",
    operations=[
        CommitOperationAdd(
            path_in_repo="modeling_mage_vl.py",
            path_or_fileobj=os.path.join(root, "modeling_mage_vl.py"),
        ),
        CommitOperationAdd(
            path_in_repo="streammind_gate.py",
            path_or_fileobj=os.path.join(root, "streammind_gate.py"),
        ),
        CommitOperationAdd(
            path_in_repo="streammind_gate.safetensors",
            path_or_fileobj=os.path.join(root, "streammind_gate.safetensors"),
        ),
    ],
)
print(commit.commit_url)

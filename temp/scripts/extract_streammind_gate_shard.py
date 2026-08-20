import json

from safetensors import safe_open
from safetensors.torch import save_file


source = (
    "/data/kaichen/StreamMind/p16_4b_stage2_train_all_20260715-121029/"
    "checkpoint-417994/model.safetensors"
)
output = "/tmp/model-00003-of-00003.safetensors"
prefix = "model.mm_projector."

weights = {}
with safe_open(source, framework="pt", device="cpu") as f:
    for key in f.keys():
        if key.startswith(prefix):
            new_key = "model.streammind_gate." + key[len(prefix):]
            weights[new_key] = f.get_tensor(key)

save_file(weights, output, metadata={"format": "pt"})
print(json.dumps({"output": output, "keys": len(weights), "bytes": sum(x.numel() * x.element_size() for x in weights.values())}))

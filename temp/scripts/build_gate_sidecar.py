import json

from safetensors import safe_open
from safetensors.torch import save_file


source = (
    "/data/kaichen/StreamMind/p16_4b_stage2_train_all_20260715-121029/"
    "checkpoint-417994/model.safetensors"
)
output = "/tmp/mage_vl_streammind_release/streammind_gate.safetensors"
prefix = "model.mm_projector."

weights = {}
with safe_open(source, framework="pt", device="cpu") as f:
    for key in f.keys():
        if key.startswith(prefix):
            weights[key[len(prefix):]] = f.get_tensor(key)

save_file(weights, output, metadata={"format": "pt"})

config_path = "/tmp/mage_vl_streammind_release/config.json"
with open(config_path) as f:
    config = json.load(f)
config["architectures"] = ["MageVLForConditionalGeneration"]
config["auto_map"]["AutoModelForCausalLM"] = (
    "modeling_mage_vl.MageVLStreamMindForConditionalGeneration"
)
config["auto_map"]["AutoModelForImageTextToText"] = (
    "modeling_mage_vl.MageVLStreamMindForConditionalGeneration"
)
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

print({"keys": len(weights), "output": output})

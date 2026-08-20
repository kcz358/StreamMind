import json
import os


root = "/tmp/mage_vl_streammind_release"
index_path = os.path.join(root, "model.safetensors.index.json")
gate_shard = "model-00003-of-00003.safetensors"

with open(index_path) as f:
    index = json.load(f)
for key in list(index["weight_map"]):
    index["weight_map"][key] = index["weight_map"][key].replace("of-00002", "of-00003")

from safetensors import safe_open

gate_path = os.path.join(root, gate_shard)
with safe_open(gate_path, framework="pt") as f:
    gate_keys = list(f.keys())
    gate_size = sum(f.get_tensor(k).numel() * f.get_tensor(k).element_size() for k in gate_keys)
for key in gate_keys:
    index["weight_map"][key] = gate_shard
index["metadata"]["total_size"] += gate_size

os.rename(os.path.join(root, "model-00001-of-00002.safetensors"), os.path.join(root, "model-00001-of-00003.safetensors"))
os.rename(os.path.join(root, "model-00002-of-00002.safetensors"), os.path.join(root, "model-00002-of-00003.safetensors"))
with open(index_path, "w") as f:
    json.dump(index, f, indent=2, sort_keys=True)

with open(os.path.join(root, "config.json")) as f:
    config = json.load(f)
config["architectures"] = ["MageVLStreamMindForConditionalGeneration"]
config["auto_map"]["AutoModelForCausalLM"] = "modeling_mage_vl.MageVLStreamMindForConditionalGeneration"
config["auto_map"]["AutoModelForImageTextToText"] = "modeling_mage_vl.MageVLStreamMindForConditionalGeneration"
with open(os.path.join(root, "config.json"), "w") as f:
    json.dump(config, f, indent=2)

print(json.dumps({"gate_keys": len(gate_keys), "gate_size": gate_size, "total_size": index["metadata"]["total_size"]}))

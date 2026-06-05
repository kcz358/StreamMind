import os 
import re
from typing import Dict, Optional, Sequence, List
import torch


from streammind.constants import NUM_FRAMES, IGNORE_INDEX, MMODAL_TOKEN_INDEX, DEFAULT_MMODAL_TOKEN, DEFAULT_MMODAL_START_TOKEN, DEFAULT_MMODAL_END_TOKEN
from streammind.mm_utils import tokenizer_MMODAL_token, tokenizer_image_token, expand2square, process_video, process_image


def extract_video_half(video_data_path):
    # Extract the filename from the path
    filename = os.path.basename(video_data_path)
    
    match = re.match(r"(\d+)_\d+p\.mkv", filename)
    if match:
        return int(match.group(1))
    return None


def trans_video_2_json(file_paths,data_type):
    new_path = file_paths.replace("features_video", "dataset/MatchTime/" + data_type)
    if "1_224p.mkv" in new_path:
        new_path = new_path.replace("1_224p.mkv", "Labels-caption.json")
    elif "2_224p.mkv" in new_path:
        new_path = new_path.replace("2_224p.mkv", "Labels-caption.json")
    return new_path


def find_video_files(root_path, target_filenames):
    paths = []
    # Traverse the directory structure
    for dirpath, _, filenames in os.walk(root_path):
        # Check if either of the target files is in the current directory
        for target_filename in target_filenames:
            if target_filename in filenames:
                # Append the full path of the found file
                paths.append(os.path.join(dirpath, target_filename))
    return paths




def preprocess_llama_2_soccer_cls(
    caption_data,video_data,timestamp,tokenizer,data_type
) -> Dict:
    return {"labels" :None,
        "video":None,
        "input_ids" : torch.tensor([0]),
        "timestamp" : timestamp,
        "caption_info":caption_data,
        "video_path":video_data,
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"cls"}

def preprocess_qwen3_soccer(
    caption_data,video_data,timestamp,tokenizer,data_type
) -> Dict:
    """Stage-1 prompt construction for OV2 (Qwen3 backbone).

    Same high-level structure as preprocess_llama_2_soccer but using Qwen3's
    chat template (<|im_start|>...<|im_end|>) and matching label-mask
    boundaries. The literal `<video>` is replaced by MMODAL_TOKEN_INDEX['VIDEO']
    (-201) via tokenizer_MMODAL_token, identically to the original.
    """
    MODAL_list = ['VIDEO']
    SYS_PROMPT = (
        "A chat between a curious user and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the user's questions."
    )
    USER_PREFIX = "Please describe the video content in detail based on the provided information."

    messages = [{'role': 'system', 'content': SYS_PROMPT}]
    for i, caption in enumerate(caption_data):
        user_msg = (USER_PREFIX + '<video>\n') if i == 0 else '<video>\n'
        messages.append({'role': 'user', 'content': user_msg})
        messages.append({'role': 'assistant', 'content': caption})

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = torch.stack(
        [tokenizer_MMODAL_token(prompt, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[0]], return_tensors='pt')],
        dim=0,
    )

    targets = input_ids.clone()
    targets[:] = IGNORE_INDEX

    for k in range(len(caption_data)):
        prefix_msgs = messages[: 1 + 2 * k + 1]
        full_msgs = messages[: 1 + 2 * k + 2]
        prefix_str = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True)
        full_str = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)
        prefix_ids = tokenizer_MMODAL_token(prefix_str, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[0]], return_tensors='pt')
        full_ids = tokenizer_MMODAL_token(full_str, tokenizer, MMODAL_TOKEN_INDEX[MODAL_list[0]], return_tensors='pt')
        prefix_len = prefix_ids.size(0)
        full_len = full_ids.size(0)
        targets[0, prefix_len:full_len] = input_ids[0, prefix_len:full_len]

    return {"labels" :targets,
        "video":None,
        "input_ids" : input_ids,
        "timestamp" : timestamp,
        "caption_info":caption_data,
        "video_path":video_data,
        "past_review_caption":None,
        "data_type":data_type,
        "model_type":"llm"}

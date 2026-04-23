"""
Inference pipeline for SRDC-GPT ransomware detection.
Handles preprocessing → tokenization → model forward → output decoding.
"""

import torch
import torch.nn.functional as F
from preprocessing import tokenize_input
from model_loader import IDX_TO_FAMILY, IDX_TO_BINARY


@torch.no_grad()
def predict_family(model, tokenizer, raw_text: str) -> dict:
    """
    Run family classification on raw behavioral text.

    Returns:
        label        : predicted family name (str)
        confidence   : probability of predicted class (float 0-1)
        all_probs    : {family_name: probability} for all 11 classes
        tokens       : list of input_ids (for debug display)
        section_texts: list of 7 decoded section strings
    """
    enc = tokenize_input(raw_text, tokenizer)

    input_ids   = enc['input_ids'].unsqueeze(0)       # (1, 512)
    attn_mask   = enc['attention_mask'].unsqueeze(0)  # (1, 512)
    sec_masks   = enc['section_masks'].unsqueeze(0)   # (1, 7, 512)

    batch = {
        'input_ids'     : input_ids,
        'attention_mask': attn_mask,
        'section_masks' : sec_masks,
    }

    logits = model(batch)                             # (1, 11)
    probs  = F.softmax(logits, dim=-1).squeeze(0)     # (11,)

    pred_idx   = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())

    all_probs = {
        IDX_TO_FAMILY[i]: float(probs[i].item())
        for i in range(len(probs))
    }

    return {
        'label'        : IDX_TO_FAMILY[pred_idx],
        'confidence'   : confidence,
        'all_probs'    : all_probs,
        'tokens'       : enc['input_ids'].tolist(),
        'section_texts': enc['section_texts'],
    }


@torch.no_grad()
def predict_zeroday(model, tokenizer, raw_text: str) -> dict:
    """
    Run zero-day detection on raw behavioral text.

    Returns:
        label        : 'Malicious (Ransomware)' or 'Benign (Goodware)'
        is_malicious : bool
        probability  : P(ransomware) (float 0-1)
        all_probs    : {label: probability} for both classes
        tokens       : list of input_ids (for debug display)
        section_texts: list of 7 decoded section strings
    """
    enc = tokenize_input(raw_text, tokenizer)

    input_ids = enc['input_ids'].unsqueeze(0)
    attn_mask = enc['attention_mask'].unsqueeze(0)
    sec_masks = enc['section_masks'].unsqueeze(0)

    batch = {
        'input_ids'     : input_ids,
        'attention_mask': attn_mask,
        'section_masks' : sec_masks,
    }

    logits = model(batch)                             # (1, 2)
    probs  = F.softmax(logits, dim=-1).squeeze(0)     # (2,)

    pred_idx      = int(probs.argmax().item())
    prob_malicious = float(probs[1].item())

    all_probs = {
        IDX_TO_BINARY[i]: float(probs[i].item())
        for i in range(len(probs))
    }

    return {
        'label'        : IDX_TO_BINARY[pred_idx],
        'is_malicious' : pred_idx == 1,
        'probability'  : prob_malicious,
        'all_probs'    : all_probs,
        'tokens'       : enc['input_ids'].tolist(),
        'section_texts': enc['section_texts'],
    }

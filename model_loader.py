"""
Model architectures and loading utilities for SRDC-GPT ransomware detection.
Architectures exactly match the training notebooks.
"""

import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

# ── Label maps ───────────────────────────────────────────────────────────────
# V5: Goodware (family 0) is now class 0; ransomware families 1-11 → idx 1-11
IDX_TO_FAMILY = {
    0 : 'Goodware',
    1 : 'Citroni',
    2 : 'CryptLocker',
    3 : 'CryptoWall',
    4 : 'Kollah',
    5 : 'Kovter',
    6 : 'Locker',
    7 : 'Matsnu',
    8 : 'PGPCODER',
    9 : 'Reveton',
    10: 'TeslaCrypt',
    11: 'Trojan-Ransom',
}

IDX_TO_BINARY = {
    0: 'Benign (Goodware)',
    1: 'Malicious (Ransomware)',
}

NUM_FAMILY_CLASSES = 12   # V5: 0=Goodware + 11 ransomware families
NUM_BINARY_CLASSES = 2
HIDDEN_SIZE        = 768
NUM_SECTIONS       = 7
POOL_DIM           = HIDDEN_SIZE * NUM_SECTIONS  # 5376

# GPT-2 layers fine-tuned per model
FAMILY_UNFREEZE_LAYERS  = 3   # h9, h10, h11
ZERODAY_UNFREEZE_LAYERS = 6   # h6, h7, h8, h9, h10, h11


# ── Shared pooling module ────────────────────────────────────────────────────
class SectionMeanPool(nn.Module):
    """
    Section-aware mean pooling. Exact replica from training notebooks.
    Computes mean of hidden states for each of the 7 feature sections,
    then concatenates → (batch, 7 * 768) = (batch, 5376).
    """
    def forward(self, hidden_states: torch.Tensor, section_masks: torch.Tensor) -> torch.Tensor:
        section_vecs = []
        for s in range(section_masks.size(1)):
            mask     = section_masks[:, s, :]           # (batch, seq_len)
            mask_exp = mask.unsqueeze(-1).float()        # (batch, seq_len, 1)
            sum_h    = (hidden_states * mask_exp).sum(dim=1)          # (batch, hidden)
            count    = mask_exp.sum(dim=1).clamp(min=1e-9)            # (batch, 1)
            section_vecs.append(sum_h / count)
        return torch.cat(section_vecs, dim=-1)           # (batch, 5376)


# ── Family classifier (multiclass) ──────────────────────────────────────────
class FamilyClassifier(nn.Module):
    """
    Ransomware family classifier.
    Architecture: GPT-2 → SectionMeanPool → Linear(5376, 11)
    Fine-tuned layers: h9, h10, h11 + ln_f
    """
    def __init__(self, gpt_model):
        super().__init__()
        self.gpt        = gpt_model
        self.pool       = SectionMeanPool()
        self.classifier = nn.Linear(POOL_DIM, NUM_FAMILY_CLASSES)

    def forward(self, batch: dict) -> torch.Tensor:
        out    = self.gpt(
            input_ids            = batch['input_ids'],
            attention_mask       = batch['attention_mask'],
            output_hidden_states = True,
        )
        hidden   = out.hidden_states[-1]                 # (batch, 512, 768)
        combined = self.pool(hidden, batch['section_masks'])
        return self.classifier(combined)                 # (batch, 11)


# ── Zero-day binary classifier ───────────────────────────────────────────────
class ZeroDayClassifier(nn.Module):
    """
    Zero-day ransomware binary detector.
    Architecture: GPT-2 → SectionMeanPool → MLP(5376→512→2)
    Fine-tuned layers: h6, h7, h8, h9, h10, h11 + ln_f
    """
    def __init__(self, gpt_model, dropout: float = 0.4):
        super().__init__()
        self.gpt  = gpt_model
        self.pool = SectionMeanPool()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(POOL_DIM, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, NUM_BINARY_CLASSES),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        out    = self.gpt(
            input_ids            = batch['input_ids'],
            attention_mask       = batch['attention_mask'],
            output_hidden_states = True,
        )
        hidden   = out.hidden_states[-1]
        combined = self.pool(hidden, batch['section_masks'])
        return self.classifier(combined)                 # (batch, 2)


# ── Loading utilities ────────────────────────────────────────────────────────
def _load_gpt2(device: str = 'cpu') -> AutoModelForCausalLM:
    """Load base GPT-2 (124M). Used as backbone for both models."""
    gpt = AutoModelForCausalLM.from_pretrained('gpt2', output_hidden_states=True)
    return gpt.to(device)


def _apply_finetuned_layers(gpt_model, state: dict, n_layers: int):
    """
    Inject fine-tuned GPT-2 layer weights into the base model.
    Keys in state: h{i} for i in range(12-n_layers, 12), plus 'ln_f'.
    """
    start = 12 - n_layers
    for i in range(start, 12):
        key = f'h{i}'
        if key in state:
            gpt_model.transformer.h[i].load_state_dict(state[key])
    if 'ln_f' in state:
        gpt_model.transformer.ln_f.load_state_dict(state['ln_f'])


def load_family_model(weights_path: str, device: str = 'cpu') -> FamilyClassifier:
    """
    Load the family classification model from a checkpoint.

    Expected checkpoint format (family_model.pth):
        {
            'h9': ..., 'h10': ..., 'h11': ...,   # fine-tuned GPT-2 layers
            'ln_f': ...,                            # fine-tuned layer norm
            'classifier': ...,                      # Linear(5376, 11) state_dict
        }
    """
    gpt   = _load_gpt2(device)
    model = FamilyClassifier(gpt).to(device)

    state = torch.load(weights_path, map_location=device)

    # Accept both wrapped and flat state dicts
    if 'model_state_dict' in state:
        state = state['model_state_dict']

    # Apply fine-tuned GPT-2 layers
    _apply_finetuned_layers(model.gpt, state, FAMILY_UNFREEZE_LAYERS)

    # Apply classifier head
    clf_state = state.get('classifier', state)
    if isinstance(clf_state, dict) and 'weight' in clf_state:
        model.classifier.load_state_dict(clf_state)

    model.eval()
    return model


def load_zeroday_model(weights_path: str, device: str = 'cpu') -> ZeroDayClassifier:
    """
    Load the zero-day detection model from a checkpoint.

    Expected checkpoint format (zeroday_model.pth):
        {
            'h6': ..., 'h7': ..., 'h8': ...,
            'h9': ..., 'h10': ..., 'h11': ...,   # fine-tuned GPT-2 layers
            'ln_f': ...,                            # fine-tuned layer norm
            'classifier': ...,                      # MLP state_dict
        }
    """
    gpt   = _load_gpt2(device)
    model = ZeroDayClassifier(gpt).to(device)

    state = torch.load(weights_path, map_location=device)

    if 'model_state_dict' in state:
        state = state['model_state_dict']

    _apply_finetuned_layers(model.gpt, state, ZERODAY_UNFREEZE_LAYERS)

    clf_state = state.get('classifier', None)
    if clf_state is not None and isinstance(clf_state, dict):
        model.classifier.load_state_dict(clf_state)

    model.eval()
    return model


def download_from_gdrive(file_id: str, dest_path: str) -> bool:
    """Download a file from Google Drive using gdown. Returns True on success."""
    try:
        import gdown
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)
        return os.path.exists(dest_path)
    except Exception as e:
        print(f"Download failed: {e}")
        return False

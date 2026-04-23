"""
Exact replica of training-time preprocessing from ransomware detection notebooks.
DO NOT modify without also updating the training notebooks.
"""

import re
import torch
from transformers import AutoTokenizer

# ── Constants (match training exactly) ──────────────────────────────────────
MAX_LEN         = 512
PER_FEATURE_MAX = MAX_LEN // 7   # 73 (same as MAX_LEN // len(FEATURE_COLS))

FEATURE_COLS = [
    'apiFeatures', 'dropFeatures', 'regFeatures',
    'filesFeatures', 'filesEXTFeatures', 'dirFeatures', 'strFeatures',
]

FEATURE_SEPARATORS = {
    'apiFeatures'     : '[API]',
    'dropFeatures'    : '[DROP]',
    'regFeatures'     : '[REG]',
    'filesFeatures'   : '[FILES]',
    'filesEXTFeatures': '[EXT]',
    'dirFeatures'     : '[DIR]',
    'strFeatures'     : '[STR]',
}

# Prefix in user's pasted text → feature column
PREFIX_TO_COL = {
    'API'     : 'apiFeatures',
    'REG'     : 'regFeatures',
    'FILES'   : 'filesFeatures',
    'DIR'     : 'dirFeatures',
    'DROP'    : 'dropFeatures',
    'STR'     : 'strFeatures',
    'EXT'     : 'filesEXTFeatures',
    'FILESEXT': 'filesEXTFeatures',
}

# ── Semantic processing (exact from notebooks) ───────────────────────────────
VERB_MAP = {
    'OPENED'   : 'Opened',     'READ'     : 'Read',
    'WRITE'    : 'Written to', 'WRITTEN'  : 'Written to',
    'DELETED'  : 'Deleted',    'CREATED'  : 'Created',
    'ENUMERATED': 'Enumerated','MOVED'    : 'Moved',
    'RENAMED'  : 'Renamed',    'COPIED'   : 'Copied',
    'EXECUTED' : 'Executed',   'MODIFIED' : 'Modified',
    'CLOSED'   : 'Closed',     'DROPPED'  : 'Dropped',
}

FEATURE_TYPE_MAP = {
    'API'  : 'Windows API', 'REG'   : 'registry',
    'FILES': 'file',        'DIR'   : 'directory',
    'DROP' : 'file',        'STR'   : 'string',
}


def camel_to_phrase(name: str) -> str:
    """CamelCase → lowercase phrase. Exact replica of notebook function."""
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', name)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return spaced.lower().strip()


def process_single_token(token: str) -> str:
    """
    Convert a raw behavioral token to natural language.
    Exact replica of process_single_token() from training notebooks.
    """
    token = token.strip()
    if not token:
        return token
    parts     = token.split(':', 2)
    feat_type = parts[0].upper()

    if feat_type == 'API' and len(parts) >= 2:
        func_name = re.sub(r'[WA]$', '', parts[1])
        return f"Windows API: {camel_to_phrase(func_name)}"

    if len(parts) == 3:
        verb = VERB_MAP.get(parts[1].upper(), parts[1].lower())
        noun = FEATURE_TYPE_MAP.get(feat_type, feat_type.lower())
        return f"{verb} {noun} {parts[2]}"

    if len(parts) == 2:
        noun = FEATURE_TYPE_MAP.get(feat_type, feat_type.lower())
        return f"{noun}: {parts[1]}"

    return token


def process_feature_text(raw_text: str) -> str:
    """
    Convert a feature column's raw text to processed NL text.
    Exact replica of process_feature_text() from training notebooks.
    """
    if not isinstance(raw_text, str) or raw_text.strip() in ('', 'nan', 'none'):
        return ''
    tokens    = re.split(r'[\s,;]+', raw_text.strip())
    processed = [process_single_token(t) for t in tokens if t]
    return ' '.join(processed)


# ── User input parsing ────────────────────────────────────────────────────────
def parse_user_input(raw_text: str) -> dict:
    """
    Parse user's pasted behavioral sequence into 7 feature columns.

    Input format (one token per line):
        API:CreateRemoteThread
        REG:OPENED:HKEY_LOCAL_MACHINE\\SOFTWARE
        FILES:READ:C:\\Users\\victim\\file.docx
        DIR:ENUMERATED:C:\\Users\\victim\\Desktop
        STR:ransom_note.txt
    """
    features = {col: [] for col in FEATURE_COLS}

    for line in raw_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        prefix = line.split(':')[0].upper()
        col    = PREFIX_TO_COL.get(prefix)
        if col:
            features[col].append(line)

    return {col: ' '.join(tokens) for col, tokens in features.items()}


# ── Tokenization pipeline (matches tokenize_row from training notebooks) ─────
def tokenize_input(raw_text: str, tokenizer) -> dict:
    """
    Full preprocessing → tokenization pipeline.
    Replicates tokenize_row() from training notebooks exactly.

    Returns dict with:
        input_ids      : (512,) LongTensor
        attention_mask : (512,) LongTensor
        section_masks  : (7, 512) BoolTensor
        section_texts  : list[str] — decoded section texts (for debug display)
    """
    features = parse_user_input(raw_text)

    parts        = []
    section_lens = []

    for col in FEATURE_COLS:
        sep  = FEATURE_SEPARATORS[col]
        raw  = features.get(col, '')
        proc = process_feature_text(raw)

        if proc:
            # Truncate to per-feature budget (PER_FEATURE_MAX - 1, exact match)
            ids  = tokenizer(
                proc,
                truncation=True,
                max_length=PER_FEATURE_MAX - 1,
                add_special_tokens=False,
            )['input_ids']
            proc = tokenizer.decode(ids, skip_special_tokens=True)
            section_text = f"{sep} {proc}"
        else:
            section_text = sep

        parts.append(section_text)
        section_lens.append(len(
            tokenizer(section_text, add_special_tokens=False)['input_ids']
        ))

    # Final tokenization of the concatenated sequence
    enc = tokenizer(
        ' '.join(parts),
        truncation=True,
        max_length=MAX_LEN,
        padding='max_length',
        return_tensors='pt',
    )
    input_ids      = enc['input_ids'].squeeze(0)       # (512,)
    attention_mask = enc['attention_mask'].squeeze(0)  # (512,)

    # Build section masks (same logic as training)
    section_masks = []
    cumulative    = 0
    for sec_len in section_lens:
        mask  = torch.zeros(MAX_LEN, dtype=torch.bool)
        start = cumulative
        end   = min(cumulative + sec_len, MAX_LEN)
        if start < MAX_LEN:
            mask[start:end] = True
        mask = mask & attention_mask.bool()
        section_masks.append(mask)
        cumulative += sec_len

    while len(section_masks) < len(FEATURE_COLS):
        section_masks.append(torch.zeros(MAX_LEN, dtype=torch.bool))

    return {
        'input_ids'     : input_ids,
        'attention_mask': attention_mask,
        'section_masks' : torch.stack(section_masks),  # (7, 512)
        'section_texts' : parts,
    }


def load_tokenizer(tokenizer_path: str = None) -> AutoTokenizer:
    """
    Load the GPT-2 tokenizer with pad_token = eos_token (matches training).
    Falls back to 'gpt2' from HuggingFace if no local path given.
    """
    import os
    if tokenizer_path and os.path.exists(tokenizer_path):
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tok = AutoTokenizer.from_pretrained('gpt2')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

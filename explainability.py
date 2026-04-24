"""
Explainability for SRDC-GPT predictions.

Two complementary approaches:
  1. Section ablation  — zero out each of the 7 feature channels one at a time
                         and measure the probability drop.  Tells you which
                         channel drove the decision.
  2. Indicator scan    — rule-based keyword search for known ransomware /
                         goodware patterns in the raw input text.
"""

import torch
import torch.nn.functional as F
import numpy as np
from preprocessing import tokenize_input

SECTION_NAMES = ['API', 'DROP', 'REG', 'FILES', 'EXT', 'DIR', 'STR']

# ── Known ransomware behavioural indicators ──────────────────────────────────
RANSOMWARE_FLAGS = [
    (
        'Encryption APIs',
        ['CryptEncrypt', 'BCryptEncrypt', 'CryptGenRandom', 'CryptAcquireContext',
         'RtlEncryptMemory', 'SystemFunction036', 'CryptCreateHash', 'CryptHashData'],
    ),
    (
        'Process injection',
        ['CreateRemoteThread', 'VirtualAllocEx', 'WriteProcessMemory',
         'NtCreateThreadEx', 'NtAllocateVirtualMemory', 'ZwCreateThreadEx'],
    ),
    (
        'Shadow copy / backup deletion',
        ['vssadmin', 'wbadmin', 'bcdedit', 'shadowcopy', 'delete shadows',
         'resize shadowstorage'],
    ),
    (
        'Startup persistence write',
        ['CurrentVersion\\Run', 'CurrentVersion\\RunOnce',
         'CurrentVersion\\RunServices', 'Winlogon\\Shell'],
    ),
    (
        'Encrypted file extensions',
        ['.locked', '.encrypted', '.crypt', '.cerber', '.zepto',
         '.wallet', '.wnry', '.wncry', '.wcry', '.enc'],
    ),
    (
        'Ransom note strings',
        ['ransom', 'decrypt', 'bitcoin', 'wallet', 'payment',
         '.onion', 'tor2web', 'your files', 'restore files',
         'recover files', 'pay', 'deadline'],
    ),
    (
        'Process / privilege enumeration',
        ['CreateToolhelp32Snapshot', 'Process32First', 'Process32Next',
         'AdjustTokenPrivileges', 'LookupPrivilegeValue', 'SeDebugPrivilege'],
    ),
    (
        'Network C2 activity',
        ['HttpSendRequest', 'InternetOpenUrl', 'WSASend',
         'WSAConnect', 'URLDownloadToFile', 'WinHttpSendRequest'],
    ),
]

# ── Known goodware behavioural indicators ────────────────────────────────────
GOODWARE_FLAGS = [
    (
        'UI / application framework APIs',
        ['CreateWindow', 'ShowWindow', 'RegisterClass', 'DialogBox',
         'MessageBox', 'DefWindowProc', 'TranslateMessage', 'DispatchMessage'],
    ),
    (
        'Font / resource access',
        ['.ttf', '.fon', 'Windows\\Fonts', 'LoadString',
         'FindResource', 'LoadResource', 'LockResource'],
    ),
    (
        'Standard DLL / library loading',
        ['LoadLibrary', 'FreeLibrary', 'GetProcAddress', '.dll'],
    ),
    (
        'Config / preference management',
        ['config.', 'settings.', 'preferences.', '.xml', '.json',
         '.ini', 'AppData\\Roaming', 'AppData\\Local'],
    ),
    (
        'Normal memory management',
        ['HeapAlloc', 'HeapFree', 'HeapCreate', 'GlobalAlloc',
         'LocalAlloc', 'VirtualFree'],
    ),
    (
        'Application / version strings',
        ['version', '_loaded', '_enabled', '_updated', '_restored',
         'autosave', 'user_preferences', 'cache_updated', 'update_check'],
    ),
    (
        'System information reads (read-only)',
        ['GetSystemInfo', 'GetComputerName', 'GetUserName',
         'GetSystemTime', 'GetVersion', 'GetWindowsDirectory'],
    ),
]


# ── Indicator scanner ────────────────────────────────────────────────────────
def scan_indicators(raw_text: str) -> tuple:
    """
    Scan raw input for ransomware and goodware indicator patterns.
    Returns (ransom_hits, goodware_hits) where each item is (label, [matched]).
    """
    text_lower = raw_text.lower()

    ransom_hits = []
    for label, patterns in RANSOMWARE_FLAGS:
        matched = [p for p in patterns if p.lower() in text_lower]
        if matched:
            ransom_hits.append((label, matched[:4]))   # cap display at 4

    goodware_hits = []
    for label, patterns in GOODWARE_FLAGS:
        matched = [p for p in patterns if p.lower() in text_lower]
        if matched:
            goodware_hits.append((label, matched[:4]))

    # Mass file deletion heuristic
    deleted = [l for l in raw_text.splitlines() if 'FILES:DELETED' in l.upper()]
    if len(deleted) >= 4:
        ransom_hits.append(('Mass file deletion', [f'{len(deleted)} DELETE events']))

    # Temp-file cleanup heuristic (created AND deleted → goodware sign)
    created_paths = {
        l.split(':', 2)[-1].strip()
        for l in raw_text.splitlines()
        if 'FILES:CREATED' in l.upper()
    }
    deleted_paths = {
        l.split(':', 2)[-1].strip()
        for l in raw_text.splitlines()
        if 'FILES:DELETED' in l.upper()
    }
    if created_paths & deleted_paths:
        goodware_hits.append(('Temp file create-then-delete (cleanup)', []))

    return ransom_hits, goodware_hits


# ── Ablation-based section contribution ──────────────────────────────────────
@torch.no_grad()
def compute_section_contributions(
    model, tokenizer, raw_text: str, pred_class: int
) -> np.ndarray:
    """
    For each of the 7 feature sections, zero out its mask and measure
    how much P(pred_class) drops.

    contribution[s] > 0  → section pushed TOWARD the prediction
    contribution[s] < 0  → section pushed AGAINST the prediction
    """
    enc = tokenize_input(raw_text, tokenizer)

    input_ids = enc['input_ids'].unsqueeze(0)
    attn_mask = enc['attention_mask'].unsqueeze(0)
    sec_masks = enc['section_masks'].unsqueeze(0)   # (1, 7, 512)

    base_batch = {
        'input_ids'     : input_ids,
        'attention_mask': attn_mask,
        'section_masks' : sec_masks,
    }
    base_prob = float(
        F.softmax(model(base_batch), dim=-1).squeeze(0)[pred_class]
    )

    contributions = []
    for s in range(7):
        ablated = sec_masks.clone()
        ablated[0, s, :] = False          # zero out section s

        mod_batch = {
            'input_ids'     : input_ids,
            'attention_mask': attn_mask,
            'section_masks' : ablated,
        }
        mod_prob = float(
            F.softmax(model(mod_batch), dim=-1).squeeze(0)[pred_class]
        )
        contributions.append(base_prob - mod_prob)

    return np.array(contributions)


# ── Main explanation builder ──────────────────────────────────────────────────
def build_explanation(
    raw_text: str,
    result: dict,
    model,
    tokenizer,
    model_type: str,    # 'family' or 'zeroday'
) -> dict:
    """
    Returns a dict with:
      section_contributions : (7,) numpy array
      ransom_flags          : list of (label, [matched])
      goodware_flags        : list of (label, [matched])
      summary               : plain-text paragraph
      top_section           : name of the most influential section
    """
    # Resolve predicted class index
    if model_type == 'family':
        from model_loader import IDX_TO_FAMILY
        pred_class = next(
            k for k, v in IDX_TO_FAMILY.items() if v == result['label']
        )
    else:
        pred_class = 1 if result['is_malicious'] else 0

    contributions              = compute_section_contributions(model, tokenizer, raw_text, pred_class)
    ransom_flags, goodware_hits = scan_indicators(raw_text)

    top_idx     = int(np.argmax(contributions))
    top_section = SECTION_NAMES[top_idx]

    # Build plain-text summary
    parts = []

    if model_type == 'family':
        label = result['label']
        conf  = result['confidence']
        if label == 'Goodware':
            parts.append(
                f"The model classified this sample as Goodware with "
                f"{conf*100:.1f}% confidence."
            )
        else:
            parts.append(
                f"The model identified this as {label} ransomware with "
                f"{conf*100:.1f}% confidence."
            )
    else:
        if result['is_malicious']:
            parts.append(
                f"The zero-day detector flagged this sample as ransomware "
                f"({result['probability']*100:.1f}% malicious probability)."
            )
        else:
            parts.append(
                f"The zero-day detector classified this sample as benign "
                f"({(1-result['probability'])*100:.1f}% benign probability)."
            )

    if contributions[top_idx] > 0.005:
        parts.append(
            f"The {top_section} channel had the strongest influence on this "
            f"decision — removing it causes the largest drop in prediction confidence."
        )
    else:
        parts.append(
            "No single feature channel dominated the decision; "
            "the prediction is based on a broad combination of signals."
        )

    if ransom_flags:
        labels = ', '.join(lbl for lbl, _ in ransom_flags)
        parts.append(
            f"Ransomware-associated patterns were found: {labels}. "
            "These patterns correlate strongly with malicious behaviour in the training data."
        )
    else:
        parts.append(
            "No known ransomware indicator patterns were detected in the input."
        )

    if goodware_hits:
        labels = ', '.join(lbl for lbl, _ in goodware_hits)
        parts.append(
            f"Goodware-associated patterns were also present: {labels}."
        )

    # Overall lean
    if len(ransom_flags) > len(goodware_hits):
        parts.append(
            "On balance, the indicator profile leans malicious."
        )
    elif len(goodware_hits) > len(ransom_flags):
        parts.append(
            "On balance, the indicator profile leans benign."
        )
    else:
        parts.append(
            "The indicator profile is mixed or absent — "
            "the model relied primarily on learned sequence patterns."
        )

    return {
        'section_contributions': contributions,
        'ransom_flags'         : ransom_flags,
        'goodware_flags'       : goodware_hits,
        'summary'              : ' '.join(parts),
        'top_section'          : top_section,
    }

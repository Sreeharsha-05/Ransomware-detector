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

    contributions               = compute_section_contributions(model, tokenizer, raw_text, pred_class)
    ransom_flags, goodware_hits = scan_indicators(raw_text)

    top_idx     = int(np.argmax(contributions))
    top_section = SECTION_NAMES[top_idx]

    # ── Determine whether model and indicator scan agree ─────────────────────
    if model_type == 'family':
        model_says_malicious = (result['label'] != 'Goodware')
    else:
        model_says_malicious = result['is_malicious']

    indicator_leans_malicious = len(ransom_flags) > len(goodware_hits)
    indicator_leans_benign    = len(goodware_hits) > len(ransom_flags)
    indicators_present        = bool(ransom_flags or goodware_hits)

    # Conflict: model and indicators point in opposite directions
    conflict = indicators_present and (
        (model_says_malicious and indicator_leans_benign) or
        (not model_says_malicious and indicator_leans_malicious)
    )

    # ── Confidence level label ────────────────────────────────────────────────
    if model_type == 'zeroday':
        prob = result['probability'] if model_says_malicious else 1 - result['probability']
    else:
        prob = result['confidence']

    if prob >= 0.80:
        conf_label = "high confidence"
    elif prob >= 0.65:
        conf_label = "moderate confidence"
    else:
        conf_label = "low confidence — treat this result with caution"

    # ── Build summary ─────────────────────────────────────────────────────────
    parts = []

    # Verdict sentence
    if model_type == 'family':
        label = result['label']
        if label == 'Goodware':
            parts.append(
                f"The model classified this as Goodware ({prob*100:.1f}%, {conf_label})."
            )
        else:
            parts.append(
                f"The model identified this as {label} ransomware "
                f"({prob*100:.1f}%, {conf_label})."
            )
    else:
        if model_says_malicious:
            parts.append(
                f"The zero-day detector flagged this as ransomware "
                f"({result['probability']*100:.1f}% malicious probability, {conf_label})."
            )
        else:
            parts.append(
                f"The zero-day detector classified this as benign "
                f"({(1-result['probability'])*100:.1f}% benign probability, {conf_label})."
            )

    # Channel influence
    if contributions[top_idx] > 0.005:
        parts.append(
            f"The {top_section} channel had the strongest influence — "
            f"removing it causes the largest drop in prediction confidence."
        )
    else:
        parts.append(
            "No single channel dominated the decision; "
            "the model relied on a broad combination of signals."
        )

    # Indicator findings
    if ransom_flags:
        labels = ', '.join(lbl for lbl, _ in ransom_flags)
        parts.append(f"Ransomware-associated patterns detected: {labels}.")
    else:
        parts.append("No known ransomware indicator patterns were detected.")

    if goodware_hits:
        labels = ', '.join(lbl for lbl, _ in goodware_hits)
        parts.append(f"Goodware-associated patterns detected: {labels}.")

    # Conflict / agreement conclusion
    if conflict and model_says_malicious:
        parts.append(
            "CONFLICT: The rule-based indicator scan found mostly benign patterns, "
            "yet the model predicted ransomware. "
            "This means the model detected subtle statistical patterns in the "
            f"{top_section} channel that are not captured by keyword rules — "
            "possibly a non-obvious ransomware signature, or a false positive. "
            "Given the low confidence, treat this prediction with scepticism and "
            "verify with a full sandbox log."
        )
    elif conflict and not model_says_malicious:
        parts.append(
            "CONFLICT: The rule-based indicator scan found ransomware patterns, "
            "yet the model predicted benign. "
            "The model may have overridden the indicators based on the overall "
            "sequence context. Manual review is recommended."
        )
    elif not indicators_present:
        parts.append(
            "Neither ransomware nor goodware indicators were found in the input. "
            "The prediction is based entirely on learned sequence patterns, "
            "which may not be reliable for manually composed or sparse inputs."
        )
    elif indicator_leans_malicious:
        parts.append("The indicator profile corroborates the model's verdict.")
    else:
        parts.append("The indicator profile corroborates the model's verdict.")

    return {
        'section_contributions': contributions,
        'ransom_flags'         : ransom_flags,
        'goodware_flags'       : goodware_hits,
        'summary'              : ' '.join(parts),
        'top_section'          : top_section,
        'conflict'             : conflict,
        'model_says_malicious' : model_says_malicious,
        'confidence_label'     : conf_label,
    }

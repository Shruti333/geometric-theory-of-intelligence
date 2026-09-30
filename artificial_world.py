"""Harvest GPT-2 hidden states for text prompts describing 2D positions."""
import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Several phrasings, so results don't hinge on one wording.
TEMPLATES = [
    "The object is at x = {x}, y = {y}.",
    "Position ({x}, {y}) on the map.",
    "Coordinates: {x} east, {y} north.",
]

# Additional phrasings used ONLY for the robustness analysis (the main analysis still
# averages the three templates above, so earlier numbers are reproducible). They differ
# in wording and in the final token ('.', ']', ')'), which matters for last-token pooling.
EXTRA_TEMPLATES = [
    "At x = {x} and y = {y}, the robot waits.",
    "Grid cell ({x},{y})",
    "The mouse stands at row {x}, column {y}.",
    "Location: [{x}, {y}]",
    "He walked {x} blocks east and {y} blocks north.",
]
ALL_TEMPLATES = TEMPLATES + EXTRA_TEMPLATES


def load_model(model_name="gpt2", random_init=False, seed=0, device=None):
    """random_init=True gives an UNTRAINED network with the same architecture:
    the key control for 'did training create this geometry?'."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    if random_init:
        torch.manual_seed(seed)
        model = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
    else:
        model = AutoModel.from_pretrained(model_name)
    return tok, model.eval().to(device), device


@torch.no_grad()
def extract_activations(tok, model, device, coords, template, pooling="last"):
    """Returns array (n_layers + 1, N, d_model). Index 0 = embedding layer.

    pooling='last': hidden state at the final token; 'mean': average over tokens.
    Keep coordinates single-digit (grid_size <= 10) so every prompt has the same
    token count; otherwise prompt length becomes a confound.
    """
    per_point, lengths = [], set()
    for x, y in coords:
        enc = tok(template.format(x=int(x), y=int(y)), return_tensors="pt").to(device)
        lengths.add(enc["input_ids"].shape[1])
        hidden = model(**enc, output_hidden_states=True).hidden_states
        stack = torch.stack([h[0] for h in hidden])          # (L+1, T, d)
        vec = stack[:, -1] if pooling == "last" else stack.mean(dim=1)
        per_point.append(vec.float().cpu().numpy())
    if len(lengths) > 1:
        raise ValueError(f"Prompts for template {template!r} have different token counts "
                         f"{sorted(lengths)} under this tokenizer; prompt length would be a "
                         "confound. Use a model whose tokenizer gives single-token digits.")
    return np.stack(per_point, axis=1)


@torch.no_grad()
def digit_embedding_baseline(tok, model, coords):
    """Lexical baseline: concatenated input-embedding vectors of the two digits.
    Tells you what geometry you'd get if the model only knew digit identity."""
    wte = model.get_input_embeddings().weight.detach().cpu().numpy()
    ids = {}
    for d in range(10):
        enc = tok.encode(" " + str(d), add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(f"Digit {d} is not a single token in this tokenizer; "
                             "the lexical baseline is not defined for it.")
        ids[d] = enc[0]
    return np.stack([np.concatenate([wte[ids[int(x)]], wte[ids[int(y)]]])
                     for x, y in coords])
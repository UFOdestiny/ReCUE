"""Chain-of-Embedding (CoE) baseline — the real method behind "CoDE-Stop".

Wang et al., ICLR 2025 (arXiv:2410.13640). Output-free self-evaluation from the
hidden-state trajectory across layers. Per response:
  h_l = mean over response tokens of layer-l hidden state  (l=0..L)
  M(l) = ||h_{l+1}-h_l||_2 ,  A(l)=arccos(<h_l,h_{l+1}>/(||h_l|| ||h_{l+1}||))
  Z_Mag=M(h_0,h_L), Z_Ang=A(h_0,h_L)
  CoE_R = (1/L) sum_l [ M(l)/Z_Mag - A(l)/Z_Ang ]
  CoE_C = sqrt( (mean_l M(l)cosA(l))^2 + (mean_l M(l)sinA(l))^2 )
Higher => more likely correct. Single forward pass (HF, output_hidden_states),
no sampling. We recompute over the cached primary response tokens.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from acd.env import EXP_ROOT, model_path, save_json
from acd.generate import build_messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=2)
    args = ap.parse_args()

    out = EXP_ROOT / "coe" / f"{args.gen_tag}_coe.json"
    if out.exists():
        print(f"[coe] exists {out}"); return

    from transformers import AutoModelForCausalLM, AutoTokenizer

    gen = json.loads((EXP_ROOT / "gen" / f"{args.gen_tag}.json").read_text())
    if args.limit:
        gen = gen[: args.limit]
    mpath = model_path(args.model)
    tok = AutoTokenizer.from_pretrained(mpath, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        mpath, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map="cuda", attn_implementation="sdpa").eval()

    def coe_from_hidden(hs_list, resp_mask):
        # hs_list: tuple of (L+1) tensors [T, d]; resp_mask: [T] bool over response tokens
        idx = resp_mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return None, None
        H = []
        for hs in hs_list:
            h = hs[idx].mean(0).float()  # [d]
            H.append(h)
        H = torch.stack(H, 0)  # [L+1, d]
        L = H.shape[0] - 1
        M, A = [], []
        for l in range(L):
            a, b = H[l], H[l + 1]
            M.append(torch.norm(b - a).item())
            cos = torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)
            A.append(torch.arccos(torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)).item())
        M = np.array(M); A = np.array(A)
        z_mag = np.linalg.norm((H[-1] - H[0]).cpu().numpy()) + 1e-8
        a0, aL = H[0], H[-1]
        cos0 = torch.dot(a0, aL) / (torch.norm(a0) * torch.norm(aL) + 1e-8)
        z_ang = float(torch.arccos(torch.clamp(cos0, -1 + 1e-6, 1 - 1e-6))) + 1e-8
        coe_r = float(np.mean(M / z_mag - A / z_ang))
        coe_c = float(np.sqrt(np.mean(M * np.cos(A)) ** 2 + np.mean(M * np.sin(A)) ** 2))
        return coe_r, coe_c

    scores = {}
    B = args.batch
    for bstart in range(0, len(gen), B):
        batch = gen[bstart:bstart + B]
        prompts, resp_texts = [], []
        for g in batch:
            try:
                p = tok.apply_chat_template(build_messages(g["question"]), tokenize=False,
                                            add_generation_prompt=True, enable_thinking=True)
            except TypeError:
                p = tok.apply_chat_template(build_messages(g["question"]), tokenize=False,
                                            add_generation_prompt=True)
            prompts.append(p); resp_texts.append(g["primary_text"])
        for g, p, rt in zip(batch, prompts, resp_texts):
            full = p + rt
            enc = tok(full, return_tensors="pt", truncation=True, max_length=args.max_len).to("cuda")
            plen = len(tok(p, add_special_tokens=False)["input_ids"])
            T = enc["input_ids"].shape[1]
            resp_mask = torch.zeros(T, dtype=torch.bool)
            resp_mask[min(plen, T - 1):] = True
            with torch.no_grad():
                o = model(**enc, output_hidden_states=True)
            hs = [h[0] for h in o.hidden_states]  # each [T,d]
            r, c = coe_from_hidden(hs, resp_mask)
            scores[g["id"]] = {"coe_r": r, "coe_c": c}
        del o
        torch.cuda.empty_cache()
    save_json(scores, out)
    print(f"[coe] saved {len(scores)} -> {out}")


if __name__ == "__main__":
    main()

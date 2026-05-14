"""
Part 2 – Reward-Model Reranking (inference-time)
=================================================

Scores every candidate response from a pool of policy JSONLs using the
trained reward model, and emits the best-scoring response per prompt.

How it works
------------
1. For each prompt, generate N candidate responses from one or more trained
   policies (e.g. DPO β=0.1, DPO β=0.2, AOT).
2. Score every candidate with the reward model.
3. Emit the highest-scoring candidate as the final response.

This is equivalent to Best-of-N sampling when all candidates come from the
same policy, but can also mix candidates from different policies.

Usage
-----
    from llm_rl_final_proj.offline.rerank import rerank_candidates

    best_responses = rerank_candidates(
        prompts=prompts,                  # List[str]
        candidate_lists=candidate_lists,  # List[List[str]]  – one list per prompt
        reward_model=reward_model,
        reward_tokenizer=reward_tokenizer,
        max_prompt_tokens=700,
        max_response_tokens=512,
        device="cuda",
    )

Command-line example (after training DPO and AOT policies and a reward model)
------------------------------------------------------------------------------
See scripts/modal_train.py for the entrypoint.  A minimal local call:

uv run python -m llm_rl_final_proj.offline.rerank \
    --prompts_file public_eval/public_test_gen_prompts_128.jsonl \
    --policy_jsonl_files "out/dpo_b01.jsonl,out/dpo_b02.jsonl,out/wdpo_cf01.jsonl,out/wdpo_cf02.jsonl,out/wdpo_cf05.jsonl,out/rfdpo_rf01.jsonl,out/rfdpo_rf005.jsonl,out/apo_ld20.jsonl" \
    --reward_model_name Qwen/Qwen2.5-1.5B-Instruct \
    --reward_adapter_path reward_model_adapter \
    --output_file submissions/offline_best.jsonl \
    --stats_file submissions/rerank_stats.json  
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import PreTrainedTokenizerBase


def score_response(
    prompt: str,
    response: str,
    reward_model,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_prompt_tokens: int = 700,
    max_response_tokens: int = 512,
    device: str = "cuda",
) -> float:
    """Scalar reward-model score for a single (prompt, response) pair."""
    messages = [{"role": "user", "content": prompt},
                {"role": "assistant", "content": response}]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    enc = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_tokens + max_response_tokens,
    )
    with torch.no_grad():
        outputs = reward_model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        score = outputs.logits
        if score.ndim == 2 and score.shape[-1] == 1:
            score = score[:, 0]
    return float(score.item())



def rerank_candidates(
    prompts: List[str],
    candidate_lists: List[List[str]],
    policy_names: List[str],
    reward_model,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_prompt_tokens: int = 700,
    max_response_tokens: int = 512,
    device: str = "cuda",
) -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Returns
    -------
    best_responses : List[str]
    winner_names   : List[str]   which policy won each prompt
    all_scores     : List[List[float]]  shape [n_prompts][n_policies]
    """
    best_responses, winner_names, all_scores = [], [], []

    for i, (prompt, candidates) in enumerate(zip(prompts, candidate_lists)):
        scores = [
            score_response(
                prompt, cand, reward_model, tokenizer,
                max_prompt_tokens=max_prompt_tokens,
                max_response_tokens=max_response_tokens,
                device=device,
            )
            for cand in candidates
        ]
        best_idx = int(torch.tensor(scores).argmax().item())
        best_responses.append(candidates[best_idx])
        winner_names.append(policy_names[best_idx])
        all_scores.append(scores)

        if (i + 1) % 10 == 0:
            print(f"  scored {i+1}/{len(prompts)} prompts ...")

    return best_responses, winner_names, all_scores


def load_candidate_lists(
    prompts_file: str | Path,
    policy_jsonl_files: List[str | Path],
) -> Tuple[List[str], List[List[str]]]:
    """Returns (prompts, candidate_lists[prompt_idx][policy_idx])."""
    prompts: List[str] = []
    with Path(prompts_file).open() as f:
        for line in f:
            obj = json.loads(line)
            if "prompt" in obj:
                prompts.append(obj.get("prompt_text", ""))
            else:
                user_msgs = [m["content"] for m in obj.get("messages", []) if m["role"] == "user"]
                prompts.append(user_msgs[-1] if user_msgs else "")

    all_responses: List[List[str]] = []
    for path in policy_jsonl_files:
        responses: List[str] = []
        with Path(path).open() as f:
            for line in f:
                obj = json.loads(line)
                responses.append(obj.get("response_text", ""))
        if len(responses) != len(prompts):
            raise ValueError(f"{path}: expected {len(prompts)} lines, got {len(responses)}")
        all_responses.append(responses)

    # transpose: [policy][prompt] -> [prompt][policy]
    candidate_lists = [
        [all_responses[j][i] for j in range(len(all_responses))]
        for i in range(len(prompts))
    ]
    return prompts, candidate_lists


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reward-model reranking - Part 2.")
    parser.add_argument("--prompts_file", required=True)
    parser.add_argument(
        "--policy_jsonl_files", required=True,
        help="Comma-separated list of policy JSONL files, one per method/checkpoint."
    )
    parser.add_argument(
        "--policy_names", default="",
        help="Optional comma-separated labels (defaults to filenames)."
    )
    parser.add_argument("--reward_model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--reward_adapter_path", required=True)
    parser.add_argument("--output_file", required=True,
                        help="Path for reranked output JSONL (your submission file).")
    parser.add_argument("--stats_file", default="",
                        help="Optional JSON file with per-policy win counts and mean RM scores.")
    parser.add_argument("--max_prompt_tokens", type=int, default=700)
    parser.add_argument("--max_response_tokens", type=int, default=512)
    args = parser.parse_args()

    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_files = [p.strip() for p in args.policy_jsonl_files.split(",")]
    policy_names = (
        [n.strip() for n in args.policy_names.split(",")]
        if args.policy_names
        else [Path(p).stem for p in policy_files]
    )
    if len(policy_names) != len(policy_files):
        raise ValueError("--policy_names count must match --policy_jsonl_files count.")

    print(f"Policies ({len(policy_files)}):")
    for name, path in zip(policy_names, policy_files):
        print(f"  {name:<30}  {path}")

    print("\nLoading reward model ...")
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model_name)
    base = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_name, num_labels=1, torch_dtype=torch.bfloat16
    ).to(device)
    reward_model = PeftModel.from_pretrained(base, args.reward_adapter_path).to(device)
    reward_model.eval()

    print("Loading candidates ...")
    prompts, candidate_lists = load_candidate_lists(args.prompts_file, policy_files)
    print(f"  {len(prompts)} prompts x {len(policy_files)} policies\n")

    print("Scoring ...")
    best_responses, winner_names, all_scores = rerank_candidates(
        prompts, candidate_lists, policy_names, reward_model, tokenizer,
        max_prompt_tokens=args.max_prompt_tokens,
        max_response_tokens=args.max_response_tokens,
        device=device,
    )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for prompt, response in zip(prompts, best_responses):
            f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
    print(f"\nWrote {len(best_responses)} reranked responses -> {out_path}")

    # Win-count summary (free proxy metric - no GPT-4.1 needed)
    win_counts = Counter(winner_names)
    mean_scores = {
        name: float(torch.tensor([all_scores[i][j] for i in range(len(prompts))]).mean())
        for j, name in enumerate(policy_names)
    }

    print("\n-- Reranking summary (reward-model proxy) --")
    print(f"  {'Policy':<30}  {'Wins':>6}  {'Win%':>6}  {'Mean RM score':>14}")
    for name in policy_names:
        wins = win_counts.get(name, 0)
        print(f"  {name:<30}  {wins:>6}  {wins/len(prompts)*100:>5.1f}%  {mean_scores[name]:>14.4f}")

    if args.stats_file:
        stats = {
            "n_prompts": len(prompts),
            "policies": policy_names,
            "win_counts": dict(win_counts),
            "win_rates": {n: win_counts.get(n, 0) / len(prompts) for n in policy_names},
            "mean_rm_scores": mean_scores,
        }
        Path(args.stats_file).write_text(json.dumps(stats, indent=2))
        print(f"Stats written -> {args.stats_file}")


if __name__ == "__main__":
    main()
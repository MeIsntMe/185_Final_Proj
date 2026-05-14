from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from llm_rl_final_proj.models.logprobs import compute_per_token_logprobs, masked_mean_per_row
from llm_rl_final_proj.models.load import PolicyModel
from llm_rl_final_proj.offline.batch import PreferenceBatch
from llm_rl_final_proj.utils.peft_utils import disable_adapter_if_possible


@dataclass
class SequenceScores:
    chosen_logp_sum: torch.Tensor
    rejected_logp_sum: torch.Tensor
    chosen_logp_mean: torch.Tensor
    rejected_logp_mean: torch.Tensor


@dataclass
class OfflineLossOutput:
    loss: torch.Tensor
    metrics: Dict[str, float]


def compute_policy_and_reference_scores(
    model: PolicyModel,
    batch: PreferenceBatch,
    *,
    need_reference: bool,
    policy_enable_grad: bool = True,
) -> tuple[SequenceScores, SequenceScores | None]:
    policy_scores = _compute_sequence_scores(model, batch=batch, enable_grad=policy_enable_grad)
    reference_scores = None
    if need_reference:
        with torch.no_grad():
            with disable_adapter_if_possible(model):
                reference_scores = _compute_sequence_scores(model, batch=batch, enable_grad=False)
    return policy_scores, reference_scores


def compute_offline_preference_loss(
    *,
    algo: str,
    beta: float,
    policy_scores: SequenceScores,
    reference_scores: SequenceScores | None,
    example_weights: torch.Tensor | None = None,
    # Part 2 hyperparameters (ignored by Part 1 algorithms)
    conf_floor: float = 0.2,
    apo_lambda_up: float = 1.0,
    apo_lambda_down: float = 1.0,
    rf_target_margin: float = 0.05,
    rf_sft_weight: float = 0.05,
) -> OfflineLossOutput:
    
    algo = str(algo).strip().lower()
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")

    policy_margin_sum = policy_scores.chosen_logp_sum - policy_scores.rejected_logp_sum
    policy_margin_mean = policy_scores.chosen_logp_mean - policy_scores.rejected_logp_mean

    metrics: Dict[str, float] = {
        "preference/policy_margin_sum_mean": float(policy_margin_sum.detach().mean().item()),
        "preference/policy_margin_mean_mean": float(policy_margin_mean.detach().mean().item()),
        "preference/policy_accuracy_sum": float((policy_margin_sum.detach() > 0).float().mean().item()),
        "preference/policy_accuracy_mean": float((policy_margin_mean.detach() > 0).float().mean().item()),
        "preference/policy_chosen_logp_sum_mean": float(policy_scores.chosen_logp_sum.detach().mean().item()),
        "preference/policy_rejected_logp_sum_mean": float(policy_scores.rejected_logp_sum.detach().mean().item()),
        "preference/policy_chosen_logp_mean_mean": float(policy_scores.chosen_logp_mean.detach().mean().item()),
        "preference/policy_rejected_logp_mean_mean": float(policy_scores.rejected_logp_mean.detach().mean().item()),
    }

    # Part 1 methods
    # ------------------------------

    if algo == "dpo":
        if reference_scores is None:
            raise ValueError("DPO requires reference scores.")
        ref_margin_sum = reference_scores.chosen_logp_sum - reference_scores.rejected_logp_sum
        # TODO(student): compute the reference-corrected DPO logits.
        # Hint: compare the policy margin against the frozen reference margin.
        logits = beta * (policy_margin_sum - ref_margin_sum)

        # TODO(student): calculate DPO logistic loss.
        losses = -F.logsigmoid(logits)
        metrics.update(
            {
                "preference/reference_margin_sum_mean": float(ref_margin_sum.detach().mean().item()),
                "preference/reference_corrected_margin_mean": float(logits.detach().mean().item()),
                "preference/reference_corrected_accuracy": float((logits.detach() > 0).float().mean().item()),
            }
        )
    elif algo == "ipo":
        if reference_scores is None:
            raise ValueError("IPO requires reference scores.")
        ref_margin_sum = reference_scores.chosen_logp_sum - reference_scores.rejected_logp_sum

        # TODO(student): compute the reference-corrected IPO logits.
        logits = policy_margin_sum - ref_margin_sum
        target_gap = 1.0 / (2.0 * beta)
        # TODO(student): implement the squared IPO target-gap objective.
        losses = (logits - target_gap) ** 2

        metrics.update(
            {
                "preference/reference_margin_sum_mean": float(ref_margin_sum.detach().mean().item()),
                "preference/reference_corrected_margin_mean": float(logits.detach().mean().item()),
                "preference/ipo_target_gap": float(target_gap),
            }
        )
    elif algo == "aot":
        if reference_scores is None:
            raise ValueError("AOT requires reference scores.")
        # TODO(student): convert policy/reference scores into chosen and rejected rewards,
        # sort both reward vectors, and apply a DPO-style logistic loss to the quantile gaps.
        chosen_rewards = beta * (policy_scores.chosen_logp_sum - reference_scores.chosen_logp_sum)
        rejected_rewards = beta * (policy_scores.rejected_logp_sum - reference_scores.rejected_logp_sum)
        chosen_rewards = chosen_rewards[chosen_rewards.argsort()]
        rejected_rewards = rejected_rewards[rejected_rewards.argsort()]
        quantile_gap = chosen_rewards - rejected_rewards
        losses = -F.logsigmoid(quantile_gap)
        
        metrics.update(
            {
                "preference/aot_chosen_reward_mean": float(chosen_rewards.detach().mean().item()),
                "preference/aot_rejected_reward_mean": float(rejected_rewards.detach().mean().item()),
                "preference/aot_quantile_gap_mean": float(quantile_gap.detach().mean().item()),
                "preference/aot_quantile_accuracy": float((quantile_gap.detach() > 0).float().mean().item()),
            }
        )
    
    # Part 2 methods  
    # ------------------------------

    elif algo == "wdpo": 
        # Confidence-Aware Weighted DPO (wdpo)
        # hyperparameters: conf_floor
        #         
        if reference_scores is None:
            raise ValueError("wdpo requires reference scores.")

        ref_margin_sum = reference_scores.chosen_logp_sum - reference_scores.rejected_logp_sum

        # Standard DPO logits
        logits = beta * (policy_margin_sum - ref_margin_sum)
        dpo_losses = -F.logsigmoid(logits)

        # Confidence weights from the reference margin
        raw_conf = torch.sigmoid(ref_margin_sum.detach())  # no grad through weights
        # conf_floor passed in via kwarg; see TrainConfig.conf_floor
        weights = raw_conf.clamp(min=conf_floor)

        # Weighted average (normalise so effective batch size is preserved)
        losses = dpo_losses * weights / weights.mean().clamp(min=1e-6)

        metrics.update(
            {
                "preference/reference_margin_sum_mean": float(ref_margin_sum.detach().mean().item()),
                "preference/wdpo_logits_mean": float(logits.detach().mean().item()),
                "preference/wdpo_accuracy": float((logits.detach() > 0).float().mean().item()),
                "preference/wdpo_conf_mean": float(weights.mean().item()),
                "preference/wdpo_conf_min": float(weights.min().item()),
            }
        )

    elif algo == "apo":
        # Asymmetric Preference Optimization (apo)
        # hyperparameters: 
        #   apo_lambda_up   – weight on the chosen push-up term (default 1.0)
        #   apo_lambda_down – weight on the rejected push-down term (default 1.0)
        #                     sweep: (1.0,1.0) / (1.0,0.5) / (0.5,1.0)
        if reference_scores is None:
            raise ValueError("apo requires reference scores.")

        # Per-side reference-corrected rewards (no margin – independent terms)
        chosen_reward  = beta * (policy_scores.chosen_logp_sum  - reference_scores.chosen_logp_sum)
        rejected_reward = beta * (policy_scores.rejected_logp_sum - reference_scores.rejected_logp_sum)

        # Push up chosen: reward should be positive (above reference)
        loss_up   = -F.logsigmoid(chosen_reward)

        # Push down rejected: penalise when rejected reward is above the reference.
        # -log σ(-x) = log(1 + e^x), which is large when x > 0 (rejected above ref).
        loss_down = -F.logsigmoid(-rejected_reward)

        # apo_lambda_up / apo_lambda_down passed in via kwargs; see TrainConfig
        losses = apo_lambda_up * loss_up + apo_lambda_down * loss_down

        metrics.update(
            {
                "preference/apo_chosen_reward_mean":   float(chosen_reward.detach().mean().item()),
                "preference/apo_rejected_reward_mean": float(rejected_reward.detach().mean().item()),
                "preference/apo_loss_up_mean":         float(loss_up.detach().mean().item()),
                "preference/apo_loss_down_mean":       float(loss_down.detach().mean().item()),
                "preference/apo_chosen_accuracy":      float((chosen_reward.detach() > 0).float().mean().item()),
                "preference/apo_rejected_accuracy":    float((rejected_reward.detach() < 0).float().mean().item()),
            }
        )

    elif algo == "rf_dpo":
        # Reference-Free DPO 
        # hyperparameters 
        #   rf_target_margin – desired per-token margin (default 0.05)
        #                      sweep: 0.02 / 0.05 / 0.1
        #   rf_sft_weight    – SFT anchor coefficient   (default 0.05)
        #                      sweep: 0.0 / 0.05 / 0.1
        # ----------------------------------------------------------------
        # rf_target_margin / rf_sft_weight passed in via kwargs; see TrainConfig

        # Use mean log-probs to remove length bias
        margin = policy_margin_mean

        # Soft hinge: zero loss once margin exceeds target, grows for margin < target
        pref_losses = F.softplus(beta * (rf_target_margin - margin))

        # SFT anchor: prevents margin inflation via chosen-logp collapse
        sft_anchor = -policy_scores.chosen_logp_mean

        losses = pref_losses + rf_sft_weight * sft_anchor

        metrics.update(
            {
                "preference/rf_dpo_margin_mean":   float(margin.detach().mean().item()),
                "preference/rf_dpo_pref_loss_mean": float(pref_losses.detach().mean().item()),
                "preference/rf_dpo_sft_mean":       float(sft_anchor.detach().mean().item()),
                "preference/rf_dpo_accuracy":       float((margin.detach() > rf_target_margin).float().mean().item()),
            }
        )

    else:
        raise ValueError(
            f"Unknown offline preference algo: {algo}. "
            "The student starter exposes only Part 1 algorithms by default. "
            "Add your Part 2 method here."
        )

    if example_weights is not None:
        weights = example_weights.to(losses.device, dtype=losses.dtype).clamp_min(1e-6)
        if losses.shape != weights.shape:
            raise ValueError(
                f"example_weights shape {tuple(weights.shape)} is incompatible with losses shape {tuple(losses.shape)}"
            )
        weighted_loss = (losses * weights).sum() / weights.sum()
        metrics["preference/example_weight_mean"] = float(weights.detach().mean().item())
        metrics["preference/example_weight_min"] = float(weights.detach().min().item())
        metrics["preference/example_weight_max"] = float(weights.detach().max().item())
    else:
        weighted_loss = losses.mean()

    metrics["preference/loss"] = float(weighted_loss.detach().item())
    return OfflineLossOutput(loss=weighted_loss, metrics=metrics)


def _compute_sequence_scores(model: PolicyModel, *, batch: PreferenceBatch, enable_grad: bool) -> SequenceScores:
    input_ids = torch.cat([batch.chosen_input_ids, batch.rejected_input_ids], dim=0)
    attention_mask = torch.cat([batch.chosen_attention_mask, batch.rejected_attention_mask], dim=0)
    response_mask = torch.cat([batch.chosen_response_mask, batch.rejected_response_mask], dim=0)

    per_token_logprobs = compute_per_token_logprobs(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        enable_grad=enable_grad,
    )
    seq_logp_sum = (per_token_logprobs * response_mask).sum(dim=1)
    seq_logp_mean = masked_mean_per_row(per_token_logprobs, response_mask)

    chosen_sum, rejected_sum = seq_logp_sum.chunk(2, dim=0)
    chosen_mean, rejected_mean = seq_logp_mean.chunk(2, dim=0)
    return SequenceScores(
        chosen_logp_sum=chosen_sum,
        rejected_logp_sum=rejected_sum,
        chosen_logp_mean=chosen_mean,
        rejected_logp_mean=rejected_mean,
    )

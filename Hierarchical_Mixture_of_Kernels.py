#!/usr/bin/env python
# dpo_hmk_all_in_one.py
# A single-file example: DPO + Hierarchical Mixture of Kernels + Custom Divergences.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ===========================
# 1) Install / Import TRL & Transformers
#    (pip install transformers accelerate datasets trl==0.4.7 pot)
# ===========================
try:
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    from trl import DPOTrainer, DPOConfig
except ImportError:
    raise ImportError(
        "Please install 'transformers' and 'trl' libraries:\n"
        "  pip install transformers accelerate datasets trl==0.4.7 pot"
    )

# For Wasserstein, we need POT (Python Optimal Transport)
try:
    import ot
except ImportError:
    pass  # We'll only use Wasserstein if 'pot' is installed.


# ===========================
# 2) Hierarchical Mixture of Kernels (HMK) Module
# ===========================
class HierarchicalMixtureOfKernels(nn.Module):
    """
    Implements a "Hierarchical Mixture of Kernels" (HMK) combining:
      - Local kernels:  RBF, Polynomial
      - Global kernels: Spectral, Mahalanobis

    The final kernel is:
        K(x,y) = tau_local * K_local(x,y) + tau_global * K_global(x,y),
      where
        K_local(x,y)  = alpha_rbf * K_rbf(x,y) + alpha_poly * K_poly(x,y),
        K_global(x,y) = alpha_spec * K_spec(x,y) + alpha_maha * K_maha(x,y).

    We enforce:
      - alpha_rbf + alpha_poly = 1
      - alpha_spec + alpha_maha = 1
      - tau_local + tau_global = 1
    by storing raw logits and applying softmax in each group.

    The kernel parameters (e.g. sigma for RBF, etc.) can also be learned. 
    This is a minimal example; feel free to expand as needed.
    """

    def __init__(self, d):
        """
        Args:
          d: dimensionality of embeddings (for Mahalanobis).
        """
        super().__init__()
        # 2 local mixture weights, 2 global mixture weights, 2 top mixture weights
        self.local_logits = nn.Parameter(torch.zeros(2))   # [logit_rbf, logit_poly]
        self.global_logits = nn.Parameter(torch.zeros(2))  # [logit_spec, logit_maha]
        self.top_logits = nn.Parameter(torch.zeros(2))     # [logit_local, logit_global]

        # Example kernel parameters
        self.sigma = nn.Parameter(torch.tensor(1.0))  # RBF
        self.c     = nn.Parameter(torch.tensor(1.0))  # Polynomial offset
        self.degree = 2                               # fixed polynomial degree, or make it a Parameter
        # For Spectral, define 2 components:
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.5]))
        self.alphas  = nn.Parameter(torch.tensor([1.0, 0.1]))
        # For Mahalanobis, define an inverse covariance or something simpler (like identity).
        # We'll store a lower‐triangular matrix to ensure positivity, or just do identity for demo:
        self.maha_inv = nn.Parameter(torch.eye(d))  # or a Cholesky factor if you want to ensure PD.

    def forward(self, emb_x, emb_y):
        """
        Returns the kernel value K(emb_x, emb_y), shape: (batch,).
        """
        # 1) Compute each sub‐kernel
        k_rbf   = self._rbf_kernel(emb_x, emb_y)
        k_poly  = self._poly_kernel(emb_x, emb_y)
        k_spec  = self._spectral_kernel(emb_x, emb_y)
        k_maha  = self._mahalanobis_kernel(emb_x, emb_y)

        # 2) Convert logits -> weights that sum to 1 in each group
        local_weights = F.softmax(self.local_logits, dim=0)   # [alpha_rbf, alpha_poly]
        global_weights = F.softmax(self.global_logits, dim=0) # [alpha_spec, alpha_maha]
        top_weights = F.softmax(self.top_logits, dim=0)       # [tau_local, tau_global]

        alpha_rbf, alpha_poly = local_weights
        alpha_spec, alpha_maha = global_weights
        tau_local, tau_global  = top_weights

        # 3) Construct local & global mixtures
        k_local  = alpha_rbf * k_rbf + alpha_poly * k_poly
        k_global = alpha_spec * k_spec + alpha_maha * k_maha

        # 4) Final mixture
        k_final = tau_local * k_local + tau_global * k_global
        return k_final

    def _rbf_kernel(self, x, y):
        # K_rbf(x,y) = exp( -||x-y||^2 / (2 sigma^2) )
        dist_sq = torch.sum((x - y)**2, dim=-1)
        return torch.exp(-dist_sq / (2.0 * self.sigma.clamp(min=1e-6)**2))

    def _poly_kernel(self, x, y):
        # K_poly(x,y) = (x·y + c)^degree
        dot_xy = torch.sum(x * y, dim=-1) + self.c
        return torch.pow(torch.clamp(dot_xy, min=1e-12), self.degree)

    def _spectral_kernel(self, x, y):
        # K_spec(x,y) = sum_k lambdas[k] * exp(-alphas[k] ||x-y||^2)
        dist_sq = torch.sum((x - y)**2, dim=-1, keepdim=True)  # (batch,1)
        # shape (K,) for lambdas, alphas
        exps = []
        for lam, alpha in zip(self.lambdas, self.alphas):
            exps.append(lam * torch.exp(-alpha.clamp(min=0.0) * dist_sq.squeeze(-1)))
        exps = torch.stack(exps, dim=-1).sum(dim=-1)
        return exps

    def _mahalanobis_kernel(self, x, y):
        # K_maha(x,y) = exp( -(x-y)^T Sigma_inv (x-y) / 2 )
        diff = x - y
        quad = torch.sum(torch.matmul(diff, self.maha_inv) * diff, dim=-1)
        return torch.exp(-0.5 * quad)


def hmk_log_ratio(hmk_module, emb_x, emb_y_pos, emb_y_neg):
    """
    Compute log( K(emb_x, emb_y_pos) / K(emb_x, emb_y_neg) ) using HMK.
    """
    eps = 1e-12
    k_pos = hmk_module(emb_x, emb_y_pos)
    k_neg = hmk_module(emb_x, emb_y_neg)
    return torch.log(torch.clamp(k_pos, min=eps)) - torch.log(torch.clamp(k_neg, min=eps))


# ===========================
# 3) Custom Divergence Function
#    (KL, JS, Rényi, Bhattacharyya, Wasserstein, f‐Divergence)
# ===========================
def compute_divergence(log_p, log_q, divergence_type='kl', **kwargs):
    """
    Returns D( p || q ) for each sample, shape: (batch,).
    p = exp(log_p), q = exp(log_q).
    """
    eps = 1e-12
    p = torch.clamp(torch.exp(log_p), min=eps)
    q = torch.clamp(torch.exp(log_q), min=eps)

    if divergence_type == 'kl':
        return torch.sum(p * (log_p - log_q), dim=-1)

    elif divergence_type == 'js':
        m = 0.5 * (p + q)
        log_m = torch.log(m)
        kl_pm = torch.sum(p * (log_p - log_m), dim=-1)
        kl_qm = torch.sum(q * (log_q - log_m), dim=-1)
        return 0.5 * kl_pm + 0.5 * kl_qm

    elif divergence_type == 'renyi':
        alpha = kwargs.get('alpha', 1.01)
        pq_term = torch.sum(p**alpha * q**(1.0 - alpha), dim=-1)
        return (1.0 / (alpha - 1.0)) * torch.log(pq_term)

    elif divergence_type == 'bhattacharyya':
        bc_coeff = torch.sum(torch.sqrt(p * q), dim=-1)
        return -torch.log(torch.clamp(bc_coeff, min=eps))

    elif divergence_type == 'wasserstein':
        cost_matrix = kwargs.get('cost_matrix', None)
        if cost_matrix is None:
            raise ValueError("Must provide 'cost_matrix' for Wasserstein.")
        # Each example: solve discrete OT
        batch_size = p.shape[0]
        cost_np = cost_matrix.cpu().numpy()
        div_list = []
        for i in range(batch_size):
            p_i = p[i].cpu().numpy()
            q_i = q[i].cpu().numpy()
            T = ot.emd(p_i, q_i, cost_np)
            w = np.sum(T * cost_np)
            div_list.append(w)
        return torch.tensor(div_list, device=log_p.device, dtype=log_p.dtype)

    elif divergence_type == 'f_divergence':
        # D_f(P||Q) = sum_i q_i * f( p_i / q_i )
        f_func = kwargs.get('f_func', None)
        if f_func is None:
            raise ValueError("Must provide 'f_func' for f_divergence.")
        r = p / q
        f_r = f_func(r)
        return torch.sum(q * f_r, dim=-1)

    else:
        raise ValueError(f"Unsupported divergence_type={divergence_type}.")


# ===========================
# 4) Custom DPOTrainer that uses HMK + Custom Divergence
# ===========================
from trl import DPOTrainer, DPOConfig

class HMKDPOTrainer(DPOTrainer):
    """
    A specialized DPOTrainer that:
      - Uses a Hierarchical Mixture of Kernels for the embedding-based term
      - Uses a custom divergence (KL, JS, Rényi, etc.) instead of pure KL
    """
    def __init__(
        self,
        *args,
        hmk_module=None,          # HierarchicalMixtureOfKernels
        embedding_model=None,     # model to get embeddings
        kernel_scale=1.0,         # "kappa" in the formula
        divergence_type='kl',
        divergence_scale=1.0,     # "alpha" in the formula
        divergence_kwargs=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.hmk_module = hmk_module
        self.embedding_model = embedding_model
        self.kernel_scale = kernel_scale
        self.divergence_type = divergence_type
        self.divergence_scale = divergence_scale
        self.divergence_kwargs = divergence_kwargs if divergence_kwargs else {}

    def _compute_loss(self, batch):
        """
        Overridden method to compute the DPO loss:
          L = - E[ kernel_scale * ( log(pi(y+|x)/pi(y-|x)) + log(Kpos/Kneg) ) ] 
              + divergence_scale * D( pi(.|x), pi_ref(.|x) ).
        """
        chosen_ids = batch["chosen_ids"]
        rejected_ids = batch["rejected_ids"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # --- Forward pass for policy (chosen vs. rejected)
        policy_output_chosen = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=chosen_ids
        )
        policy_output_rejected = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=rejected_ids
        )
        log_probs_chosen = self._compute_sequence_log_probs(
            policy_output_chosen.logits, chosen_ids, attention_mask
        )
        log_probs_rejected = self._compute_sequence_log_probs(
            policy_output_rejected.logits, rejected_ids, attention_mask
        )
        # Sum over tokens
        prob_term = (log_probs_chosen - log_probs_rejected).sum(dim=-1)  # (batch,)

        # --- Embedding-based HMK ratio
        if self.embedding_model is not None and self.hmk_module is not None:
            with torch.no_grad():
                # Example: embed the entire input or just the first token, or the response, etc.
                emb_x     = self.embedding_model(input_ids).last_hidden_state[:, 0, :]
                emb_y_pos = self.embedding_model(chosen_ids).last_hidden_state[:, 0, :]
                emb_y_neg = self.embedding_model(rejected_ids).last_hidden_state[:, 0, :]
            kernel_ratio = hmk_log_ratio(self.hmk_module, emb_x, emb_y_pos, emb_y_neg)
        else:
            kernel_ratio = torch.zeros_like(prob_term)

        hybrid_term = self.kernel_scale * (prob_term + kernel_ratio)

        # --- Custom Divergence: compare policy vs. ref on final token distribution
        ref_output_chosen = self.ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=chosen_ids
        )
        # final token index for chosen
        last_pos_idx = (chosen_ids != self.tokenizer.pad_token_id).sum(dim=-1) - 1
        final_logits_chosen = policy_output_chosen.logits[range(chosen_ids.size(0)), last_pos_idx]
        final_ref_logits_chosen = ref_output_chosen.logits[range(chosen_ids.size(0)), last_pos_idx]

        log_p = F.log_softmax(final_logits_chosen, dim=-1)
        log_q = F.log_softmax(final_ref_logits_chosen, dim=-1)

        div_per_example = compute_divergence(
            log_p, log_q,
            divergence_type=self.divergence_type,
            **self.divergence_kwargs
        )
        divergence_penalty = self.divergence_scale * div_per_example

        # --- Final Loss
        total_loss = -(hybrid_term - divergence_penalty)
        return total_loss.mean()


# ===========================
# 5) Main: Demo Training on Dummy Data
# ===========================
def main():
    # 1) Load policy & ref models, plus an embedding model
    policy_name = "gpt2"
    ref_name    = "gpt2"
    embed_name  = "bert-base-uncased"  # an encoder for embeddings

    print("Loading models...")
    policy_model = AutoModelForCausalLM.from_pretrained(policy_name)
    ref_model    = AutoModelForCausalLM.from_pretrained(ref_name)
    embedding_model = AutoModel.from_pretrained(embed_name)  # e.g. BERT encoder

    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    tokenizer.pad_token = tokenizer.eos_token  # ensure pad token

    # 2) Create our HMK module (assume dimension ~768 for BERT's [CLS] embedding)
    #    If you know the exact dimension, pass it. We'll assume 768.
    hmk = HierarchicalMixtureOfKernels(d=768)

    # 3) Create a DPO config & trainer
    config = DPOConfig(
        model_name=policy_name,
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
    )
    trainer = HMKDPOTrainer(
        config=config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        hmk_module=hmk,
        embedding_model=embedding_model,
        kernel_scale=1.0,             # kappa
        divergence_type='renyi',      # choose any: 'kl','js','renyi','bhattacharyya','wasserstein','f_divergence'
        divergence_scale=1.0,         # alpha
        divergence_kwargs={'alpha': 0.9},  # e.g. for Rényi
    )

    # 4) Build a tiny dummy dataset
    def random_ids(seq_len, vocab_size=50257):
        return [np.random.randint(1000, vocab_size) for _ in range(seq_len)]

    dummy_data = []
    for _ in range(4):
        prompt_ids = random_ids(8)
        chosen_ids = prompt_ids + random_ids(4)
        rejected_ids = prompt_ids + random_ids(4)
        dummy_data.append({
            "input_ids": torch.tensor(prompt_ids),
            "attention_mask": torch.ones(len(prompt_ids), dtype=torch.long),
            "chosen_ids": torch.tensor(chosen_ids),
            "rejected_ids": torch.tensor(rejected_ids),
        })

    # 5) Train on dummy data
    print("Starting training on dummy data...")
    trainer.train(dummy_data, epochs=1)
    print("Training complete.")

    # 6) Example generation
    test_prompt = "What is the capital of France?"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids
    gen_out = trainer.model.generate(input_ids, max_length=40)
    print("Model output:", tokenizer.decode(gen_out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

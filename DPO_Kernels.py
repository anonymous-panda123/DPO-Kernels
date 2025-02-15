#!/usr/bin/env python
# dpo_all_in_one.py
# A single-file example combining kernels, custom divergences, and DPO training via TRL.

import torch
import torch.nn.functional as F
import numpy as np

# -------------------------------
# 1) Install / Import TRL & Transformers
#    Make sure you have: pip install transformers accelerate datasets trl==0.4.7 pot
# -------------------------------
try:
    from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
    from trl import DPOTrainer, DPOConfig
except ImportError:
    raise ImportError("Please install 'transformers' and 'trl' libraries (pip install transformers trl).")


# -------------------------------
# 2) Define Kernel Log-Ratio Function
#    K(e_x, e_y) can be polynomial, RBF, spectral, or Mahalanobis, returning log(K(pos)/K(neg)).
# -------------------------------
def kernel_log_ratio(emb_x, emb_y_pos, emb_y_neg, kernel_type='rbf', **kwargs):
    """
    Returns log( K(emb_x, emb_y_pos) / K(emb_x, emb_y_neg) ), shape: (batch,).
    Implementing 4 common kernels:
      - 'polynomial': (x·y + c)^d
      - 'rbf': exp( -||x-y||^2 / (2 sigma^2) )
      - 'spectral': sum_k lambda_k exp( -alpha_k ||x-y||^2 )
      - 'mahalanobis': exp( - (x-y)^T Sigma_inv (x-y) / 2 )
    """
    eps = 1e-12  # for numerical safety

    if kernel_type == 'polynomial':
        c = kwargs.get('c', 1.0)
        d = kwargs.get('degree', 2)
        # (x·y + c)^d => log-ratio => d * [log(x·y_pos + c) - log(x·y_neg + c)]
        xy_pos = torch.sum(emb_x * emb_y_pos, dim=-1) + c
        xy_neg = torch.sum(emb_x * emb_y_neg, dim=-1) + c
        log_num = d * torch.log(torch.clamp(xy_pos, min=eps))
        log_den = d * torch.log(torch.clamp(xy_neg, min=eps))
        return log_num - log_den

    elif kernel_type == 'rbf':
        # RBF: exp( -||x-y||^2 / (2 sigma^2) )
        sigma = kwargs.get('sigma', 1.0)
        dist_sq_pos = torch.sum((emb_x - emb_y_pos)**2, dim=-1)
        dist_sq_neg = torch.sum((emb_x - emb_y_neg)**2, dim=-1)
        # log( exp(-dist_pos/(2 sigma^2)) / exp(-dist_neg/(2 sigma^2)) )
        # = -(1/(2 sigma^2)) [ dist_pos - dist_neg ]
        return -(dist_sq_pos - dist_sq_neg) / (2.0 * (sigma**2))

    elif kernel_type == 'spectral':
        # K_spectral(x, y) = sum_{k=1}^K lambda_k * exp(-alpha_k ||x-y||^2)
        lambdas = kwargs.get('lambdas', [1.0])
        alphas  = kwargs.get('alphas',  [1.0])
        def spectral_sum(a, b):
            dsq = torch.sum((a - b)**2, dim=-1, keepdim=True)
            # sum_k lambda_k * exp(-alpha_k * dsq)
            exps = []
            for (lam, alpha) in zip(lambdas, alphas):
                exps.append(lam * torch.exp(-alpha * dsq.squeeze(-1)))
            return torch.stack(exps, dim=-1).sum(dim=-1)  # (batch,)
        num = spectral_sum(emb_x, emb_y_pos)
        den = spectral_sum(emb_x, emb_y_neg)
        return torch.log(torch.clamp(num, min=eps)) - torch.log(torch.clamp(den, min=eps))

    elif kernel_type == 'mahalanobis':
        # K_maha(x,y) = exp( - (x-y)^T Sigma_inv (x-y) / 2 )
        Sigma_inv = kwargs.get('Sigma_inv', None)
        if Sigma_inv is None:
            raise ValueError("Must provide 'Sigma_inv' for Mahalanobis kernel.")
        diff_pos = emb_x - emb_y_pos
        diff_neg = emb_x - emb_y_neg
        quad_pos = torch.sum(torch.matmul(diff_pos, Sigma_inv) * diff_pos, dim=-1)
        quad_neg = torch.sum(torch.matmul(diff_neg, Sigma_inv) * diff_neg, dim=-1)
        # log-ratio => -1/2 [quad_pos - quad_neg]
        return -0.5 * (quad_pos - quad_neg)

    else:
        raise ValueError(f"Unsupported kernel_type={kernel_type}.")


# -------------------------------
# 3) Define a General Divergence Function
#    For KL, JS, Rényi, Bhattacharyya, Wasserstein, or a generic f-Divergence.
# -------------------------------
def compute_divergence(log_p, log_q, divergence_type='kl', **kwargs):
    """
    Returns D( p || q ) for each sample, shape: (batch,).
    p = exp(log_p), q = exp(log_q).
    divergences: 'kl', 'js', 'renyi', 'bhattacharyya', 'wasserstein', 'f_divergence'.
    """
    eps = 1e-12
    p = torch.clamp(torch.exp(log_p), min=eps)
    q = torch.clamp(torch.exp(log_q), min=eps)

    if divergence_type == 'kl':
        # KL(P||Q) = sum_i p_i [log p_i - log q_i]
        return torch.sum(p * (log_p - log_q), dim=-1)

    elif divergence_type == 'js':
        # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), M = 0.5(P+Q)
        m = 0.5 * (p + q)
        log_m = torch.log(m)
        kl_pm = torch.sum(p * (log_p - log_m), dim=-1)
        kl_qm = torch.sum(q * (log_q - log_m), dim=-1)
        return 0.5 * kl_pm + 0.5 * kl_qm

    elif divergence_type == 'renyi':
        # Rényi_\alpha(P||Q) = 1/(alpha - 1) * log sum_i p_i^alpha * q_i^(1 - alpha)
        alpha = kwargs.get('alpha', 1.01)
        pq_term = torch.sum(p**alpha * q**(1.0 - alpha), dim=-1)
        return (1.0 / (alpha - 1.0)) * torch.log(pq_term)

    elif divergence_type == 'bhattacharyya':
        # -log sum_i sqrt(p_i q_i)
        bc_coeff = torch.sum(torch.sqrt(p * q), dim=-1)
        return -torch.log(torch.clamp(bc_coeff, min=eps))

    elif divergence_type == 'wasserstein':
        # Use POT library for discrete optimal transport
        cost_matrix = kwargs.get('cost_matrix', None)
        if cost_matrix is None:
            raise ValueError("Must provide 'cost_matrix' for Wasserstein.")
        import ot  # pip install pot
        # We'll do a loop for each example
        div_list = []
        cost_np = cost_matrix.cpu().numpy()
        batch_size = p.shape[0]
        for i in range(batch_size):
            p_i = p[i].cpu().numpy()
            q_i = q[i].cpu().numpy()
            T = ot.emd(p_i, q_i, cost_np)
            w = np.sum(T * cost_np)
            div_list.append(w)
        return torch.tensor(div_list, device=log_p.device, dtype=log_p.dtype)

    elif divergence_type == 'f_divergence':
        # D_f(P||Q) = sum_i q_i * f(p_i / q_i)
        f_func = kwargs.get('f_func', None)
        if f_func is None:
            raise ValueError("Must provide 'f_func' for f_divergence.")
        r = p / q
        f_r = f_func(r)
        return torch.sum(q * f_r, dim=-1)

    else:
        raise ValueError(f"Unsupported divergence_type={divergence_type}.")


# -------------------------------
# 4) Subclass TRL's DPOTrainer to Insert Kernel & Divergence
# -------------------------------
from trl import DPOTrainer, DPOConfig

class CustomDPOTrainer(DPOTrainer):
    """
    Extends the standard TRL DPOTrainer to incorporate:
      - Kernel-based embedding ratio
      - Custom divergence
    """
    def __init__(
        self,
        *args,
        kernel_type='rbf',
        divergence_type='kl',
        kernel_kwargs=None,
        divergence_kwargs=None,
        embedding_model=None,
        kappa=1.0,
        alpha=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.kernel_type = kernel_type
        self.divergence_type = divergence_type
        self.kernel_kwargs = kernel_kwargs if kernel_kwargs else {}
        self.divergence_kwargs = divergence_kwargs if divergence_kwargs else {}
        self.embedding_model = embedding_model  # separate or same as policy
        self.kappa = kappa
        self.alpha = alpha

    def _compute_loss(self, batch):
        """
        Overridden method that calculates the DPO loss for each batch.
        Includes:
         - log prob ratio: log π(y+|x) - log π(y-|x)
         - kernel log ratio: log K(e_x, e_{y+}) / K(e_x, e_{y-})
         - custom divergence D(pi || pi_ref)
        """
        # 1) Extract input data
        chosen_ids = batch["chosen_ids"]
        rejected_ids = batch["rejected_ids"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # 2) Forward pass: policy vs. chosen & rejected
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

        # 3) Reference model pass
        ref_output_chosen = self.ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=chosen_ids
        )
        ref_output_rejected = self.ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=rejected_ids
        )
        ref_log_probs_chosen = self._compute_sequence_log_probs(
            ref_output_chosen.logits, chosen_ids, attention_mask
        )
        ref_log_probs_rejected = self._compute_sequence_log_probs(
            ref_output_rejected.logits, rejected_ids, attention_mask
        )

        # 4) Contrastive Probability Term
        #    sum across tokens or you might do something else
        prob_term = (log_probs_chosen - log_probs_rejected).sum(dim=-1)

        # 5) Kernel Embedding Term
        #    We'll embed the entire input_ids or just the last hidden states, etc.
        if self.embedding_model is not None:
            # Example: use the [CLS] or first token embedding from an encoder
            with torch.no_grad():
                # For demonstration, embed the input (x). 
                # In practice, you might embed the prompt or the entire (x,y).
                emb_x = self.embedding_model(input_ids).last_hidden_state[:, 0, :]
                emb_y_pos = self.embedding_model(chosen_ids).last_hidden_state[:, 0, :]
                emb_y_neg = self.embedding_model(rejected_ids).last_hidden_state[:, 0, :]
            kernel_ratio = kernel_log_ratio(
                emb_x, emb_y_pos, emb_y_neg,
                kernel_type=self.kernel_type,
                **self.kernel_kwargs
            )
        else:
            # If no separate embedding model, just skip or set kernel_ratio = 0
            kernel_ratio = torch.zeros_like(prob_term)

        hybrid_loss = self.kappa * (prob_term + kernel_ratio)

        # 6) Custom Divergence across the final token (or entire distribution)
        #    For demonstration, let's just use final logits for chosen:
        last_pos_idx = (chosen_ids != self.tokenizer.pad_token_id).sum(dim=-1) - 1
        final_logits_chosen = policy_output_chosen.logits[
            range(policy_output_chosen.logits.size(0)), last_pos_idx
        ]
        final_ref_logits_chosen = ref_output_chosen.logits[
            range(ref_output_chosen.logits.size(0)), last_pos_idx
        ]
        log_p = F.log_softmax(final_logits_chosen, dim=-1)
        log_q = F.log_softmax(final_ref_logits_chosen, dim=-1)

        div_per_example = compute_divergence(
            log_p, log_q,
            divergence_type=self.divergence_type,
            **self.divergence_kwargs
        )
        divergence_penalty = self.alpha * div_per_example

        # 7) Final DPO Loss
        total_loss = -(hybrid_loss - divergence_penalty)
        return total_loss.mean()


# -------------------------------
# 5) A Minimal "main" to Demonstrate Usage
# -------------------------------
def main():
    # a) Load policy, reference, and embedding models
    policy_name = "gpt2"
    ref_name    = "gpt2"
    embed_name  = "bert-base-uncased"  # for embeddings

    print("Loading models/tokenizers...")
    policy_model = AutoModelForCausalLM.from_pretrained(policy_name)
    ref_model    = AutoModelForCausalLM.from_pretrained(ref_name)
    embedding_model = AutoModel.from_pretrained(embed_name)  # just an encoder
    tokenizer = AutoTokenizer.from_pretrained(policy_name)
    tokenizer.pad_token = tokenizer.eos_token  # ensure there's a pad token

    # b) Create config & trainer
    config = DPOConfig(
        model_name=policy_name,
        learning_rate=1e-5,
        batch_size=2,
        mini_batch_size=2,
        gradient_accumulation_steps=1,
    )
    trainer = CustomDPOTrainer(
        config=config,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        kernel_type='rbf',
        divergence_type='renyi',
        kernel_kwargs={'sigma': 0.5},
        divergence_kwargs={'alpha': 0.9},
        embedding_model=embedding_model,
        kappa=1.0,
        alpha=1.0,
    )

    # c) Construct a tiny dummy dataset with random IDs for demonstration
    #    In real usage, you'd have (prompt, chosen, rejected) examples.
    #    We'll create 4 samples, each "prompt" of length 8, "chosen"/"rejected" of length 4.
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

    # d) Train (on dummy data)
    print("Starting training on dummy data...")
    trainer.train(dummy_data, epochs=1)
    print("Training complete.")

    # e) Test generation
    test_prompt = "What is the capital of France?"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids
    gen_out = trainer.model.generate(input_ids, max_length=40)
    print("Model output:", tokenizer.decode(gen_out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()

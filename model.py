"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # Assign special tokens first, in the exact order provided.
    vocab = {token: idx for idx, token in enumerate(specials)}

    # Add corpus tokens in order of first appearance, skipping specials
    # and tokens that have already been added.
    next_id = len(specials)

    for sentence in sentences:
        for token in sentence.split():
            if token not in vocab:
                vocab[token] = next_id
                next_id += 1

    return vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # Build the inverse mapping: id -> token.
    return {idx: token for token, idx in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # Get the ID used for unknown/out-of-vocabulary tokens.
    unk_id = token_to_id[unk_token]

    # Convert each whitespace-separated token to its vocabulary ID.
    # Unknown tokens fall back to unk_id.
    return [token_to_id.get(token, unk_id) for token in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # Map each token ID to its corresponding token string,
    # preserving the original order.
    return [id_to_token[idx] for idx in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # Truncate the sequence if it is longer than max_len.
    padded = list(ids[:max_len])

    # Pad on the right until the sequence reaches exactly max_len.
    padded.extend([pad_id] * (max_len - len(padded)))

    return padded

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    return torch.tensor(padded_sequences, dtype=torch.long)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
def compute_positional_div_term(d_model):
    # Frequency divisors for even feature indices:
    # exp(-log(10000) * (2k / d_model)), k = 0, 1, ..., d_model//2 - 1.
    k = torch.arange(0, d_model // 2, dtype=torch.float32)
    return torch.exp(-torch.log(torch.tensor(10000.0)) * (2 * k / d_model))

# Step 9 - build_position_index_column
def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    return torch.arange(max_len, dtype=torch.float32).unsqueeze(1)

# Step 10 - fill_even_indices_with_sin
def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    pe[:, 0::2] = torch.sin(position * div_term)
    return pe

# Step 11 - fill_odd_indices_with_cos
def fill_odd_indices_with_cos(pe, position, div_term):
    # Fill odd-indexed feature columns with cos(position * div_term).
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    pe = torch.zeros((max_len, d_model), dtype=torch.float32)

    position = build_position_index_column(max_len)
    div_term = compute_positional_div_term(d_model)

    pe = fill_even_indices_with_sin(pe, position, div_term)
    pe = fill_odd_indices_with_cos(pe, position, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # Get the sequence length L from the embedded batch.
    seq_len = embedded_batch.size(1)

    # Add the first L positional encodings. Shape (L, d_model)
    # broadcasts across the batch dimension (B, L, d_model).
    return embedded_batch + positional_encoding[:seq_len]

# Step 14 - build_padding_mask
def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    return (token_ids != pad_id).unsqueeze(1).unsqueeze(2)

# Step 15 - build_causal_mask
def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    return torch.tril(
        torch.ones((seq_len, seq_len), dtype=torch.bool)
    ).unsqueeze(0).unsqueeze(0)

# Step 16 - combine_padding_and_causal_masks
def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # Combine the masks using broadcasting.
    # padding_mask: (B, 1, 1, L)
    # causal_mask:  (1, 1, L, L)
    # result:       (B, 1, L, L)
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    return torch.matmul(query, key.transpose(-2, -1))

# Step 18 - scale_attention_scores
def scale_attention_scores(scores, d_k):
    # Scale the attention scores by sqrt(d_k).
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    return scores.masked_fill(~mask, float('-inf'))

# Step 20 - softmax_attention_weights
def softmax_attention_weights(masked_scores):
    # Identify rows where every score is -inf.
    all_masked = torch.isneginf(masked_scores).all(dim=-1, keepdim=True)

    # Replace all-masked rows with zeros temporarily so softmax
    # does not produce NaNs.
    safe_scores = masked_scores.masked_fill(all_masked, 0.0)

    # Apply softmax along the key dimension.
    weights = torch.softmax(safe_scores, dim=-1)

    # Ensure completely masked rows are exactly zero.
    return weights.masked_fill(all_masked, 0.0)

# Step 21 - apply_attention_weights_to_values
def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    return torch.matmul(attention_weights, value)

# Step 22 - scaled_dot_product_attention
def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # Compute raw attention scores: (..., Lq, Lk)
    scores = compute_raw_attention_scores(query, key)

    # Scale scores by sqrt(d_k).
    d_k = query.size(-1)
    scores = scale_attention_scores(scores, d_k)

    # Optionally mask out disallowed positions.
    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)

    # Convert scores to attention weights.
    attention_weights = softmax_attention_weights(scores)

    # Mix the value vectors using the attention weights.
    context = apply_attention_weights_to_values(attention_weights, value)

    return context, attention_weights

# Step 23 - split_last_dim_into_heads
def split_last_dim_into_heads(tensor, num_heads):
    # Split the final feature dimension into (num_heads, d_k).
    batch_size, seq_len, d_model = tensor.shape
    d_k = d_model // num_heads

    return tensor.reshape(batch_size, seq_len, num_heads, d_k)

# Step 24 - transpose_heads_before_sequence
def transpose_heads_before_sequence(split_tensor):
    # Rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.permute(0, 2, 1, 3)

# Step 25 - merge_heads_back_to_model_dim
def merge_heads_back_to_model_dim(multi_head_tensor):
    # Rearrange (B, num_heads, L, d_k) into (B, L, num_heads, d_k),
    # then merge the final two dimensions into d_model.
    batch_size, num_heads, seq_len, d_k = multi_head_tensor.shape

    return (
        multi_head_tensor
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(batch_size, seq_len, num_heads * d_k)
    )

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # Apply the affine transformation:
    # y = x @ weight.T + bias
    output = x @ weight.transpose(-2, -1)

    if bias is not None:
        output = output + bias

    return output

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # Apply the three independent linear projections.
    q = apply_linear_projection(x, w_q, b_q)
    k = apply_linear_projection(x, w_k, b_k)
    v = apply_linear_projection(x, w_v, b_v)

    return q, k, v

# Step 28 - split_qkv_into_heads
def split_qkv_into_heads(q, k, v, num_heads):
    # Split the last feature dimension into (num_heads, d_k),
    # then move the head dimension before the sequence dimension.
    q_h = transpose_heads_before_sequence(
        split_last_dim_into_heads(q, num_heads)
    )
    k_h = transpose_heads_before_sequence(
        split_last_dim_into_heads(k, num_heads)
    )
    v_h = transpose_heads_before_sequence(
        split_last_dim_into_heads(v, num_heads)
    )

    return q_h, k_h, v_h

# Step 29 - multi_head_scaled_dot_product_attention
def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # Run scaled dot-product attention independently for each head.
    # The head dimension is treated as an additional batch dimension.
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
def merge_heads_and_project_output(context, w_o, b_o):
    # Merge (B, num_heads, L, d_k) into (B, L, d_model).
    merged = merge_heads_back_to_model_dim(context)

    # Apply the final output projection.
    return apply_linear_projection(merged, w_o, b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(
    query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None
):
    # Project query, key, and value into their respective feature spaces.
    q, _, _ = project_to_query_key_value(
        query, w_q, None, w_k, None, w_v, None
    )
    _, k, _ = project_to_query_key_value(
        key, w_q, None, w_k, None, w_v, None
    )
    _, _, v = project_to_query_key_value(
        value, w_q, None, w_k, None, w_v, None
    )

    # Split the projected tensors into independent attention heads.
    q_h, k_h, v_h = split_qkv_into_heads(q, k, v, num_heads)

    # Run scaled dot-product attention independently across all heads.
    context, _ = multi_head_scaled_dot_product_attention(
        q_h, k_h, v_h, mask
    )

    # Merge the heads and apply the final output projection.
    return merge_heads_and_project_output(context, w_o, None)

# Step 32 - apply_ffn_first_linear_and_relu
import torch

def apply_ffn_first_linear_and_relu(x, w1, b1):
    # Project from d_model to d_ff, then apply ReLU.
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
def apply_ffn_second_linear(hidden, w2, b2):
    # Project from d_ff back to d_model and add the output bias.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # First linear projection followed by ReLU.
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)

    # Second linear projection back to d_model.
    return apply_ffn_second_linear(hidden, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
def compute_layer_norm_mean_and_variance(x):
    # Compute the mean over the last feature dimension.
    mean = x.mean(dim=-1, keepdim=True)

    # Compute the population (biased) variance over the last dimension.
    variance = ((x - mean) ** 2).mean(dim=-1, keepdim=True)

    return mean, variance

# Step 36 - normalize_and_scale_with_gamma_beta
def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # Compute the mean and population variance along the last dimension.
    mean, variance = compute_layer_norm_mean_and_variance(x)

    # Normalize x with numerical stability from eps.
    normalized = (x - mean) / torch.sqrt(variance + eps)

    # Apply the learned affine transformation.
    return gamma * normalized + beta

# Step 37 - apply_residual_add_and_norm
def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # Add the residual connection, then apply layer normalization.
    combined = residual_input + sublayer_output
    return normalize_and_scale_with_gamma_beta(combined, gamma, beta, eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # Apply the keep mask and use inverted-dropout scaling
    # so the expected value remains unchanged during training.
    return x * keep_mask.to(dtype=x.dtype) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(
    x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask
):
    # Run multi-head self-attention with x as query, key, and value.
    attention_output = assemble_multi_head_attention_forward(
        x,
        x,
        x,
        w_q,
        w_k,
        w_v,
        w_o,
        num_heads,
        src_mask,
    )

    # Apply the residual connection followed by layer normalization.
    return apply_residual_add_and_norm(
        x,
        attention_output,
        gamma,
        beta,
    )

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # Run the position-wise feed-forward network.
    ffn_output = position_wise_feed_forward_network(
        x, w1, b1, w2, b2
    )

    # Apply the residual connection followed by layer normalization.
    return apply_residual_add_and_norm(
        x,
        ffn_output,
        gamma,
        beta,
    )

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # First sublayer: multi-head self-attention + residual add-and-norm.
    x = encoder_layer_self_attention_sublayer(
        x,
        layer_params["w_q"],
        layer_params["w_k"],
        layer_params["w_v"],
        layer_params["w_o"],
        layer_params["attn_gamma"],
        layer_params["attn_beta"],
        num_heads,
        src_mask,
    )

    # Second sublayer: position-wise FFN + residual add-and-norm.
    x = encoder_layer_feed_forward_sublayer(
        x,
        layer_params["w1"],
        layer_params["b1"],
        layer_params["w2"],
        layer_params["b2"],
        layer_params["ffn_gamma"],
        layer_params["ffn_beta"],
    )

    return x

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # Sequentially apply each encoder layer to the running hidden state.
    for layer_params in encoder_layer_params_list:
        x = assemble_encoder_layer(
            x,
            layer_params,
            num_heads,
            src_mask,
        )

    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer (not yet solved)
# TODO: implement

# Step 44 - decoder_layer_cross_attention_sublayer (not yet solved)
# TODO: implement

# Step 45 - decoder_layer_feed_forward_sublayer (not yet solved)
# TODO: implement

# Step 46 - assemble_decoder_layer (not yet solved)
# TODO: implement

# Step 47 - stack_decoder_layers (not yet solved)
# TODO: implement

# Step 48 - apply_final_output_projection (not yet solved)
# TODO: implement

# Step 49 - tie_output_projection_to_token_embeddings (not yet solved)
# TODO: implement

# Step 50 - apply_log_softmax_over_vocab (not yet solved)
# TODO: implement

# Step 51 - run_transformer_forward (not yet solved)
# TODO: implement

# Step 52 - init_encoder_layer_parameters (not yet solved)
# TODO: implement

# Step 53 - init_decoder_layer_parameters (not yet solved)
# TODO: implement

# Step 54 - init_embedding_and_projection_parameters (not yet solved)
# TODO: implement

# Step 55 - collect_model_parameters_into_list (not yet solved)
# TODO: implement

# Step 56 - shift_targets_right_with_start_token (not yet solved)
# TODO: implement

# Step 57 - compute_noam_learning_rate (not yet solved)
# TODO: implement

# Step 58 - build_uniform_smoothing_distribution (not yet solved)
# TODO: implement

# Step 59 - set_confidence_on_gold_tokens (not yet solved)
# TODO: implement

# Step 60 - zero_pad_column_and_pad_token_rows (not yet solved)
# TODO: implement

# Step 61 - compute_label_smoothed_kl_loss (not yet solved)
# TODO: implement

# Step 62 - average_loss_over_non_pad_tokens (not yet solved)
# TODO: implement

# Step 63 - compute_token_accuracy_ignoring_pad (not yet solved)
# TODO: implement

# Step 64 - initialize_adam_optimizer_state (not yet solved)
# TODO: implement

# Step 65 - update_adam_first_moment (not yet solved)
# TODO: implement

# Step 66 - update_adam_second_moment (not yet solved)
# TODO: implement

# Step 67 - apply_adam_bias_correction (not yet solved)
# TODO: implement

# Step 69 - apply_adam_step_to_all_parameters (not yet solved)
# TODO: implement

# Step 70 - zero_all_parameter_gradients (not yet solved)
# TODO: implement

# Step 71 - compute_batch_training_loss (not yet solved)
# TODO: implement

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement


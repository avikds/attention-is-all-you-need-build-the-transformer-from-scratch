# Attention Is All You Need: Build the Transformer From Scratch

Reimplement the original encoder-decoder Transformer end to end in PyTorch, from token vocabularies and sinusoidal positional encodings through multi-head attention, label smoothing, Noam scheduling, and beam search. By the end you will have a working seq2seq Transformer training loop and inference pipeline assembled from first principles.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** build_token_to_id_vocab
- [x] **2.** build_id_to_token_vocab
- [x] **3.** encode_sentence_to_ids
- [x] **4.** decode_ids_to_tokens
- [x] **5.** pad_id_sequence
- [x] **6.** stack_padded_sequences_to_batch
- [x] **7.** scale_embeddings_by_sqrt_d_model
- [x] **8.** compute_positional_div_term
- [x] **9.** build_position_index_column
- [x] **10.** fill_even_indices_with_sin
- [x] **11.** fill_odd_indices_with_cos
- [x] **12.** build_sinusoidal_positional_encoding
- [x] **13.** add_positional_encoding_to_embeddings
- [x] **14.** build_padding_mask
- [x] **15.** build_causal_mask
- [x] **16.** combine_padding_and_causal_masks
- [x] **17.** compute_raw_attention_scores
- [x] **18.** scale_attention_scores
- [x] **19.** mask_attention_scores_with_neg_inf
- [x] **20.** softmax_attention_weights
- [x] **21.** apply_attention_weights_to_values
- [x] **22.** scaled_dot_product_attention
- [x] **23.** split_last_dim_into_heads
- [x] **24.** transpose_heads_before_sequence
- [x] **25.** merge_heads_back_to_model_dim
- [x] **26.** apply_linear_projection
- [x] **27.** project_to_query_key_value
- [x] **28.** split_qkv_into_heads
- [x] **29.** multi_head_scaled_dot_product_attention
- [x] **30.** merge_heads_and_project_output
- [x] **31.** assemble_multi_head_attention_forward
- [x] **32.** apply_ffn_first_linear_and_relu
- [x] **33.** apply_ffn_second_linear
- [x] **34.** position_wise_feed_forward_network
- [ ] **35.** compute_layer_norm_mean_and_variance
- [ ] **36.** normalize_and_scale_with_gamma_beta
- [ ] **37.** apply_residual_add_and_norm
- [ ] **38.** apply_dropout_with_keep_mask
- [ ] **39.** encoder_layer_self_attention_sublayer
- [ ] **40.** encoder_layer_feed_forward_sublayer
- [ ] **41.** assemble_encoder_layer
- [ ] **42.** stack_encoder_layers
- [ ] **43.** decoder_layer_masked_self_attention_sublayer
- [ ] **44.** decoder_layer_cross_attention_sublayer
- [ ] **45.** decoder_layer_feed_forward_sublayer
- [ ] **46.** assemble_decoder_layer
- [ ] **47.** stack_decoder_layers
- [ ] **48.** apply_final_output_projection
- [ ] **49.** tie_output_projection_to_token_embeddings
- [ ] **50.** apply_log_softmax_over_vocab
- [ ] **51.** run_transformer_forward
- [ ] **52.** init_encoder_layer_parameters
- [ ] **53.** init_decoder_layer_parameters
- [ ] **54.** init_embedding_and_projection_parameters
- [ ] **55.** collect_model_parameters_into_list
- [ ] **56.** shift_targets_right_with_start_token
- [ ] **57.** compute_noam_learning_rate
- [ ] **58.** build_uniform_smoothing_distribution
- [ ] **59.** set_confidence_on_gold_tokens
- [ ] **60.** zero_pad_column_and_pad_token_rows
- [ ] **61.** compute_label_smoothed_kl_loss
- [ ] **62.** average_loss_over_non_pad_tokens
- [ ] **63.** compute_token_accuracy_ignoring_pad
- [ ] **64.** initialize_adam_optimizer_state
- [ ] **65.** update_adam_first_moment
- [ ] **66.** update_adam_second_moment
- [ ] **67.** apply_adam_bias_correction
- [ ] **69.** apply_adam_step_to_all_parameters
- [ ] **70.** zero_all_parameter_gradients
- [ ] **71.** compute_batch_training_loss
- [ ] **72.** run_training_step_with_backprop
- [ ] **73.** run_training_loop_for_steps
- [ ] **74.** pick_next_token_by_argmax
- [ ] **75.** compute_length_penalty
- [ ] **76.** compute_candidate_scores
- [ ] **77.** select_top_k_candidates
- [ ] **78.** append_tokens_to_beam_sequences
- [ ] **79.** mark_finished_beams
- [ ] **80.** select_best_finished_beam

---

Built on Deep-ML.

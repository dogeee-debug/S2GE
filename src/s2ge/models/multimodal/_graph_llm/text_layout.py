"""Assemble left-padded LLM inputs with explicit graph-token spans."""

import torch

BOS = "<s>[INST]"
EOS_USER = "[/INST]"
EOS = "</s>"
IGNORE_INDEX = -100


def _prepare_text_layout(self, samples, include_labels=True, labels=None, graph_token_counts=None, return_graph_spans=False):
    """Create token IDs, masks, labels, and insertion offsets for each sample.

    Graph positions are reserved immediately after the instruction BOS marker.
    Only answer tokens contribute to the causal-LM objective; prompt, graph,
    and padding positions receive ``IGNORE_INDEX``.
    """
    llm_input_device = self.llm_input_device
    questions = self.tokenizer(samples["question"], add_special_tokens=False)
    descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
    label_tokens = None
    if include_labels:
        label_tokens = self.tokenizer(samples["label"], add_special_tokens=False) if labels is None else labels

    bos_token_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors="pt").input_ids[0].to(llm_input_device)
    eos_user_ids = self.tokenizer(EOS_USER, add_special_tokens=False).input_ids
    eos_ids = self.tokenizer(EOS, add_special_tokens=False).input_ids

    batch_size = len(samples["id"])
    if graph_token_counts is None:
        graph_token_counts = [1] * batch_size
    graph_token_counts = [max(0, int(count)) for count in graph_token_counts]
    text_sequences, label_sequences = [], []
    total_lengths = []
    bos_len = int(bos_token_ids.numel())

    for i in range(batch_size):
        label_input_ids = []
        if include_labels:
            label_input_ids = label_tokens.input_ids[i][: self.max_new_tokens] + eos_ids
        text_ids = descriptions.input_ids[i][: self.max_txt_len] + questions.input_ids[i] + eos_user_ids + label_input_ids
        text_sequences.append(text_ids)
        label_sequences.append(label_input_ids)
        total_lengths.append(bos_len + graph_token_counts[i] + len(text_ids))

    max_total_length = max(total_lengths) if total_lengths else bos_len + 1
    pad_token_id = int(self.tokenizer.pad_token_id)
    token_ids = torch.full((batch_size, max_total_length), pad_token_id, dtype=torch.long, device=llm_input_device)
    attention_mask = torch.zeros((batch_size, max_total_length), dtype=torch.long, device=llm_input_device)
    label_input_ids = None
    if include_labels:
        label_input_ids = torch.full(
            (batch_size, max_total_length),
            IGNORE_INDEX,
            dtype=torch.long,
            device=llm_input_device,
        )

    graph_positions = []
    for i, text_ids in enumerate(text_sequences):
        total_len = total_lengths[i]
        start = max_total_length - total_len
        bos_start = start
        graph_pos = bos_start + bos_len
        text_start = graph_pos + graph_token_counts[i]

        token_ids[i, bos_start : bos_start + bos_len] = bos_token_ids
        if text_ids:
            token_ids[i, text_start : text_start + len(text_ids)] = torch.tensor(
                text_ids,
                dtype=torch.long,
                device=llm_input_device,
            )
        attention_mask[i, start : start + total_len] = 1

        if include_labels and label_sequences[i]:
            label_seq = label_sequences[i]
            label_start = text_start + len(text_ids) - len(label_seq)
            label_input_ids[i, label_start : label_start + len(label_seq)] = torch.tensor(
                label_seq,
                dtype=torch.long,
                device=llm_input_device,
            )
        graph_positions.append(graph_pos)

    graph_positions = torch.tensor(graph_positions, dtype=torch.long, device=llm_input_device)
    graph_token_counts = torch.tensor(graph_token_counts, dtype=torch.long, device=llm_input_device)
    if return_graph_spans:
        return token_ids, attention_mask, label_input_ids, graph_positions, graph_token_counts
    return token_ids, attention_mask, label_input_ids, graph_positions


def _build_text_tensors(self, samples, graph_token_embeds, graph_token_mask=None, labels=None):
    """Replace reserved spans with projected graph embeddings."""
    if graph_token_embeds.dim() == 2:
        graph_token_embeds = graph_token_embeds.unsqueeze(1)
    if graph_token_mask is None:
        graph_token_mask = torch.ones(
            graph_token_embeds.shape[:2],
            dtype=torch.bool,
            device=graph_token_embeds.device,
        )
    graph_token_counts = graph_token_mask.sum(dim=1).tolist()
    token_ids, attention_mask, label_input_ids, graph_positions, graph_token_counts = self._prepare_text_layout(
        samples,
        include_labels=True,
        labels=labels,
        graph_token_counts=graph_token_counts,
        return_graph_spans=True,
    )
    inputs_embeds = self.word_embedding(token_ids)
    inputs_embeds = inputs_embeds.clone()
    for batch_idx in range(inputs_embeds.size(0)):
        token_count = int(graph_token_counts[batch_idx].item())
        graph_start = int(graph_positions[batch_idx].item())
        inputs_embeds[batch_idx, graph_start : graph_start + token_count] = graph_token_embeds[
            batch_idx, :token_count
        ].to(dtype=self.llm_embed_dtype)
    return inputs_embeds, attention_mask, label_input_ids

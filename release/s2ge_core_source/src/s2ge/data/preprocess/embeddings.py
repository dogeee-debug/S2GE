import os
import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import gensim
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer


SBERT_REPO = "sentence-transformers/all-roberta-large-v1"
CONTRIEVER_REPO = "facebook/contriever"
WORD2VEC_HIDDEN_DIM = 300
DEFAULT_BATCH_SIZE = 64
DEFAULT_TEXT_CHUNK_SIZE = 4096
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


class _TextDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {"input_ids": input_ids, "att_mask": attention_mask}

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        return {key: value[index] for key, value in self.data.items() if value is not None}


class SentenceTransformer(nn.Module):
    def __init__(self, pretrained_repo, local_files_only=False):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(pretrained_repo, local_files_only=local_files_only)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        data_type = token_embeddings.dtype
        expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * expanded_mask, 1) / torch.clamp(expanded_mask.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)
        return F.normalize(sentence_embeddings, p=2, dim=1)


@dataclass
class EmbeddingBundle:
    model: object
    tokenizer: object
    device: torch.device
    text2embedding: Callable
    embedding_dim: int


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _offline_mode_enabled():
    return any(
        os.getenv(name, "0") == "1"
        for name in ("S2GE_OFFLINE_MODE", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _local_files_only(pretrained_repo):
    return Path(str(pretrained_repo)).exists() or _offline_mode_enabled()


def _validate_local_transformers_dir_exists(path_like, label: str):
    path = Path(str(path_like))
    if not path.exists():
        raise FileNotFoundError(f"{label} not found locally: {path}")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} must be a directory: {path}")
    return path


def _validate_local_transformers_config(path_like, label: str):
    path = _validate_local_transformers_dir_exists(path_like, label)
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"{label} is incomplete, missing config.json: {path}")
    return path


def _validate_local_transformers_tokenizer_files(path_like, label: str):
    path = _validate_local_transformers_config(path_like, label)
    tokenizer_candidates = [
        path / "tokenizer.json",
        path / "tokenizer_config.json",
        path / "vocab.json",
        path / "sentence_bert_config.json",
    ]
    if not any(candidate.exists() for candidate in tokenizer_candidates):
        raise FileNotFoundError(f"{label} is incomplete, missing tokenizer files: {path}")
    return path


def _validate_local_transformers_weight_files(path_like, label: str):
    path = _validate_local_transformers_config(path_like, label)
    weight_candidates = [
        path / "pytorch_model.bin",
        path / "model.safetensors",
        path / "model.safetensors.index.json",
    ]
    if not any(candidate.exists() for candidate in weight_candidates):
        raise FileNotFoundError(f"{label} is incomplete, missing model weights: {path}")
    return path


def _validate_local_transformers_dir(path_like, label: str, require_weights: bool = True, require_tokenizer: bool = True):
    path = _validate_local_transformers_config(path_like, label)
    if require_tokenizer:
        _validate_local_transformers_tokenizer_files(path, label)
    if require_weights:
        _validate_local_transformers_weight_files(path, label)
    return path


def _resolve_pretrained_path(repo_or_path):
    path = Path(str(repo_or_path))
    return str(path) if path.exists() else repo_or_path


def _ensure_out_path(out_path, suffix=".mmap"):
    if out_path:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    fd, tmp_path = tempfile.mkstemp(prefix="s2ge_embed_", suffix=suffix)
    os.close(fd)
    Path(tmp_path).unlink(missing_ok=True)
    return Path(tmp_path)


def _prepare_storage(num_rows: int, embed_dim: int, out_path=None):
    if out_path is None:
        return torch.empty((num_rows, embed_dim), dtype=torch.float32)
    path = _ensure_out_path(out_path)
    return np.memmap(path, dtype=np.float32, mode="w+", shape=(num_rows, embed_dim))


def _finalize_storage(storage):
    if isinstance(storage, np.memmap):
        storage.flush()
        return torch.from_numpy(storage)
    return storage


def _write_rows(storage, row_start: int, rows: torch.Tensor):
    rows = rows.detach().to(dtype=torch.float32).cpu()
    row_end = row_start + rows.size(0)
    if isinstance(storage, np.memmap):
        storage[row_start:row_end] = rows.numpy()
    else:
        storage[row_start:row_end].copy_(rows)
    return row_end


def _model_hidden_size(model):
    module = model.module if isinstance(model, nn.DataParallel) else model
    if hasattr(module, "bert_model") and hasattr(module.bert_model, "config"):
        return int(module.bert_model.config.hidden_size)
    if hasattr(module, "config") and hasattr(module.config, "hidden_size"):
        return int(module.config.hidden_size)
    raise ValueError(f"Unable to infer embedding dim from model type: {type(module)}")


def _iter_sbert_batches(model, tokenizer, device, texts, batch_size=DEFAULT_BATCH_SIZE, text_chunk_size=DEFAULT_TEXT_CHUNK_SIZE):
    with torch.no_grad():
        for chunk_start in range(0, len(texts), text_chunk_size):
            chunk_texts = texts[chunk_start : chunk_start + text_chunk_size]
            encoding = tokenizer(chunk_texts, padding=True, truncation=True, return_tensors="pt")
            dataset = _TextDataset(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                yield model(input_ids=batch["input_ids"], att_mask=batch["att_mask"])


def _iter_contriever_batches(model, tokenizer, device, texts, batch_size=DEFAULT_BATCH_SIZE, text_chunk_size=DEFAULT_TEXT_CHUNK_SIZE):
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        return token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None].clamp(min=1.0)

    with torch.no_grad():
        for chunk_start in range(0, len(texts), text_chunk_size):
            chunk_texts = texts[chunk_start : chunk_start + text_chunk_size]
            inputs = tokenizer(chunk_texts, padding=True, truncation=True, return_tensors="pt")
            dataset = _TextDataset(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["att_mask"])
                embeddings = mean_pooling(outputs[0], batch["att_mask"])
                yield F.normalize(embeddings, p=2, dim=1)


def _collect_streamed_embeddings(texts, embed_dim, batch_iter: Callable[[], Iterable[torch.Tensor]], out_path=None):
    if len(texts) == 0:
        return torch.zeros((0, embed_dim), dtype=torch.float32)
    storage = _prepare_storage(len(texts), embed_dim, out_path=out_path)
    row_cursor = 0
    for rows in batch_iter():
        row_cursor = _write_rows(storage, row_cursor, rows)
    if row_cursor != len(texts):
        raise RuntimeError(f"Embedding row count mismatch: expected {len(texts)}, wrote {row_cursor}")
    return _finalize_storage(storage)


def _sbert_text2embedding(model, tokenizer, device, texts, batch_size=DEFAULT_BATCH_SIZE, text_chunk_size=DEFAULT_TEXT_CHUNK_SIZE, out_path=None):
    embed_dim = _model_hidden_size(model)
    return _collect_streamed_embeddings(
        texts,
        embed_dim,
        batch_iter=lambda: _iter_sbert_batches(
            model,
            tokenizer,
            device,
            texts,
            batch_size=batch_size,
            text_chunk_size=text_chunk_size,
        ),
        out_path=out_path,
    )


def _contriever_text2embedding(model, tokenizer, device, texts, batch_size=DEFAULT_BATCH_SIZE, text_chunk_size=DEFAULT_TEXT_CHUNK_SIZE, out_path=None):
    embed_dim = _model_hidden_size(model)
    return _collect_streamed_embeddings(
        texts,
        embed_dim,
        batch_iter=lambda: _iter_contriever_batches(
            model,
            tokenizer,
            device,
            texts,
            batch_size=batch_size,
            text_chunk_size=text_chunk_size,
        ),
        out_path=out_path,
    )


def _stable_token_index(token: str, embed_dim: int):
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=16).digest()
    index = int.from_bytes(digest[:8], byteorder="little", signed=False) % embed_dim
    sign = 1.0 if digest[8] % 2 == 0 else -1.0
    return index, sign


def light_text2embedding(texts, embed_dim=1024, out_path=None, shard_size=DEFAULT_TEXT_CHUNK_SIZE):
    if embed_dim <= 0:
        raise ValueError(f"embed_dim must be positive, got {embed_dim}")
    if len(texts) == 0:
        return torch.zeros((0, embed_dim), dtype=torch.float32)

    storage = _prepare_storage(len(texts), embed_dim, out_path=out_path)
    row_cursor = 0
    for chunk_start in range(0, len(texts), shard_size):
        chunk = texts[chunk_start : chunk_start + shard_size]
        rows = torch.zeros((len(chunk), embed_dim), dtype=torch.float32)
        for local_idx, text in enumerate(chunk):
            tokens = TOKEN_PATTERN.findall(str(text).lower())
            if not tokens:
                continue
            for token in tokens:
                token_idx, sign = _stable_token_index(token, embed_dim)
                rows[local_idx, token_idx] += sign
            rows[local_idx] = F.normalize(rows[local_idx].unsqueeze(0), p=2, dim=1).squeeze(0)
        row_cursor = _write_rows(storage, row_cursor, rows)
    return _finalize_storage(storage)


def _word2vec_text2embedding(model, tokenizer, device, texts, out_path=None, shard_size=DEFAULT_TEXT_CHUNK_SIZE):
    if isinstance(texts, str):
        texts = [texts]
    if len(texts) == 0:
        return torch.zeros((0, WORD2VEC_HIDDEN_DIM), dtype=torch.float32)

    storage = _prepare_storage(len(texts), WORD2VEC_HIDDEN_DIM, out_path=out_path)
    row_cursor = 0
    for chunk_start in range(0, len(texts), shard_size):
        chunk = texts[chunk_start : chunk_start + shard_size]
        rows = torch.zeros((len(chunk), WORD2VEC_HIDDEN_DIM), dtype=torch.float32)
        for local_idx, text in enumerate(chunk):
            word_vectors = []
            for word in str(text).split():
                try:
                    word_vectors.append(model[word])
                except KeyError:
                    continue
            if word_vectors:
                rows[local_idx] = torch.tensor(np.mean(word_vectors, axis=0), dtype=torch.float32)
        row_cursor = _write_rows(storage, row_cursor, rows)
    return _finalize_storage(storage)


def _sbert_bundle(pretrained_repo=SBERT_REPO):
    pretrained_repo = _resolve_pretrained_path(pretrained_repo)
    local_files_only = _local_files_only(pretrained_repo)
    if Path(str(pretrained_repo)).exists():
        _validate_local_transformers_tokenizer_files(pretrained_repo, "Embedding tokenizer path")
        _validate_local_transformers_weight_files(pretrained_repo, "Embedding model path")
    model = SentenceTransformer(pretrained_repo, local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo, local_files_only=local_files_only)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    device = _device()
    model.to(device)
    model.eval()
    return EmbeddingBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        text2embedding=_sbert_text2embedding,
        embedding_dim=_model_hidden_size(model),
    )


def _contriever_bundle():
    pretrained_repo = _resolve_pretrained_path(CONTRIEVER_REPO)
    local_files_only = _local_files_only(pretrained_repo)
    if Path(str(pretrained_repo)).exists():
        _validate_local_transformers_tokenizer_files(pretrained_repo, "Embedding tokenizer path")
        _validate_local_transformers_weight_files(pretrained_repo, "Embedding model path")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_repo, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(pretrained_repo, local_files_only=local_files_only)
    device = _device()
    model.to(device)
    model.eval()
    return EmbeddingBundle(
        model=model,
        tokenizer=tokenizer,
        device=device,
        text2embedding=_contriever_text2embedding,
        embedding_dim=_model_hidden_size(model),
    )


def _word2vec_bundle(word2vec_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    return EmbeddingBundle(
        model=model,
        tokenizer=None,
        device=_device(),
        text2embedding=_word2vec_text2embedding,
        embedding_dim=WORD2VEC_HIDDEN_DIM,
    )


def load_embedding_bundle(model_name="sbert", word2vec_path="word2vec/GoogleNews-vectors-negative300.bin.gz"):
    candidate_path = Path(str(model_name))
    if candidate_path.exists():
        return _sbert_bundle(str(candidate_path))
    if model_name == "sbert":
        return _sbert_bundle()
    if isinstance(model_name, str) and "/" in model_name:
        return _sbert_bundle(model_name)
    if model_name == "contriever":
        return _contriever_bundle()
    if model_name == "word2vec":
        return _word2vec_bundle(word2vec_path)
    raise ValueError(f"Unknown embedding model: {model_name}")


__all__ = [
    "EmbeddingBundle",
    "SentenceTransformer",
    "light_text2embedding",
    "load_embedding_bundle",
]

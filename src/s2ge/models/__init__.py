"""Public model factory and canonical decoder identifiers."""

DEFAULT_LLM_MODEL_PATHS = {
    "7b": "meta-llama/Llama-2-7b-hf",
    "7b_chat": "meta-llama/Llama-2-7b-chat-hf",
    "13b": "meta-llama/Llama-2-13b-hf",
    "13b_chat": "meta-llama/Llama-2-13b-chat-hf",
    "llama3_8b_instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen2_5_1_5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
}


def build_model(args, graph_type, init_prompt=None):
    """Build the configured graph-language model."""
    if args.model_name == "graph_llm":
        from s2ge.models.multimodal.graph_llm import GraphLLM

        model_cls = GraphLLM
    else:
        raise KeyError(f"Unknown model_name: {args.model_name}")
    return model_cls(graph_type=graph_type, args=args, init_prompt=init_prompt)

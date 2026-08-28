# S2GE Authors' Notes

<p align="center">
  <a href="AUTHORS_NOTES.md">English</a> |
  <a href="AUTHORS_NOTES.zh-CN.md">简体中文</a>
</p>

## Motivations and Inspirations

S2GE started from a deliberately small question: what is the minimal graph task
that can tell us whether a language model can actually use the topology exposed
to it?

We chose shortest-hop prediction not because counting hops is the final problem
we want to solve. Instead, HopQA is a small and inspectable structural primitive.
More complicated graph tasks can often be decomposed into locating relevant
nodes, identifying connectivity, comparing paths, interpreting externally
computed structural signals, and mapping those signals to downstream decisions.

This perspective is deeply influenced by software engineering. A well-designed
system gives components bounded responsibilities and relies on stable interfaces
between them. A graph algorithm does not need to perform language understanding,
and a language model does not necessarily need to reproduce BFS, shortest-path
search, centrality estimation, or every other graph algorithm internally. What
matters is that one component's output remains readable and usable by the next.

This intuition shapes S2GE. Query-aware sampling determines which evidence enters
the bounded interface. Role-based perception makes endpoints, nearby nodes, and
context distinguishable. Adjacency-based alignment encourages local structural
relationships to survive projection. The decoder then consumes this interface
through ordinary generation. In the paper, these stages correspond to evidence
inclusion, role readability, and adjacency recoverability.

Modern LLM systems increasingly rely on retrieval systems, graph engines,
symbolic programs, tools, and databases. External computation can remove the need
for a language model to perform every graph operation, but it does not remove the
need for the model to understand the result. A retriever may return a subgraph, a
graph engine may return a path, and a planner may expose a dependency graph.
Eventually, a downstream model still has to interpret those results and act.

This is why we distinguish graph-signal existence from decoder usability. In our
experiments, graph-only and retrieval-execution controls recover useful graph
signal even when native generation remains weak. Useful structure before the
decoder therefore does not guarantee that the decoder can use it. Stronger
external computation may make a reliable graph-language interface more important,
not less.

S2GE does not show that the interface problem has been solved. It shows that the
failure can be mitigated: native generation improves when query-relevant evidence
is selected carefully, endpoint roles are made readable, and local adjacency is
better preserved.

The phrase "sampling first" also reflects reasoning under a bounded information
bottleneck. A large graph is rarely presented to a language model in its entirety;
the model receives a bounded view. The interface must decide which distinctions
survive and which information can be discarded. Good compression is not merely
about reducing token count. It should remove irrelevant information while
preserving distinctions important to the query.

Once two situations have been mapped to an indistinguishable interface, a larger
downstream decoder cannot reconstruct the missing distinction. This is the
intuition formalized by the paper's fixed-interface information bound. The DBLP
sampling audit gives a concrete example: degree-only sampling fails to include
required endpoint-conditioned evidence, while query-aware sampling substantially
improves endpoint coverage and path recall.

For us, "sampling first" is therefore a statement about where reasoning begins:
before asking a model to reason well, we must decide what information it is
allowed to see.

## Outlook

### From Structural Primitives to Stronger Tasks

We view HopQA as a starting point rather than the final application of S2GE. A
natural next step is to test whether the same interface principles support
reachability, path comparison, path witnesses, graph-grounded retrieval,
recommendation, link reasoning, relational planning, and other tasks where
structural computation must eventually be consumed by a language model.

The goal is not to turn every downstream task into another HopQA variant. We are
interested in whether a reliable structural interface can be reused while the
downstream objective changes—in software-engineering terms, whether the head can
change without rebuilding the entire interface.

### Larger Models

Our experiments use LLaMA-3-8B-Instruct. The available compute budget did not
permit a systematic study across decoder scales. This leaves an open empirical
question: how much of the observed failure is scale-sensitive, and how much is
interface-limited?

The theoretical analysis says only that scaling a decoder cannot recover
distinctions already removed by a fixed exposed interface. It does not imply that
larger models behave identically to the evaluated 8B backbone. We welcome tests
of S2GE-style interfaces with larger and more capable models.

### Optimization Sensitivity and Training Dynamics

One observation that we did not have enough time to study systematically is the
sensitivity of the output-state dynamics to optimization configuration.

During development, the low-dimensional transition between failure states and
high-EM states became particularly visible after changes to the numerical and
training configuration. This observation motivated us to inspect generated
outputs across checkpoints rather than looking only at loss or final accuracy,
which eventually led to the output-state diagnostics and checkpoint PCA reported
in the paper.

However, these configuration changes were made together during development, so
we do not claim a causal attribution to any individual factor. In particular,
the effects of numerical precision and gradient checkpointing were not isolated
in a controlled factorial experiment.

A simple experiment we would have liked to run is a small 2x2 study:

| Precision | Gradient checkpointing ON | Gradient checkpointing OFF |
| --- | --- | --- |
| FP32 | ? | ? |
| BF16 / FP16 | ? | ? |

### Beyond Graph-LLMs

We do not know whether our future work will remain centered on graph-augmented
LLMs. Research directions change. We nevertheless hope S2GE continues to evolve
through better diagnostics, stronger backbones, new downstream tasks, and tests
in real systems.

### A Possible Industrial Role

One possible industrial role for S2GE is an interface layer between external
structured computation and an LLM-based agent. An external system may already
know how to construct a service-dependency graph, workflow DAG, knowledge
subgraph, compiler call graph, or database-schema graph. The remaining question
is whether a downstream agent can reliably understand and use that structure.

The architecture would be:

```text
External structured computation -> topology-readable interface -> LLM / agent
```

S2GE is not intended to replace graph algorithms. Its role is to translate their
outputs into representations that remain usable after crossing the graph-language
boundary.

S2GE is ultimately less about teaching an LLM to count hops than about asking a
broader question: how should structured computation cross the boundary into a
language model?

**Graph evidence is not enough. What matters is whether the evidence remains
usable.**

— **Xiaoyu Guo**, on behalf of all authors

Thank you for reading our paper.

Contact: gxyhome030404@gmail.com

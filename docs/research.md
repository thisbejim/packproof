# Public-evidence research

Research was performed on 2026-09-11. The project was selected because the
same low-level correctness boundary appears in framework documentation,
implementation code, and independent bug reports.

## Evidence

1. [TRL data utilities](https://huggingface.co/docs/trl/v0.26.0/en/data_utils)
   documents `pack_dataset`, fixed `seq_length` packing, and emitted sequence
   lengths. It also shows that the packed representation is not merely a
   concatenated text file: the metadata is part of the trainer contract.
2. [Meta torchtune sample-packing documentation](https://meta-pytorch.org/torchtune/stable/basics/packing.html)
   explicitly says packed training needs document masking and relative
   position IDs so irrelevant samples do not cross-attend.
3. [Hugging Face TRL issue #3705](https://github.com/huggingface/trl/issues/3705)
   reports a packing path where `seq_lengths` was absent when the collator ran,
   causing positions to be generated as though the entire row were one
   sequence. That is exactly the sort of silent boundary drift this gate can
   catch in a saved artifact.
4. [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027)
   formalizes the need for block-diagonal attention behavior and discusses
   preserving the un-packed training result.
5. [Hugging Face packing-with-FA2 article](https://github.com/huggingface/blog/blob/main/packing-with-FA2.md)
   describes padding-free packed fine-tuning and the importance of handling
   positions and masks in the data collator.

## Existing-tool check

TRL and torchtune can produce packed data, and framework tests can catch bugs
inside their own implementation. Those tools do not provide a small,
provider-neutral, post-serialization gate that a custom preprocessing script
can run before handing a JSONL artifact to any trainer. General dataset
profilers validate columns and counts but do not reason about cross-segment
attention, position reset, or loss-mask padding leaks.

## Scope decision

The first release focuses on the portable representation that teams commonly
log or export: JSONL rows of token IDs and parallel arrays. It verifies
structural invariants and optional explicit policies; it does not attempt to
reimplement a tokenizer, trainer, or GPU kernel. That keeps the tool local,
auditable, and useful across OpenAI-compatible data pipelines, Llama-family
fine-tuning workflows, and custom PyTorch trainers.

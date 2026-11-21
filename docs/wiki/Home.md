# Têmpera Wiki

> Train ML models the right way, on the cheapest machine that gets the job done.

## Start here

1. **[Vision and Problem](Vision-and-Problem.md)**: who it is for, what pain it solves, what we don't do.
2. **[User Journey](User-Journey.md)**: the end-to-end experience.
3. **[Architecture](Architecture.md)**: how the system is put together.
4. **[Roadmap](Roadmap.md)**: the order in which we will build it.

## Technical reference

- [Data Audit](Data-Audit.md): bias, leakage, quality; role of the SLM
- [Algorithm Catalog](Algorithm-Catalog.md): algorithms × backends
- [Execution and Machines](Execution-and-Machines.md): local, SSH/LAN, cloud, agent
- [Agent Traces](Agent-Traces.md): auditing chat/tool-call data before LLM fine-tuning
- [LLM Fine-Tuning](Fine-Tuning.md): LoRA/QLoRA on agent traces, baseline first, on your own GPU
- [Agents (MCP)](Agents-MCP.md): using Têmpera from AI agents, with the same guardrails
- [Cost and Estimation](Cost-and-Estimation.md): ETA, memory, cost, savings
- [Contracts and Specifications](Contracts.md): `tempera.yaml`, gRPC, Arrow
- [Security and Privacy](Security.md)
- [Tech Stack](Tech-Stack.md)
- [Architecture Decisions (ADRs)](../adr/README.md)
- [Glossary](Glossary.md)
- [Contributing](Contributing.md)

## The 5 principles

1. **Correct before fast.** A model trained on leaked data is money thrown away. The audit comes before training.
2. **Cheap by default.** Start with the simplest baseline, use the machine you already have, and only move up to a paid GPU when the numbers justify it.
3. **Statistics decide, the LLM explains.** The SLM is an advisor and never selects data on its own.
4. **Best tool for each part.** Rust for the system, Python for the ML ecosystem, with clear contracts between the two.
5. **Local-first and private.** Nothing leaves your infrastructure unless you say so.

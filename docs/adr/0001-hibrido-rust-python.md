# ADR-0001: Arquitetura híbrida Rust (sistema) + Python (ecossistema ML)

- **Status:** Aceito
- **Data:** 2026-11-16

## Contexto
Queremos um sistema robusto, distribuível como binário único, com execução remota confiável, **e** acesso ao melhor do ecossistema de ML (PyTorch, HF, sklearn, LightGBM, Cleanlab, Fairlearn). O ecossistema de ML em Rust (Burn, Candle, linfa) é promissor, mas tem bem menos modelos pré-treinados, algoritmos e bibliotecas de qualidade de dados.

## Decisão
- **Rust** para sistema: server, orquestrador, executores, agente, dados (Polars), auditoria estatística, estimativas, SLM local, serving.
- **Python** para treino quando o ecossistema é decisivo.
- **Rust nativo** para treino quando basta (tabular simples, MLP/CNN pequenas). Sobe instantaneamente e não exige Python.
- Contratos: Arrow (dados), safetensors/ONNX (modelos), gRPC (controle).

## Alternativas consideradas
- **Tudo em Rust:** sem compromissos de linguagem, mas o catálogo seria anos mais pobre. Rejeitado.
- **Tudo em Python:** ecossistema completo, mas perde o binário único, a robustez do agente e o desempenho de dados, e herda o inferno de ambientes. Rejeitado.

## Consequências
- (+) Cada parte usa a melhor ferramenta.
- (−) Duas toolchains e CI mais complexa; os workers precisam de Python quando o job exige (mitigado com `uv` + cache).

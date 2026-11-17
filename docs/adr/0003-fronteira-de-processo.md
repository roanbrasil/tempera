# ADR-0003: Fronteira de processo (subprocesso + JSONL) em vez de FFI

- **Status:** Aceito
- **Data:** 2026-11-16

## Contexto
Podemos integrar Python via PyO3 (no mesmo processo) ou via subprocesso.

## Decisão
O backend Python roda como **subprocesso** do agente e reporta eventos em JSONL pelo stdout. PyO3 só entra em extensões de desempenho chamadas *de dentro* do Python (ex.: data loader em Rust).

## Alternativas consideradas
- PyO3 embutido: menor latência, mas um crash ou OOM do Python derruba o agente, há disputa de GIL e o empacotamento fica complexo.
- gRPC dentro do worker Python: mais acoplamento e dependências sem ganho real.

## Consequências
- (+) Isolamento de falhas, adaptadores triviais e backends em qualquer linguagem.
- (−) Overhead de processo, desprezível frente ao tempo de treino.

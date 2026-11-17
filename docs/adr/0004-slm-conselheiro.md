# ADR-0004: SLM como conselheiro, estatística como decisora

- **Status:** Aceito
- **Data:** 2026-11-16

## Contexto
LLMs entendem a semântica de colunas (identificam vazamento, atributos sensíveis), mas são não determinísticos, podem alucinar e trazem viés próprio.

## Decisão
- O SLM **classifica colunas, explica achados e sugere ações** em JSON validado por schema.
- O SLM **nunca** seleciona linhas, remove dados ou calcula métricas.
- Toda sugestão do SLM exige confirmação do usuário, e a decisão fica registrada.
- Roda localmente por padrão (mistral.rs + modelo quantizado ~3–4B). API remota só com opt-in.

## Consequências
- (+) Reprodutibilidade (as decisões que valem são estatísticas ou humanas), privacidade e custo zero de API.
- (−) Exige ~3 GB de RAM para o SLM; é opcional e o sistema funciona sem ele.

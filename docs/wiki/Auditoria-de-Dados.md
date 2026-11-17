# Auditoria de Dados

> **Princípio:** a estatística decide e o SLM explica e sugere. O LLM nunca seleciona linhas sozinho (ver [ADR-0004](../adr/0004-slm-conselheiro.md)).

"Dados viciados" cobre problemas bem diferentes. Cada um tem uma técnica certa:

## Checagens

| # | Problema | Como detectamos | Severidade padrão | Ação sugerida |
|---|---|---|---|---|
| A1 | **Vazamento do alvo** (*target leakage*) | Importância de feature isolada (AUC de um modelo com uma feature só ≈ 1,0); correlação/informação mútua extrema; SLM avalia a semântica ("existe só depois do evento?") | 🔴 | Remover a coluna |
| A2 | **Vazamento treino↔teste** por duplicatas | Hash exato + near-duplicates (MinHash/LSH em texto, pHash em imagens) | 🔴 | Deduplicar antes do split |
| A3 | **Vazamento por grupo** | Entidades (ex.: `cliente_id`) aparecendo nos dois lados; SLM sugere qual coluna representa a entidade | 🔴 | Split por grupo (GroupKFold) |
| A4 | **Vazamento temporal** | Existe uma coluna de tempo e o split é aleatório | 🔴 | Split temporal, ou walk-forward em séries |
| A5 | **Desbalanceamento de classes** | Distribuição do alvo | 🟡 | Split estratificado, métrica adequada (PR-AUC/F1), `class_weight` |
| A6 | **Diferença de distribuição** treino↔teste ou treino↔produção | *Adversarial validation* (classificador treino×teste; AUC > 0,6 é alerta); PSI/KS por feature | 🟡 | Investigar; reponderar; split diferente |
| A7 | **Rótulos ruidosos** | *Confident learning* (Cleanlab) com predições out-of-fold | 🟡 | Revisar os top-N suspeitos; nunca remover automaticamente |
| A8 | **Viés de fairness** | Métricas por grupo sensível (paridade demográfica, equal opportunity) via Fairlearn | 🟡 | Relatório; mitigação opcional |
| A9 | **Proxies de atributo sensível** | Uma feature prevê o atributo sensível (ex.: CEP → raça); o SLM aponta proxies semânticos | 🟡 | Avaliar remoção ou mitigação |
| A10 | **Colunas inúteis/perigosas** | IDs, constantes, cardinalidade alta, texto livre com dados pessoais | 🟢/🟡 | Remover ou tratar |
| A11 | **Dados pessoais** (LGPD) | Regex + NER local (CPF, e-mail, telefone, nome) | 🟡 | Mascarar ou remover; registrar |
| A12 | **Amostra pequena demais** | Regras por tarefa (ex.: < 50 por classe); curva de aprendizado | 🟡 | Sugerir pré-treinado ou coletar mais dados |
| A13 | **Nulos/outliers estruturais** | Profiling; padrões de ausência correlacionados com o alvo | 🟢/🟡 | Imputação ou feature indicadora |

## Seleção de dados de treino e teste "não viciados"

O sistema monta o split nesta ordem:

1. **Deduplicar** (A2).
2. **Escolher a estratégia de split** pela auditoria: temporal (A4) > grupo (A3) > estratificado (A5) > aleatório.
3. **Separar o teste primeiro** (padrão: 15–20%). Ele fica guardado e é usado **uma única vez** no final.
4. Treino/validação com k-fold estratificado ou por grupo para seleção de modelo.
5. **Adversarial validation** no split final para confirmar que treino e teste vêm da mesma distribuição.
6. Tudo é registrado: seed, estratégia e hashes de cada partição.

## O papel do SLM

**Onde ele ajuda** (semântica que a estatística não vê):

- Ler nomes, descrições e amostras de colunas e classificar cada uma: `identificador`, `tempo`, `alvo-pós-evento` (suspeita de vazamento), `sensível`, `proxy-sensível`, `texto-livre`, `dado-pessoal`.
- Explicar cada achado do relatório em português simples, com a ação recomendada.
- Sugerir a coluna de grupo e a coluna de tempo.
- Gerar uma descrição legível do dataset para o relatório final.

**Onde ele NÃO entra:**

- Escolher quais linhas vão para treino ou teste.
- Remover dados automaticamente.
- Calcular métricas.

**Saída estruturada** (validada por JSON Schema; se o JSON for inválido, o sistema descarta e cai na regra estatística):

```json
{
  "column": "valor_estornado",
  "role": "alvo-pos-evento",
  "confidence": 0.86,
  "evidence": "Estorno só ocorre após a fraude confirmada; correlação com alvo = 0.97",
  "suggested_action": "drop",
  "requires_user_confirmation": true
}
```

### Modelos candidatos (locais, quantizados)

| Modelo | Tamanho | RAM aprox. (Q4) | Observação |
|---|---|---|---|
| Qwen3 4B | 4B | ~3 GB | Bom em JSON estruturado e em português; **padrão sugerido** |
| Phi-4-mini | 3,8B | ~3 GB | Bom raciocínio para o tamanho |
| Gemma 3 4B | 4B | ~3 GB | Alternativa |
| Llama 3.2 3B | 3B | ~2,5 GB | Leve |
| SmolLM3 3B | 3B | ~2,5 GB | Muito leve, para máquinas fracas |

O modelo é **configurável**. A escolha padrão sai de um benchmark interno de classificação de colunas (issue do M3). Também dá para usar uma API remota (Claude, OpenAI, Ollama) por opção explícita, com aviso de privacidade.

## Benchmark de auditoria

Para medir se a auditoria funciona de verdade, montamos datasets públicos com **problemas plantados** (vazamento, duplicatas, rótulos invertidos, proxies) e medimos a revocação e a precisão de cada checagem. Isso roda na CI.

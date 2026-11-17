# Catálogo de Algoritmos

Cada algoritmo tem **um backend preferido** e, quando existe, um **backend alternativo**. O usuário não precisa saber qual roda: o sistema escolhe pelo tamanho dos dados, pelo hardware e pela disponibilidade.

Legenda de backend: 🦀 nativo Rust (`tempera-native`) · 🐍 Python (`tempera_worker`)

## Tabular: supervisionado

| Algoritmo | Tarefa | Preferido | Alternativo | Milestone |
|---|---|---|---|---|
| Regressão linear / Ridge / Lasso | Regressão | 🦀 linfa | 🐍 sklearn | M1 |
| Regressão logística | Classificação | 🦀 linfa | 🐍 sklearn | M1 |
| Árvore de decisão | Ambas | 🦀 linfa | 🐍 sklearn | M1 |
| Random Forest | Ambas | 🐍 sklearn | 🦀 smartcore | M1 |
| **LightGBM** | Ambas | 🐍 lightgbm | 🦀 perpetual | M1 |
| XGBoost | Ambas | 🐍 xgboost | — | M1 |
| CatBoost | Ambas (categóricas) | 🐍 catboost | — | M2 |
| kNN | Ambas | 🦀 linfa | 🐍 sklearn | M2 |
| SVM | Classificação | 🦀 linfa | 🐍 sklearn | M2 |
| MLP tabular | Ambas | 🦀 Burn | 🐍 PyTorch | M5 |

**Baseline padrão em tabular:** um modelo trivial (classe majoritária ou média), seguido de uma regressão linear/logística, seguido de LightGBM. Na prática, o GBM costuma ser difícil de bater em dados tabulares, e roda em CPU.

## Tabular: não supervisionado

| Algoritmo | Preferido | Milestone |
|---|---|---|
| k-means | 🦀 linfa | M2 |
| DBSCAN | 🦀 linfa | M2 |
| PCA | 🦀 linfa | M2 |
| Isolation Forest (anomalias) | 🐍 sklearn | M2 |

## Séries temporais

| Abordagem | Preferido | Milestone |
|---|---|---|
| Baselines (naive, seasonal naive) | 🦀 nativo | M2 |
| GBM com lags/janelas (feature engineering automático) | 🐍 LightGBM | M2 |
| Modelos de fundação pré-treinados (zero-shot / fine-tune, ex.: Chronos) | 🐍 PyTorch | M5 |

Validação sempre **walk-forward**.

## Visão computacional

| Tarefa | Do zero | Pré-treinado (padrão) | Milestone |
|---|---|---|---|
| Classificação de imagens | CNN pequena (🦀 Burn / 🐍 PyTorch) | Fine-tune ResNet / ConvNeXt / ViT (🐍 timm) | M5 |
| Detecção de objetos | — | Fine-tune de detector moderno (🐍) | pós-M7 |

## Texto / NLP

| Tarefa | Pré-treinado (padrão) | Milestone |
|---|---|---|
| Classificação de texto | Fine-tune de encoder (BERT-like multilíngue) via 🐍 HF Transformers | M5 |
| Embeddings + classificador leve | Encoder congelado + logística (barato!) | M5 |
| Fine-tuning de LLM pequeno (instrução/chat) | LoRA/QLoRA via 🐍 PEFT/Unsloth | M5 |

## "Do zero" × "pré-treinado": regras do recomendador

O recomendador gera uma sugestão **com justificativa**. A pessoa pode sobrescrever.

| Situação | Recomendação |
|---|---|
| Tabular (qualquer tamanho) | Treinar do zero (GBM). Pré-treinado tabular ainda é nicho. |
| Imagens < ~50k exemplos | Fine-tune de um pré-treinado. Do zero tende a ser pior **e** mais caro. |
| Imagens de domínio muito diferente (raio-X, satélite) e > 100k | Testar os dois, começando pelo fine-tune |
| Texto, classificação | Primeiro embeddings congelados + logística (centavos); depois fine-tune do encoder se não bastar |
| LLM | Sempre LoRA/QLoRA. Treinar do zero está fora do escopo. |
| Dados sensíveis em que o pré-treinado vem de fonte desconhecida | Sinalizar o risco de licença/origem e deixar o usuário escolher |

## Busca de hiperparâmetros

- **Padrão:** espaços de busca curados por algoritmo, com poucos trials.
- **Estratégia:** *successive halving / ASHA*, que mata cedo as configurações ruins.
- **Orçamento:** a busca respeita o teto de tempo e custo do job.
- Early stopping em todos os modelos iterativos.

## Métricas e relatório

| Tarefa | Métricas |
|---|---|
| Classificação | AUC-ROC, PR-AUC, F1, acurácia balanceada, log-loss, calibração (ECE), matriz de confusão |
| Regressão | RMSE, MAE, MAPE, R², resíduos |
| Séries temporais | MAE/MASE por horizonte |
| Clusterização | Silhouette, Davies-Bouldin |
| Todas | Fairness por grupo (se houver atributo sensível), custo, tempo |

## Adicionando um algoritmo novo

Um algoritmo é um **plugin** com:

1. Um manifesto (`algorithm.yaml`): nome, tarefas suportadas, backend, espaço de hiperparâmetros, requisitos de memória e se suporta checkpoint.
2. Uma implementação do adaptador: classe Python que segue o protocolo `tempera_worker.Algorithm`, ou struct Rust que implementa `trait Algorithm`.
3. Um teste de contrato contra um dataset pequeno fixo.

Ver [Contribuindo](Contribuindo.md).

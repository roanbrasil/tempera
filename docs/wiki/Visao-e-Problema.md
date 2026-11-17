# Visão e Problema

## Visão

Qualquer pessoa com dados e uma pergunta deve conseguir treinar um modelo **confiável** sem gastar mais do que o necessário, sem montar infraestrutura e sem cair nas armadilhas clássicas de ML.

## O problema

Treinar um modelo hoje exige dominar quatro mundos ao mesmo tempo:

| Mundo | Onde as pessoas se machucam |
|---|---|
| **Dados** | Vazamento entre treino e teste, duplicatas, rótulos errados, desbalanceamento, viés amostral. O modelo "funciona" no notebook e falha em produção. |
| **Modelagem** | Escolher entre um algoritmo clássico e deep learning, entre treinar do zero e fazer fine-tuning, e definir os hiperparâmetros. |
| **Infraestrutura** | CUDA, drivers, versões do PyTorch, máquinas remotas, SSH, Docker. |
| **Custo** | GPU esquecida ligada, instância cara demais para o problema, treino que estoura memória depois de 3 horas pagas. |

As ferramentas existentes cobrem **um ou dois** desses mundos:

- **AutoML** (AutoGluon, H2O, Vertex AutoML): bom em modelagem, mas preso a uma nuvem ou sem controle de onde roda.
- **Orquestração de GPU** (SkyPilot, dstack): bom em infraestrutura e custo, mas não entende de dados nem de modelagem.
- **Tracking** (MLflow, W&B, ClearML): bom em acompanhamento, mas você ainda escreve todo o resto.
- **Qualidade de dados** (Cleanlab, Evidently, Great Expectations): ferramentas isoladas, que exigem um especialista para costurar.

**Ninguém junta os quatro em um fluxo guiado, local-first e com obsessão por custo.**

## Personas

### 1. Dev de produto em PME ("Ana")
- Sabe programar e não é cientista de dados. Tem uma planilha de clientes e quer prever churn.
- **Dor:** não sabe se o resultado é confiável e tem medo de gastar com nuvem.
- **Ganho:** fluxo guiado, auditoria que avisa "a coluna `data_cancelamento` vaza o alvo", baseline que roda na CPU dela por centavos.

### 2. Cientista de dados com home lab ("Bruno")
- Tem um notebook mais um PC com RTX 3090 em casa e às vezes aluga uma A100.
- **Dor:** alternar entre máquinas é manual (scp, tmux, nohup) e ele perde o controle de experimentos e custos.
- **Ganho:** um painel só, com escolha da máquina, ETA, histórico de runs e checkpoint automático.

### 3. Startup com dados sensíveis ("Carla, CTO")
- Dados de saúde ou financeiros. A LGPD impede mandar dados para APIs de terceiros.
- **Dor:** as ferramentas de AutoML na nuvem estão fora de questão e o time é pequeno.
- **Ganho:** tudo on-prem, incluindo o SLM de auditoria, com relatório de fairness para compliance.

### 4. Pesquisador / estudante ("Diego")
- Orçamento curto e quer fazer fine-tuning de um LLM pequeno com LoRA.
- **Dor:** não sabe qual GPU basta e já perdeu dinheiro com OOM.
- **Ganho:** estimativa de memória antes de alugar, GPU spot mais barata que caiba e retomada automática se a instância cair.

## Diferenciais

1. **Auditoria antes do treino.** Bloqueia ou alerta sobre vazamento e viés; nenhuma ferramenta de AutoML faz isso como etapa obrigatória.
2. **Custo como métrica de primeira classe.** Estimativa antes de rodar, teto de orçamento, custo real ao final e comparação entre máquinas.
3. **Qualquer máquina vira worker.** Local, LAN via SSH ou nuvem, com o mesmo fluxo.
4. **Binário único.** O orquestrador e o agente rodam sem ambiente Python, que só é provisionado (com `uv`) quando o job precisa.
5. **Recomendação explicada.** "Use fine-tuning de um ResNet-50 porque você tem 2.000 imagens; treinar do zero precisaria de ~100x mais."

## O que NÃO somos (por enquanto)

- ❌ Não reimplementamos PyTorch em Rust. Usamos o melhor de cada ecossistema (ver [ADR-0001](../adr/0001-hibrido-rust-python.md)).
- ❌ Não somos um notebook nem uma IDE de pesquisa.
- ❌ Não somos plataforma de MLOps de produção em escala (feature store, monitoramento de drift em tempo real). O serving é simples e fica para o M7.
- ❌ Não treinamos LLMs grandes do zero. Fine-tuning com LoRA, sim.

## Como ser bom **e** barato

| Alavanca | Economia típica | Onde está |
|---|---|---|
| Baseline clássico primeiro (GBM na CPU) | Muitos problemas tabulares nem precisam de GPU | [Catálogo](Catalogo-de-Algoritmos.md) |
| Usar hardware que você já tem (local/LAN) | 100% do aluguel | [Execução](Execucao-e-Maquinas.md) |
| Instâncias spot/interruptíveis + checkpoint | 50–80% vs. sob demanda | [Custo](Custo-e-Estimativa.md) |
| Estimativa de memória → menor GPU que cabe | Evita alugar GPU 2–4x maior | [Custo](Custo-e-Estimativa.md) |
| Auditoria evita runs inúteis | Evita retreinos inteiros | [Auditoria](Auditoria-de-Dados.md) |
| Early stopping + HPO com ASHA/successive halving | 3–10x menos compute em busca de hiperparâmetros | [Catálogo](Catalogo-de-Algoritmos.md) |
| Fine-tuning (LoRA/QLoRA) em vez de treino do zero | Ordens de grandeza | [Catálogo](Catalogo-de-Algoritmos.md) |
| Auto-desligamento + teto de orçamento | Elimina "esqueci a GPU ligada" | [Custo](Custo-e-Estimativa.md) |
| Armazenamento sem taxa de egress (ex.: R2) | Elimina custo de download de artefatos | [Stack](Stack-Tecnologica.md) |

## Métricas de sucesso do produto

- **Time-to-first-model:** < 10 min do upload até o primeiro modelo avaliado (tabular, local).
- **Custo por experimento:** comparado com o mesmo job "ingênuo" (sob demanda, GPU grande), reportado ao usuário.
- **Erro do ETA:** < 15% depois do warm-up.
- **Erro da estimativa de custo:** < 20%.
- **Taxa de problemas de dados detectados** antes do treino, medida em datasets de benchmark com vazamento plantado.

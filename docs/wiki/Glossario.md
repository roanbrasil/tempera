# Glossário

| Termo | Significado |
|---|---|
| **Adversarial validation** | Treinar um classificador para distinguir treino de teste. Se ele consegue (AUC alta), as distribuições diferem. |
| **Agente** | Binário `tempera-agent` que roda no worker e executa jobs. |
| **ASHA** | *Asynchronous Successive Halving*: busca de hiperparâmetros que interrompe cedo as configurações ruins. |
| **Backend** | Implementação que realmente treina: nativa (Rust) ou Python. |
| **Baseline** | O modelo mais simples possível; define a régua mínima. |
| **Checkpoint** | Estado salvo do treino que permite retomar. |
| **Confident learning** | Técnica (Cleanlab) para achar rótulos provavelmente errados. |
| **Do zero / from scratch** | Treinar com pesos aleatórios. |
| **EMA** | Média móvel exponencial (usada no ETA). |
| **Executor** | Abstração de "onde roda": local, SSH, nuvem. |
| **Fine-tuning** | Ajustar um modelo pré-treinado aos seus dados. |
| **Group split** | Split em que a mesma entidade nunca aparece em treino e teste ao mesmo tempo. |
| **Job** | Uma execução definida por um `tempera.yaml`. Gera um ou mais *runs*. |
| **LoRA / QLoRA** | Fine-tuning que treina só matrizes pequenas adicionais (QLoRA: com o modelo base em 4 bits). |
| **MFU** | *Model FLOPs Utilization*: fração do pico teórico da GPU efetivamente usada. |
| **Preempção** | Quando o provedor retoma uma instância spot. |
| **Reaper** | Rotina que encerra instâncias de nuvem órfãs. |
| **Run** | Uma tentativa de treino (um algoritmo + hiperparâmetros) dentro de um job. |
| **SLM** | *Small Language Model*: LLM pequeno que roda localmente. |
| **Spot / interruptível** | Instância mais barata que pode ser retomada pelo provedor. |
| **Target leakage** | Feature que contém informação do alvo indisponível no momento da predição. |
| **Worker** | Máquina que executa jobs. |

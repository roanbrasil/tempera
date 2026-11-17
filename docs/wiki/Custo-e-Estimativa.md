# Custo e Estimativa

O Têmpera trata **custo** e **tempo** como métricas de primeira classe, e estima os dois **antes** de gastar.

## 1. Estimativa de memória (cabe ou não cabe?)

Rodar com OOM depois de 40 minutos pagos é o desperdício mais comum. A estimativa é feita antes de escolher a máquina.

### Deep learning

```
mem_total ≈ pesos + gradientes + estados_do_otimizador + ativações + overhead

pesos         = P × bytes_por_param                 (fp32=4, bf16=2, int4≈0,5)
gradientes    = P_treináveis × bytes_grad
otimizador    = P_treináveis × 8   (Adam: 2 estados fp32)
ativações     ≈ f(arquitetura, batch, seq_len)      → medida por profiling
overhead      ≈ 10–20% + contexto CUDA (~0,5–1 GB)
```

- **LoRA/QLoRA:** os parâmetros treináveis são uma fração mínima do total, e o modelo base pode estar em 4 bits. É isso que viabiliza LLM de 7–8B em uma GPU de 24 GB.
- **Calibração por profiling:** sempre que possível, roda 2–3 steps em uma máquina disponível (até em CPU com batch reduzido) para medir as ativações de verdade em vez de chutar.

### Tabular

Uma heurística por algoritmo: `linhas × colunas × bytes × fator` (o fator do GBM depende do binning). Em geral cabe na CPU.

## 2. Estimativa de tempo (ETA)

Três fontes, combinadas:

1. **A priori** (antes de rodar, para comparar máquinas):
   - Transformers: `FLOPs ≈ 6 × P × tokens` no treino completo, e `2 × P × tokens` para o forward. O tempo sai de `FLOPs / (pico_da_GPU × MFU)`, com MFU (utilização efetiva) inicial de 30–40%.
   - Outros modelos: tabela de benchmark interna (algoritmo × hardware × tamanho), alimentada pelos próprios runs.
2. **Warm-up** (primeiros N steps, descartando o primeiro): mede o throughput real e extrapola.
3. **Online:** média móvel exponencial do tempo por step, mais o tempo de avaliação/checkpoint, mais o tempo de boot e download na nuvem.

`ETA = boot + preparo_de_env + download + steps_restantes × t_step + avaliações + upload_final`

O erro do ETA vira métrica do produto (meta: < 15% depois do warm-up).

## 3. Estimativa de custo

```
custo = preço_hora × (ETA + boot + margem) + armazenamento + egress
```

- **Catálogo de preços:** sincronizado periodicamente pelas APIs dos provedores (RunPod, Vast...) e guardado com data, porque preço muda.
- **Máquinas próprias:** custo zero por padrão, com opção de informar o custo de energia (W × R$/kWh) para quem quer ser exato.
- **Normalização por throughput:** "R$ por job" vale mais que "R$ por hora". Uma GPU 2x mais cara e 3x mais rápida sai mais barata no total.

## 4. Recomendação de máquina

Para um job com requisito de memória M e ETA estimado:

1. Filtra as máquinas em que o job **cabe** (com margem).
2. Calcula ETA e custo de cada uma.
3. Ordena conforme o objetivo do usuário:
   - `cheapest`: menor custo total
   - `fastest`: menor ETA
   - `cheapest_under: 2h`: o mais barato que termina em menos de 2 h (padrão)
4. Mostra a justificativa ("a A100 é 1,4x mais rápida mas 3x mais cara para este job").

## 5. Estratégias de economia (ligadas por padrão)

| Estratégia | Detalhe |
|---|---|
| **Baseline primeiro** | Modelos baratos definem a régua; o deep só roda se houver chance de ganho |
| **Local/LAN primeiro** | Se houver máquina própria que cabe e termina no prazo, ela vem primeiro |
| **Spot + checkpoint** | Padrão na nuvem para jobs com mais de 15 min; retomada automática |
| **Early stopping** | Em todo modelo iterativo |
| **ASHA no HPO** | Mata cedo as configurações ruins |
| **Precisão mista (bf16)** | Padrão nas GPUs que suportam |
| **Cache de dados e de ambiente** | Não baixa nem reinstala o que já está lá |
| **Desligamento automático + reaper** | Nenhuma GPU fica ligada à toa |
| **Teto de orçamento** | Rígido, com checkpoint ao atingir |
| **Armazenamento sem egress** | Artefatos em S3-compatível sem taxa de saída (ex.: Cloudflare R2), ou local |

## 6. Transparência

Ao fim de todo job, o relatório mostra:

```
Custo estimado: R$ 1,80   Custo real: R$ 1,62   (−10%)
ETA estimado:   38 min    Real:       41 min    (+8%)
Economia vs. execução ingênua (A100 sob demanda, sem early stopping): R$ 9,40 (85%)
```

A linha de **economia** é o argumento de venda do produto.

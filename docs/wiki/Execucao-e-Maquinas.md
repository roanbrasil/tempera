# Execução e Máquinas

## A trait `Executor`

```rust
#[async_trait]
pub trait Executor: Send + Sync {
    /// Descreve hardware, preço e disponibilidade.
    async fn describe(&self) -> Result<MachineInfo>;
    /// Garante que a máquina existe e está acessível (no caso da nuvem, aluga).
    async fn provision(&self, req: &ResourceRequest) -> Result<MachineHandle>;
    /// Instala/atualiza o tempera-agent e o inicia apontando para o server.
    async fn bootstrap_agent(&self, m: &MachineHandle, cfg: &AgentConfig) -> Result<()>;
    /// Libera a máquina (no caso da nuvem, destrói a instância).
    async fn release(&self, m: &MachineHandle) -> Result<()>;
}
```

Implementações:

| Executor | Provisionamento | Bootstrap do agente | Liberação |
|---|---|---|---|
| `LocalExecutor` | Nenhum | Inicia o agente como processo filho | Mata o processo |
| `SshExecutor` | Nenhum (máquina existente) | `scp` do binário (verifica hash/versão) + `ssh` para iniciar; túnel reverso para o server | Para o agente |
| `CloudExecutor<P: Provider>` | API do provedor (cria o pod/instância) | Imagem base com o agente pré-instalado **ou** cloud-init | API: destrói a instância |

## O agente (`tempera-agent`)

Binário Rust estático (musl), com poucos MB.

Responsabilidades:

1. **Inventário:** CPU, RAM, GPUs (via NVML), versão de driver/CUDA, disco livre. Enviado no registro.
2. **Conexão de saída** para o server via gRPC sobre TLS, o que funciona atrás de NAT. Em SSH, a conexão passa pelo port-forward do próprio SSH, então nenhuma porta precisa ser aberta.
3. **Preparação de ambiente:**
   - Dados: baixa do server/S3 com cache por hash.
   - Python (só se o job pedir): `uv` cria o venv a partir do lockfile do worker, cacheado por hash do lockfile. Com o cache quente, isso leva segundos.
   - Backend nativo: nada a preparar.
4. **Execução:** inicia o backend como subprocesso. O backend escreve eventos JSONL no stdout (métricas, logs, checkpoint pronto) e o agente repassa.
5. **Heartbeat** a cada N segundos. Se o server ficar mudo por muito tempo, o agente continua treinando e salvando checkpoints localmente, e sincroniza quando a conexão volta.
6. **Checkpoint:** recebe "checkpoint pronto" do backend e faz upload assíncrono.
7. **Telemetria de hardware:** uso de GPU, VRAM, temperatura e potência, para o painel e para calibrar o ETA.
8. **Sinal de preempção:** quando o provedor avisa que a instância spot vai cair, pede um checkpoint de emergência.

## Máquinas na rede local (SSH)

Cadastro:

```yaml
machines:
  - name: pc-sala
    kind: ssh
    host: 192.168.0.42
    user: bruno
    key: ~/.ssh/id_ed25519      # ou o ssh-agent
    tags: [gpu, rtx3090]
```

- **Descoberta opcional** via mDNS: agentes já instalados se anunciam na LAN.
- Health-check e inventário são feitos no cadastro.
- Pré-requisito no host: Linux ou macOS, mais o driver NVIDIA se houver GPU. **Não** precisa de Python nem de Docker: o `uv` é baixado sob demanda.
- Windows: suporte via WSL2 (fase posterior).

## Nuvem (GPU por API key)

Provedores em ordem de prioridade (critério: preço, API boa, spot):

| Provedor | Por que | Fase |
|---|---|---|
| **RunPod** | API simples, GPUs baratas, spot ("community") | M6 |
| **Vast.ai** | Marketplace mais barato; exige filtro de confiabilidade | M6 |
| Lambda | Bom preço sob demanda | pós-M6 |
| AWS / GCP / Azure (spot) | Para empresas que já têm conta e créditos | pós-M6 |
| Qualquer Kubernetes | Clusters próprios | futuro |

Trait `Provider`:

```rust
#[async_trait]
pub trait Provider {
    async fn list_offers(&self, req: &ResourceRequest) -> Result<Vec<Offer>>; // preço, GPU, VRAM, spot?, região
    async fn launch(&self, offer: &Offer, boot: &BootSpec) -> Result<InstanceId>;
    async fn status(&self, id: &InstanceId) -> Result<InstanceStatus>;
    async fn terminate(&self, id: &InstanceId) -> Result<()>;
}
```

## Checkpoint e retomada

- Todo backend iterativo precisa suportar `save_checkpoint` e `resume_from` (é requisito do plugin).
- Frequência adaptativa: em spot, salva a cada ~5–10 min de trabalho; em máquina própria, por época.
- Se a instância cair, o server detecta (heartbeat ou status no provedor), aluga outra oferta compatível e retoma do último checkpoint. Isso fica registrado no run e aparece na UI.
- Política configurável: `max_restarts` e `fallback_to_on_demand: true|false`.

## Guarda-costas (custo)

- **Teto de orçamento por job:** ao atingir 100%, salva o checkpoint e encerra. Avisa aos 80%.
- **Desligamento automático** de instâncias ociosas: nenhum job há X minutos e o server destrói.
- **Reaper:** um job periódico lista as instâncias abertas em cada provedor com a tag `tempera` e encerra órfãs. Isso protege contra crash do server.
- **Conciliação de custo:** compara o custo calculado pelo Têmpera com o que o provedor reporta.

## Escalonamento (fase posterior)

- Fila de jobs por máquina, com prioridade.
- Vários jobs pequenos na mesma GPU (HPO em paralelo).
- Treino distribuído multi-GPU (DDP/FSDP via PyTorch) só depois do M7.

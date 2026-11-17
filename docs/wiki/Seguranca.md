# Segurança e Privacidade

## Modelo de ameaça (resumo)

| Ativo | Ameaça | Mitigação |
|---|---|---|
| Dados do usuário | Vazamento para terceiros | Local-first; o SLM roda localmente; uso de API remota só com opt-in explícito e aviso |
| API keys de nuvem | Roubo, uso indevido | Keyring do SO ou arquivo criptografado (age); nunca em logs nem no `tempera.yaml` |
| Chaves SSH | Comprometimento de máquinas da LAN | Usa o `ssh-agent`; nunca copia a chave privada; prefere uma chave dedicada com acesso restrito |
| Canal agente↔server | MITM | TLS (rustls) com certificado do server fixado no bootstrap; token de agente de uso único que vira credencial de sessão |
| Modelos pré-treinados baixados | Código malicioso (pickle) | Só **safetensors**/ONNX; `.bin`/pickle bloqueado por padrão |
| Código de plugins | Execução arbitrária | Plugins são código confiável (instalados pelo usuário); avisar na instalação |
| Instâncias de nuvem | Dados residuais | Destrói a instância ao fim; disco efêmero; opção de criptografar os dados em trânsito e em repouso |

## Segredos

- `tempera secrets set runpod.api_key`: armazena no keyring do SO (crate `keyring`).
- Fallback em servidor sem keyring: arquivo criptografado com `age`, cuja chave vem de uma variável de ambiente.
- Segredos são injetados no agente só quando necessário, como variáveis de ambiente do processo do backend, nunca escritos em disco.
- Os logs passam por redação automática de padrões de segredo.

## LGPD / dados pessoais

- A auditoria A11 detecta dados pessoais (CPF, e-mail, telefone, nomes) e sugere mascarar.
- O relatório final lista quais colunas pessoais foram usadas, removidas ou mascaradas, o que serve de evidência de minimização.
- Modo **"nada sai daqui"**: bloqueia executores de nuvem e SLM remoto para um projeto inteiro.
- Retenção: comando para apagar o dataset, os splits, os checkpoints e os artefatos derivados (direito de eliminação).

## Autenticação (modo servidor de time)

- Fase 1: single-user, bind em `localhost`.
- Fase 2: usuários locais (argon2) + tokens de API; OIDC opcional.
- Papéis: admin (máquinas e segredos) e membro (jobs).

## Supply chain

- `cargo deny` (licenças e advisories) e `cargo audit` na CI.
- Lockfile do worker Python com hashes (`uv lock`).
- Binário do agente assinado (minisign); o executor verifica a assinatura antes de instalar no worker.

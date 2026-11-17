# Têmpera Wiki

> Treine modelos de ML do jeito certo, na máquina mais barata que resolve.

## Comece aqui

1. **[Visão e Problema](Visao-e-Problema.md)**: para quem é, que dor resolve, o que não fazemos.
2. **[Jornada do Usuário](Jornada-do-Usuario.md)**: a experiência ponta a ponta.
3. **[Arquitetura](Arquitetura.md)**: como o sistema é montado.
4. **[Roadmap](Roadmap.md)**: em que ordem vamos construir.

## Referência técnica

- [Auditoria de Dados](Auditoria-de-Dados.md): viés, vazamento, qualidade; papel do SLM
- [Catálogo de Algoritmos](Catalogo-de-Algoritmos.md): algoritmos × backends
- [Execução e Máquinas](Execucao-e-Maquinas.md): local, SSH/LAN, nuvem, agente
- [Custo e Estimativa](Custo-e-Estimativa.md): ETA, memória, custo, economia
- [Contratos e Especificações](Contratos.md): `tempera.yaml`, gRPC, Arrow
- [Segurança e Privacidade](Seguranca.md)
- [Stack Tecnológica](Stack-Tecnologica.md)
- [Decisões de Arquitetura (ADRs)](../adr/README.md)
- [Glossário](Glossario.md)
- [Contribuindo](Contribuindo.md)

## Os 5 princípios

1. **Correto antes de rápido.** Um modelo treinado em dados vazados é dinheiro jogado fora. A auditoria vem antes do treino.
2. **Barato por padrão.** Começa no baseline mais simples, usa a máquina que você já tem e só sobe para GPU paga quando os números justificam.
3. **Estatística decide, LLM explica.** O SLM é conselheiro e nunca seleciona dados sozinho.
4. **Melhor ferramenta para cada parte.** Rust para sistema, Python para o ecossistema de ML, com contratos claros entre os dois.
5. **Local-first e privado.** Nada sai da sua infraestrutura sem você mandar.

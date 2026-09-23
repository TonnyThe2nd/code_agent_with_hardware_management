# Agente de código local

## Planejamento em camadas

A execução de subtarefas usa `--action-mode structured` por padrão. O Ollama
recebe um JSON Schema em `format`, e cada resposta deve ser uma ação conhecida
com argumentos válidos ou `{"action":"final","content":"..."}`. A validação
exige uma ação antes de liberar resposta final; se há ferramentas de leitura,
a primeira ação fica restrita a `read_file` ou `list_dir`. Isso comprova uma
interação com o workspace, não que toda a implementação foi feita. A validação
local rejeita texto em volta, ferramentas desconhecidas e argumentos de tipos
incorretos. As ações passam pela mesma SafetyPolicy e ToolRegistry; não há eval
nem execução de exemplos extraídos de texto. Use `--action-mode native` para
voltar ao protocolo nativo de tool_calls. O planejador continua gerando seu
JSON de subtarefas separadamente; `agent run` mantém seu protocolo nativo.

Dentro de `backend`, execute:

```powershell
python -m src.main agent plan "Implemente validacao de entrada e testes" --workspace .
```

`config/models.yaml` define os tiers executor e planner. `--catalog CAMINHO`
permite outro catálogo e `--timeout 180` ajusta a espera pela decomposição.
O maior planner que cabe é escolhido para decompor; sem planner que caiba,
usa o maior executor. O modelo escolhido precisa estar instalado no Ollama local.
Use `ollama pull NOME:TAG` para instalar os modelos do catálogo.
Não há download automático nem substituição silenciosa de um planner ausente.

O roteamento usa exatamente `size_gb <= budget_gb`, conforme o orçamento do
contexto hardware: simples vai para o menor executor, moderada para o maior
executor, complexa para o maior planner (ou maior executor como fallback).
Essa regra difere da estimativa conservadora de `agent run`; tamanho Q4 não
garante memória suficiente para inferência. Os valores do YAML são estimativas.
O LLM decompõe e classifica complexidade; a escolha de modelos é código puro.

Sem `--execute`, gera apenas o plano. Para executar as subtarefas:

```powershell
python -m src.main agent plan "Implemente validacao de entrada e testes" --workspace . --execute --max-iter 10 --timeout 180
```

Todos os modelos atribuídos precisam estar instalados e declarar `tools` antes
da primeira subtarefa. A execução usa as ferramentas reais, um modelo/subtarefa
por vez, compartilhando o workspace e passando as respostas das dependências.
O planejador não lê arquivos; cada executor recebe instruções para inspecioná-los.
`keep_alive=0` solicita descarregamento após cada requisição, inclusive planejamento;
isso pode tornar a execução mais lenta. Não descarrega modelos carregados por outros clientes.
`--max-iter` vale por subtarefa. Falha HTTP, erro de ferramenta (verificado ao fim
da iteração) ou limite atingido interrompe o plano com código 2, sem iniciar as
próximas subtarefas. Alterações anteriores não são revertidas automaticamente.
A conclusão de uma sessão é declarada pelo modelo; não comprova correção do código.
No modo de execução do plano, respostas sem ferramentas reais recebem uma
tentativa de correção. Persistindo a ausência de ações, o plano é interrompido.
Se nenhuma ferramenta chegou a ser invocada, a execução tenta o próximo modelo
maior do catálogo que esteja instalado, suporte tools e caiba no orçamento.
Cada candidato é tentado no máximo uma vez por subtarefa, em ordem crescente de
parâmetros. A CLI mostra a troca; o plano original continua mostrando a atribuição
inicial. Falhas de ferramentas, HTTP e limite de iterações não acionam esse fallback,
pois podem ter ocorrido efeitos no workspace. Sem candidato, a execução para.
JSON de ferramenta no texto e patches de exemplo não são executados automaticamente.
A CLI informa os caminhos escritos por `write_file`; escrever conteúdo idêntico
não é contado como alteração. Essa lista registra escritas, não substitui um
diff final (um arquivo pode ser escrito e depois restaurado durante a execução).
Se nenhuma escrita ocorreu, a CLI informa explicitamente que não há alterações
registradas, mesmo que as sessões de leitura tenham terminado.
Use `--allow-tests` para habilitar pytest em Docker, com `--test-image` opcional.
Sem essa opção, o agente não pode executar pytest. Revise o diff e os testes.
Modelos atribuídos mas não instalados são apontados em amarelo. Um modelo de
planejamento sem tools pode gerar JSON; isso não garante que possa executar
futuramente as ferramentas do agente. Mais de dois modelos gera um aviso,
não um bloqueio. `min_executor_params_b` fica reservado na política: o plano
contém nomes, não metadados suficientes para validar esse limite.
Sem PyYAML, o loader avisa e usa o catálogo JSON embutido, ignorando o YAML solicitado.

Python 3.11+, DDD e Clean Architecture. O contexto `hardware` detecta recursos;
`agent` conversa com modelos Ollama e oferece leitura, escrita, listagem e comandos controlados.

## Instalação no Windows (PowerShell)

Na raiz do repositório:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd backend
..\.venv\Scripts\python.exe -m src.main --help
..\.venv\Scripts\python.exe -m src.main hardware --json
..\.venv\Scripts\python.exe -m pytest -q
```

Não é necessário ativar o ambiente virtual. No Linux/macOS, use `.venv/bin/python`.

## Ollama e múltiplos modelos

Instale e inicie o Ollama fora do projeto: https://ollama.com/download.
Se o serviço não estiver ativo, execute `ollama serve` em outro terminal.

```powershell
ollama list
python -m src.main agent models
python -m src.main agent run "Liste os arquivos do workspace" -m qwen2.5:7b -w .
```

Use o Python do ambiente virtual nos exemplos se ele não estiver ativado.
O projeto não baixa modelos automaticamente. Para instalar outro modelo:

```powershell
ollama pull NOME:TAG
python -m src.main agent models
python -m src.main agent run "Analise o projeto" -m NOME:TAG
```

Escolha modelos com capacidade `tools` na listagem. Modelos de embedding não
executam o loop de ferramentas; um nome contendo `coder` não garante suporte a tools.
O catálogo consulta `/api/tags` e `/api/show`, conforme a API oficial:
https://docs.ollama.com/api e https://docs.ollama.com/capabilities/tool-calling.

Precedência: `--model` > `CODE_AGENT_MODEL` > seleção automática por hardware.
Sem modelo explícito, o projeto usa `DetectHardwareUseCase` e seu orçamento de
RAM/VRAM; na rota de RAM, limita também a 85% da memória disponível no momento.
Filtra os modelos instalados com `tools` e contexto suficiente e escolhe o maior
número de parâmetros que caiba na estimativa. O teto automático é 250 bilhões
de parâmetros (inclui 235B). Se os parâmetros não forem informados pelo Ollama,
o tamanho é usado como desempate, sem comprovação desse teto para o modelo.
A heurística é `1,25 × tamanho em GiB + max(1, contexto / 4096) GiB`.
Ela não garante ausência de OOM nem identifica o melhor modelo por benchmark;
quantização, arquitetura, KV cache e outras aplicações afetam a memória real.
Não soma RAM com VRAM e não verifica VRAM livre neste momento.
Se nenhum modelo couber, informa o problema em vez de selecionar um incompatível.
Não troca de modelo silenciosamente nem repete uma sessão que já escreveu arquivos.
Cada sessão usa um modelo. Você pode alternar entre sessões; não há roteamento
automático de tarefas entre vários modelos nem suporte implementado a outros provedores.

```powershell
$env:CODE_AGENT_MODEL = "qwen2.5:7b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:CODE_AGENT_TIMEOUT = "180"
python -m src.main agent run "Leia requirements.txt e resuma as dependências" --max-iter 10
```

Também existem `--base-url` e `--timeout`. Use apenas servidores de confiança:
mensagens e conteúdos lidos são enviados ao endpoint escolhido. As variáveis
acima valem para o terminal atual; nenhum arquivo `.env` é carregado automaticamente.
O tamanho em disco mostrado pelo catálogo não estima toda a RAM/VRAM necessária.

```powershell
python -m src.main agent run "Analise o projeto" --num-ctx 4096
python -m src.main agent run "Analise o projeto" --base-url http://SERVIDOR:11434 --memory-budget 180 --timeout 600
```

`--memory-budget` representa memória **utilizável**, em GiB, já descontada a
reserva do servidor. Para endpoint remoto, é obrigatório informar esse orçamento
ou escolher `--model`; o hardware local não representa o servidor remoto.
Mesmo um endpoint localhost pode ser um túnel: nesse caso informe o orçamento
do servidor explicitamente. `--num-ctx` é enviado ao Ollama (padrão 4096).
Escolher `--model` ou `CODE_AGENT_MODEL` ignora a seleção por memória, mantendo
a verificação de instalação e suporte a ferramentas. Não há download automático.

## Execução de testes pelo agente

Por padrão, só `python --version` e `git --version` são aceitos, sem shell.
Não há execução arbitrária no host. Para habilitar `python -m pytest`, instale
Docker Desktop, inicie o mecanismo de containers Linux e construa a imagem:

```powershell
docker build -f Dockerfile.tests -t code-agent-tests:local .
python -m src.main agent run "Execute python -m pytest e relate o resultado" --allow-tests -w .
```

O build precisa de rede para baixar Python e dependências; a execução de testes
não tem rede. O workspace é montado somente para leitura, com usuário sem privilégios,
limites de memória, CPU e processos, filesystem raiz somente leitura e `/tmp` temporário.
O timeout do comando é 10 segundos; o container é removido ao terminar ou expirar.
Não há fallback para executar testes no host se Docker falhar.
Para outro projeto, construa uma imagem com as dependências dele e passe
`--test-image NOME` ou `CODE_AGENT_TEST_IMAGE`. Use imagens de confiança.
Testes que exigem rede, escrita no repositório ou mais de 10 segundos precisam
de um fluxo externo adequado; não são compatíveis com esse modo restrito.

## Limites e verificação

`read_file` lê até 50.000 bytes por padrão. Resultados de comandos são limitados
a 10.000 caracteres, e sua prévia na CLI a 200. O limite de iterações encerra a
sessão com código 2; erros de configuração/conexão retornam 1.
Escritas são feitas diretamente: use controle de versão e revise o diff.
A resolução de caminhos bloqueia saídas do workspace e links simbólicos externos,
mas não constitui isolamento contra alterações concorrentes de links/hard links
por outro processo. Use workspaces locais sob seu controle.

```powershell
python -B -m pytest -q -p no:cacheprovider
python -m src.main agent models
python -m src.main agent run "Use list_dir em . e resuma os nomes" -m qwen2.5:7b
```

Os testes automatizados simulam HTTP e Docker; o último comando é uma verificação
real com o modelo instalado. Não é necessário Ollama ou Docker para rodar a suíte.

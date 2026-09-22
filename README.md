# Agente de código local

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

Precedência: `--model` > `CODE_AGENT_MODEL` > seleção automática quando existe
exatamente um modelo compatível. Se houver vários, o agente pede uma escolha
explícita; não troca de modelo silenciosamente nem repete uma sessão que já escreveu arquivos.
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

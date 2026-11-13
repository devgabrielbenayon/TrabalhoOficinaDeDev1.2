# Recomendador de Cartas de Magic: The Gathering

Projeto acadêmico de sistemas de recomendação. A partir de um deck ou de cartas
escolhidas pelo usuário, recomenda cartas com filtragem baseada em conteúdo (FBC),
similaridade do cosseno e explicações dos fatores que contribuíram para cada resultado.

Magic: The Gathering é propriedade da Wizards of the Coast. Este projeto de fã não é
afiliado nem endossado pela Wizards of the Coast. As imagens são servidas pelo Scryfall.

## Requisitos e instalação

- Python 3.10 ou superior (testado com Python 3.12).
- Git instalado e acesso à internet para baixar os dados, o repositório de decklists e
  o modelo MiniLM na primeira execução.
- Espaço livre suficiente para os arquivos de dados e o cache do modelo; reserve
  aproximadamente 3 GB.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

No macOS/Linux, ative o ambiente com `source .venv/bin/activate`.
As dependências fixam as versões principais e instalam a variante CPU do PyTorch.

## Execução

```powershell
# Baixa os dados, processa, divide, ajusta e avalia o sistema
python run_all.py

# Usa os arquivos já presentes em data/raw
python run_all.py --skip-download

# Interface: execute após concluir o pipeline
streamlit run app/streamlit_app.py

# Testes automatizados
python -m pytest
```

O pipeline registra a duração de cada etapa e interrompe com erro se algum comando
falhar. O download é idempotente: arquivos brutos que já existem são reutilizados.
`--skip-download` não pula processamento e requer os arquivos de Scryfall e
MTGODecklistCache em `data/raw`.

Etapas reproduzíveis também podem ser executadas individualmente:

```text
python -m data.download
python -m src.load
python -m src.features
python -m src.split
python -m src.evaluate
```

O primeiro ciclo de avaliação baixa o modelo `sentence-transformers/all-MiniLM-L6-v2`
e o armazena em `data/processed/embeddings.npy`. Execuções seguintes reutilizam o
cache quando modelo, quantidade e texto ordenado do catálogo correspondem.

## Pipeline e protocolo de avaliação

### Representações

Cada carta tem cinco blocos: texto das regras (TF-IDF ou embeddings MiniLM), tipos,
cores/identidade de cor, keywords e atributos numéricos. Os blocos categóricos e de
texto são normalizados por linha; seus pesos vêm de `config.yaml`. O bloco numérico
preserva magnitudes conforme `features.numeric_norm`.

O TF-IDF usa texto limpo, incluindo stemming. MiniLM recebe o `oracle_text` original,
sem stemming; seus vetores são L2-normalizados e substituem somente o bloco de texto.
Tipos, cores, keywords e atributos numéricos permanecem iguais. Blocos e nomes de
features são salvos em `data/processed` para evitar reconstruções na interface.

### Protocolo sem vazamento

1. **Carga:** Scryfall fornece os metadados; decklists do MTGO são filtradas pelo
   formato, período e regras em `config.yaml`. Básicos e cartas não resolvidas são
   excluídos da matriz de interações.
2. **Split:** decks com poucas cartas distintas são removidos. Uma fração de decks é
   separada para validação; o restante compõe a avaliação. Em cada deck, cartas
   distintas são divididas entre treino e teste.
3. **Perfil:** média ponderada das cartas de treino, usando
   `min(cópias, limite) / limite`, seguida de normalização L2.
4. **Ranking:** cosseno entre o perfil e cada carta do catálogo. Cartas do treino são
   excluídas; o filtro de identidade de cor é opcional.
5. **Ajuste:** os pesos dos blocos e λ de MMR são selecionados somente na validação.
   TF-IDF e MiniLM têm buscas independentes de pesos. MMR reordena os 100 candidatos
   mais bem pontuados por TF-IDF, maximizando
   `λ · sim(perfil, carta) − (1 − λ) · max_sim(carta, selecionadas)`.
6. **Avaliação:** métodos e hiperparâmetros selecionados são comparados nos decks
   reservados, com Precision, Recall e nDCG em N = 5 e 10, cobertura do catálogo e
   diversidade intra-lista. A avaliação não participa da escolha de pesos nem de λ.

O argumento de ausência de vazamento é que o perfil é calculado exclusivamente com
cartas de treino; os pesos e λ são escolhidos em decks de validação separados; e os
vetores de cartas dependem apenas de metadados/texto do catálogo, não de interações
nem de cartas de teste. A validação não é usada para calcular métricas finais.

### Arquivos gerados

| Caminho | Conteúdo |
|---|---|
| `data/processed/*.parquet` | Catálogo, interações e split |
| `data/processed/block_*.npz` | Blocos de features pré-computados |
| `data/processed/embeddings.npy` | Embeddings MiniLM em cache |
| `results/weights_search.csv` | Busca de pesos TF-IDF na validação |
| `results/weights_search_emb.csv` | Busca de pesos MiniLM na validação |
| `results/mmr_lambda_search.csv` | Escolha de λ na validação |
| `results/metrics.md`, `metrics.csv` | Tabela final de métricas |
| `results/fig_metrics.png` | Métricas comparativas |
| `results/fig_mmr_tradeoff.png` | nDCG@10 × diversidade para λ = 1.0, 0.9, 0.8, 0.7 e 0.5 |
| `results/qualitative.md` | Recomendações e explicações qualitativas |
| `results/params.json` | Configuração, versões e hiperparâmetros selecionados |

### Resultados

Os números dependem da versão dos dados obtidos e são gerados localmente pelo
pipeline, não devem ser preenchidos com valores estimados. Consulte a tabela gerada
em [`results/metrics.md`](results/metrics.md); ela contém Precision, Recall, nDCG,
cobertura e diversidade em N = 5 e N = 10 para `random`, `popular`, `popular_color`,
`cbf_tfidf`, `cbf_tfidf_color`, `cbf_tfidf_mmr` e `cbf_emb`.

## Interface

`streamlit run app/streamlit_app.py` disponibiliza duas abas:

- **Deck do dataset:** escolhe um deck de avaliação, mostra o treino e recomenda
  cartas com imagem Scryfall, score e explicação expansível.
- **Monte seu deck:** seleciona cartas do catálogo e obtém recomendações na hora.

A barra lateral permite ajustar N, filtro de cor, TF-IDF/MiniLM e MMR/λ. A aplicação
carrega os artefatos pré-computados e não executa treinamento de features por consulta.
O primeiro carregamento do catálogo pode levar mais tempo; o cache de recursos do
Streamlit é usado nas consultas subsequentes.

## Estrutura

```text
config.yaml             parâmetros e caminhos
data/download.py        download de Scryfall e decklists
src/                    carga, features, split, recomendação e avaliação
app/streamlit_app.py    interface Streamlit
tests/                  testes automatizados
docs/                   rascunhos para relatório e slides
results/                tabelas e figuras geradas (não inclui dados brutos)
run_all.py              pipeline completo
```

Rascunhos de relatório e slides: [metodologia](docs/metodologia.md),
[exemplo numérico](docs/exemplo_numerico.md), [limitações](docs/limitacoes.md) e
[análise dos resultados](docs/analise_resultados.md).

## Equipe e apresentação

| Membro | Papel |
|---|---|
| *Preencher pela equipe* | *Preencher pela equipe* |
| *Preencher pela equipe* | *Preencher pela equipe* |

**Vídeo de apresentação:** *inserir o link da equipe*.

## Fontes e créditos

- Cartas: [Scryfall Bulk Data](https://scryfall.com/docs/api/bulk-data), conjunto
  `oracle_cards`.
- Decklists: [Jiliac/MTGODecklistCache](https://github.com/Jiliac/MTGODecklistCache),
  fork mantido no formato do repositório original arquivado. O commit usado está
  fixado em `config.yaml`; somente `mtgo.com_limited_data` é processado.
- Embeddings: modelo aberto
  [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Imagens: [Scryfall](https://scryfall.com).

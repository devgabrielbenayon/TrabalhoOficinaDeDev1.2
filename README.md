# Recomendador de Cartas de Magic: The Gathering (Filtragem Baseada em Conteúdo)

Trabalho acadêmico de Sistemas de Recomendação. Dado um deck (ou um conjunto de cartas
escolhidas pelo usuário), o sistema recomenda cartas de Magic parecidas com as que já
estão no deck, usando Filtragem Baseada em Conteúdo (TF-IDF + atributos, similaridade
do cosseno) e explica cada recomendação.

> 🚧 Em construção. As seções marcadas com *TODO* serão preenchidas nas próximas fases.

## Requisitos

- Python 3.10+ (testado com 3.12)
- ~3 GB livres em disco (dados + modelo de embeddings)

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Execução

```bash
python -m data.download                 # baixa Scryfall + decks (idempotente)
python -m src.load                      # gera data/processed/*.parquet e relatório de carga
python -m src.features                  # pré-processa o texto e gera a matriz de itens (.npz)
python -m src.split                     # split treino/teste por deck e validação/avaliação
python -m src.demo                      # Top-10 com explicações para um deck (--deck ID, --color)
python -m src.evaluate                  # busca de pesos + métricas + relatório qualitativo
python run_all.py                       # TODO (fase 7): baixa dados, processa, avalia
streamlit run app/streamlit_app.py      # TODO (fase 6): interface de demonstração
pytest                                  # testes automatizados
```

## Estrutura de pastas

```
├── config.yaml          # todos os parâmetros (seed, recorte, pesos, N...)
├── requirements.txt     # dependências com versões fixadas
├── run_all.py           # reproduz todos os resultados com um comando
├── data/
│   ├── download.py      # baixa Scryfall + MTGODecklistCache
│   ├── raw/             # dados brutos (fora do git)
│   └── processed/       # artefatos processados (fora do git)
├── src/                 # pipeline: carga, pré-processamento, features, perfil,
│                        # recomendação, explicação, baselines, split, métricas, avaliação
├── app/                 # interface Streamlit
├── notebooks/           # análise exploratória
├── tests/               # testes (pytest)
├── results/             # tabelas e figuras geradas
└── docs/                # rascunhos para o relatório
```

## Pipeline e protocolo de avaliação

*Descrição completa: TODO (fase 7).* Resumo do protocolo já implementado:

1. **Split** (`src/split.py`): decks com menos de `split.min_cards_per_deck` cartas distintas
   são descartados (a contagem fica em `data/processed/split_report.json`). 10% dos decks
   restantes formam a **validação**, usada só para escolher pesos; o resto é a **avaliação**.
   Em cada deck, 20% das cartas distintas (no mínimo 1) vão para o **teste**, e o resto é **treino**.
2. **Perfil** (`src/profile.py`): média das linhas de treino ponderada por `min(cópias, 4) / 4`,
   normalizada em L2. Não há feedback negativo: uma decklist só diz o que foi jogado, não o
   que foi rejeitado.
3. **Recomendação** (`src/recommend.py`): cosseno entre o perfil e todas as cartas, excluindo o
   treino. Há um filtro opcional por identidade de cor: só passam cartas cuja identidade é
   subconjunto da identidade do deck, e as incolores sempre passam.
4. **Explicação** (`src/explain.py`): as 5 features que mais contribuem para o produto escalar
   (a soma de todas as contribuições é exatamente o score), a carta do deck mais parecida e as
   keywords e tipos em comum.
5. **Avaliação** (`src/evaluate.py`): busca em grade dos pesos dos blocos pelo nDCG@10 na
   validação; depois, Precision, Recall, nDCG, cobertura de catálogo e diversidade intra-lista
   em N = 5 e 10, nos decks de avaliação, para `random`, `popular`, `popular_color`,
   `cbf_tfidf` e `cbf_tfidf_color`.

Saídas em `results/`: `weights_search.csv`, `metrics.csv`, `metrics.md`, `fig_metrics.png`,
`params.json` e `qualitative.md`.

> Esperado: o baseline popular tende a ganhar em precisão, porque decks competitivos copiam
> uns dos outros. A FBC deve ganhar em cobertura e diversidade.

### Checklist anti-vazamento

| Item | Status | Onde é garantido |
|---|---|---|
| O perfil usa só o treino | ✅ | `recommend_for_deck` e `deck_data` usam só linhas `part == "train"` |
| A popularidade usa só o treino | ✅ | `baselines.popularity_counts` conta só o treino dos decks de avaliação (testado) |
| Os pesos dos blocos são escolhidos na validação | ✅ | `evaluate.search_weights` recebe só os decks `validation` |
| O λ do MMR é escolhido na validação | ⏳ | fase 6 |
| O TF-IDF usa só o texto das cartas | ✅ | `features.text_block` é ajustado só sobre o texto do catálogo |
| Os embeddings usam só o texto das cartas | ⏳ | fase 6 |
| As recomendações excluem o treino | ✅ | `tests/test_recommend.py`, `tests/test_evaluate.py` |
| A seed está fixa em tudo | ✅ | `make_rng(cfg, seed_offsets.*)` no split e no baseline aleatório |

## Resultados

*TODO (fase 5): tabela copiada de `results/metrics.md`.*

## Membros e papéis

| Membro | Papel |
|---|---|
| *Nome* | *Papel* |
| *Nome* | *Papel* |

**Vídeo de apresentação:** *link*

## Fontes de dados e créditos

- Dados de cartas: [Scryfall](https://scryfall.com/docs/api/bulk-data) (bulk `oracle_cards`).
- Decklists: [MTGODecklistCache](https://github.com/Badaro/MTGODecklistCache) (torneios do Magic Online).
  O repositório original foi arquivado em 10/06/2025; usamos o fork mantido
  [Jiliac/MTGODecklistCache](https://github.com/Jiliac/MTGODecklistCache), no mesmo formato,
  com o commit fixado em `config.yaml` para reprodutibilidade. Só a pasta
  `mtgo.com_limited_data` é usada; nomes de jogadores são descartados na leitura.
- Magic: The Gathering é propriedade da Wizards of the Coast. Este é um projeto acadêmico
  de fã, sem afiliação com a Wizards of the Coast ou com o Scryfall.

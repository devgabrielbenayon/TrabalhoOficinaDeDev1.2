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

*TODO (fase 7).*

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
- Magic: The Gathering é propriedade da Wizards of the Coast. Este é um projeto acadêmico
  de fã, sem afiliação com a Wizards of the Coast ou com o Scryfall.

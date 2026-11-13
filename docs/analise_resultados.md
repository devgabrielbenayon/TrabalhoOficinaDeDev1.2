# Análise dos resultados

Este rascunho deve ser preenchido após `python run_all.py`, usando os valores
efetivamente gerados em `results/metrics.md`, `results/metrics.csv`,
`results/mmr_tradeoff.csv` e `results/params.json`. Não copie números de outra versão
dos dados nem conclua com base apenas em exemplos sintéticos.

## Comparação dos métodos

- Compare `cbf_tfidf` com `cbf_emb` em Precision, Recall e nDCG@5/@10. Relacione
  eventuais diferenças ao uso de texto lexical versus representação semântica, sem
  afirmar causalidade além do protocolo.
- Compare `cbf_tfidf` e `cbf_tfidf_color`. O filtro reduz o conjunto candidato; discuta
  se isso melhora relevância e qual o custo em cobertura.
- Compare `cbf_tfidf` e `cbf_tfidf_mmr`. Use `mmr_lambda_search.csv` para citar λ
  escolhido na validação e `fig_mmr_tradeoff.png` para discutir o compromisso entre
  nDCG@10 e diversidade na avaliação.
- Use `random`, `popular` e `popular_color` como referências. Discuta diferenças entre
  recomendação personalizada e frequência global, incluindo as limitações do viés de
  torneios.

## Métricas que não devem ser confundidas

Precision mede acertos em posições iniciais e Recall mede a fração do teste recuperada.
nDCG também considera a posição dos acertos. Cobertura é global sobre os itens
recomendados; diversidade é calculada dentro de cada lista. Uma melhora em diversidade
não implica melhora de relevância.

## Rascunho para preencher

- **Dataset e protocolo usados:** _preencher com `params.json` e relatórios de carga/split._
- **Melhor baseline e métricas:** _preencher com `metrics.md`._
- **Resultado de TF-IDF e MiniLM:** _preencher com `metrics.md`._
- **λ selecionado e trade-off observado:** _preencher com os artefatos MMR._
- **Exemplos qualitativos e falhas:** _comentar `qualitative.md` e as limitações._
- **Ameaças à validade:** _considerar período, decks observados, catálogo, relevância
  binária e ausência de feedback negativo._

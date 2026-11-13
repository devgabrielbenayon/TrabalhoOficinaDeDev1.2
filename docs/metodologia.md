# Metodologia

## Dados e unidade de recomendação

O catálogo é formado pelas cartas não básicas usadas nas decklists do formato e da
janela definidos em `config.yaml`. O pipeline associa nomes de cartas ao catálogo e
descarta ocorrências sem correspondência. A unidade da avaliação é um deck: suas
cartas distintas são separadas entre treino e teste, com a mesma lista de catálogo
para todos os métodos.

## Vetores das cartas

Para cada carta, o pipeline constrói cinco blocos:

1. **Texto:** TF-IDF sobre `clean_text`, que remove lembretes, normaliza símbolos de
   mana e aplica stemming; ou embeddings MiniLM calculados a partir de `oracle_text`
   original. Os embeddings substituem o bloco TF-IDF, não os outros atributos.
2. **Tipos:** multi-hot de supertipos, tipos e subtipos de todas as faces.
3. **Cores:** multi-hot de cores e identidade de cor; cartas incolores são explícitas.
4. **Keywords:** multi-hot das palavras-chave de regras.
5. **Numérico:** CMC, poder, resistência, indicador de criatura e indicador de P/T
   variável, com escala configurada.

Os blocos textuais e categóricos são normalizados por linha. Pesos `w_b` combinam os
blocos por concatenação:

`x_i = [w_text · text_i | w_types · types_i | w_colors · colors_i | w_keywords · keywords_i | w_numeric · numeric_i]`

Uma normalização L2 final de cada linha torna o produto escalar equivalente ao
cosseno. A busca em grade escolhe proporções de pesos na validação. O TF-IDF é
ajustado sobre o texto das cartas do catálogo; não usa dados de deck, treino ou teste.
O encoder MiniLM é pré-treinado externamente e aplicado ao texto de metadados,
independente das interações do projeto.

## Perfil e recomendação

Se `T_d` é o conjunto de cartas distintas na parte de treino do deck `d`, o peso de
cópias é `a_i = min(cópias_i, limite) / limite`. O perfil é:

`p_d = normalize_L2(Σ(i ∈ T_d) a_i · x_i)`

O score base de uma carta `j` é `s(d,j) = p_d · x_j`. Cartas de treino são removidas
do ranking. Opcionalmente, também são removidas cartas cuja identidade de cor não é
subconjunto da identidade do deck; cartas incolores sempre passam pelo filtro.

MMR parte das 100 maiores pontuações base e escolhe iterativamente:

`argmax(i ainda não escolhido) [λ · s(d,i) − (1 − λ) · max(j já escolhido) cos(x_i,x_j)]`

`λ = 1` preserva o ranking por relevância; valores menores penalizam itens
redundantes. λ é escolhido maximizando nDCG@10 na validação. A grade avaliada é
`{1.0, 0.9, 0.8, 0.7, 0.5}`.

## Split e métricas

Uma fração de decks é reservada para validação antes da separação de cartas; somente
esses decks escolhem pesos e λ. Nos outros decks, parte das cartas fica em treino e
parte em teste. O perfil usa só treino e as métricas usam só teste. Assim, nenhuma
interação de teste participa da construção do perfil ou do ajuste de parâmetros.

Com relevância binária (carta relevante se está no teste):

- `Precision@N = acertos nos N primeiros / N`
- `Recall@N = acertos nos N primeiros / número de cartas no teste`
- `DCG@N = Σ 1/log2(rank + 1)` para cada acerto; `nDCG@N = DCG / IDCG`
- `Cobertura = itens distintos recomendados / tamanho do catálogo`
- `Diversidade intra-lista = 1 − média dos cossenos de cada par na lista`

As métricas finais são calculadas em N = 5 e N = 10. A figura MMR apresenta nDCG@10
contra diversidade no conjunto de avaliação para a grade de λ; essa comparação é
descritiva. A escolha de λ é feita somente na validação.

## Explicabilidade

Para vetores e perfil normalizados, `p_d · x_j = Σ_k p_d[k] x_j[k]`. As maiores
contribuições positivas são exibidas como features do score, junto à carta de treino
mais similar e aos tipos/keywords compartilhados. Na interface com embeddings, as
dimensões são identificadas por índice; não são tratadas como palavras interpretáveis.

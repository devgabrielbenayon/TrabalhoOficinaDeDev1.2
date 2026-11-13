# Análise dos resultados (rascunho para a equipe revisar)

Números de `results/metrics.md`, `results/mmr_tradeoff.csv`, `results/params.json` e
`results/qualitative.md`, gerados por `python run_all.py` (seed 42).

## Dataset e protocolo

- Decks de Modern do MTGO entre 08/05/2026 e 08/11/2026; 10.460 decks após remover
  10.377 duplicatas exatas; catálogo de 1.772 cartas não-básicas; esparsidade de 98,8%.
- Split: 1.046 decks de validação (só para escolher pesos e λ) e **9.414 decks de
  avaliação**. Em cada deck, 20% das cartas distintas vão para o teste.
- Relevante = carta não-básica que está no teste do deck. Métricas em N = 5 e N = 10.

## Resultados principais (N = 10)

| Método | nDCG@10 | Precision@10 | Cobertura | Diversidade |
|---|---:|---:|---:|---:|
| random | 0,004 | 0,003 | 1,000 | 0,949 |
| popular | 0,115 | 0,062 | 0,011 | 0,702 |
| popular_color | **0,158** | **0,093** | 0,021 | 0,711 |
| cbf_tfidf | 0,099 | 0,051 | 0,494 | 0,588 |
| cbf_tfidf_color | 0,108 | 0,056 | **0,524** | 0,625 |
| cbf_tfidf_mmr (λ = 1,0) | 0,099 | 0,051 | 0,494 | 0,588 |
| cbf_emb | 0,051 | 0,030 | 0,275 | 0,648 |

## Observações

**1. Os baselines populares são fortes em acurácia, como esperado.** O popular com filtro
de cor tem o melhor nDCG@10 (0,158, contra 0,108 da melhor FBC). Decks competitivos
copiam uns dos outros, e a base de mana se repete: **89,5% das recomendações do popular
são terrenos** (fetchlands e shocklands como Flooded Strand, Arid Mesa e Steam Vents),
enquanto terrenos são 36,3% das cartas de teste. Quase todo deck de Modern tem alguma
fetchland no teste, então recomendar fetchlands acerta com frequência. Não é bug: é a
estrutura do formato.

**2. A FBC ganha com folga em cobertura.** `cbf_tfidf_color` recomenda 52,4% do catálogo
pelo menos uma vez, contra 2,1% do `popular_color`, cerca de **25 vezes mais**. O popular
recomenda praticamente as mesmas 20 a 40 cartas para todo mundo; a FBC personaliza pelo
conteúdo do deck. Também recomenda menos terrenos (60,4% das recomendações).

**3. O filtro de cor ajuda os dois tipos de método.** Na FBC, o nDCG@10 sobe de 0,099 para
0,108 (+9%) e a cobertura também sobe (0,494 → 0,524), porque cartas fora das cores do
deck saem da lista e abrem espaço para outras. No caso qualitativo mediano (deck branco e
azul), sem filtro a lista inclui Breeding Pool, Steam Vents e Watery Grave, cujo texto é
quase idêntico ao de Hallowed Fountain, que está no deck. Mas o filtro não é perfeito: o
acerto desse deck, Sacred Foundry, é vermelho, uma cor que só aparece nas cartas de
teste; com o filtro, ele também sairia da lista.

**4. Diversidade: a FBC gera listas mais homogêneas.** A diversidade intra-lista da FBC
(0,59 a 0,63) é menor que a do popular (0,70), porque o ranking por similaridade puxa
cartas parecidas entre si. É o problema de superespecialização (ver `limitacoes.md`). Toda
a coluna é medida no mesmo espaço (matriz TF-IDF selecionada), então os valores são
comparáveis entre métodos.

**5. MMR: a validação escolheu λ = 1,0 (sem reordenação).** Por isso `cbf_tfidf_mmr` é
idêntico a `cbf_tfidf` na tabela. A curva na avaliação (`fig_mmr_tradeoff.png`) mostra o
custo-benefício:

| λ | nDCG@10 | Diversidade |
|---:|---:|---:|
| 1,0 | 0,0989 | 0,588 |
| 0,9 | 0,0993 | 0,674 |
| 0,8 | 0,0929 | 0,766 |
| 0,7 | 0,0906 | 0,841 |
| 0,5 | 0,0746 | 0,888 |

Com λ = 0,9 a diversidade sobe 15% praticamente sem perda de nDCG; com λ = 0,7 sobe 43%,
perdendo 8% de nDCG. Na validação, λ = 1,0 venceu λ = 0,9 por pouco (0,1097 contra 0,1085);
o protocolo exige ficar com a escolha da validação, mas vale discutir que um λ entre 0,8 e
0,9 é uma boa opção de produto quando se quer listas mais variadas.

**6. Embeddings (MiniLM) foram piores que TF-IDF.** nDCG@10 de 0,051 contra 0,099. Hipótese:
o texto de regras de Magic é um jargão muito específico ("deals 3 damage to any target",
"search your library for a basic land"), em que termos exatos importam mais que a
semântica geral que o MiniLM aprendeu em textos comuns. O TF-IDF com bigramas captura
exatamente esses termos. Os embeddings geram listas um pouco mais diversas (0,648), mas
com menos acertos. Os pesos dos dois foram escolhidos de forma independente na validação.

**7. O que o TF-IDF não captura: sinergia.** No caso de erro do qualitativo (deck 0,
"Amulet Titan"), o teste tem Amulet of Vigor e Cultivator Colossus, cartas que funcionam
por **combo** com terrenos que entram virados. O texto delas não se parece com o das
cartas do deck; o sistema recomenda terrenos e rampa parecidos (Demolition Field,
Sakura-Tribe Elder), que fazem sentido isoladamente, mas não acerta as peças do combo.
Similaridade de conteúdo mede "parecido", não "funciona junto".

## Casos qualitativos (`results/qualitative.md`)

- **Maior nDCG (deck 1442, branco e vermelho, nDCG@10 = 0,967):** deck de controle com
  destruição de terrenos (Cleansing Wildfire, Field of Ruin, Flagstones of Trokair). O
  sistema acerta Erode, Ghost Quarter e Price of Freedom nas posições 1, 2 e 4: todas têm
  o mesmo texto de "procurar terreno básico" das cartas de treino.
- **Mediano entre os decks com acerto (deck 4207, branco e azul, nDCG@10 = 0,195):** acerta
  Sacred Foundry em 3º lugar, empatada com outras shocklands de texto quase idêntico (só a
  cor as distingue, e o vermelho de Sacred Foundry não aparece no treino); não encontra
  Teferi, Time Raveler nem Ranger-Captain of Eos.
- **Caso de erro (deck 0, azul, vermelho e verde, nDCG@10 = 0):** ver observação 7.

Em 58,5% dos decks de avaliação a FBC não acerta nenhuma carta no Top-10 (51,8% no
popular). Com cerca de 4 cartas de teste por deck num catálogo de 1.772, a tarefa é difícil, e as
métricas absolutas são baixas para todos os métodos.

## Ameaças à validade

- **Relevância binária e split aleatório:** uma carta só é relevante se estava no deck;
  uma carta boa que o jogador não usou conta como erro.
- **Período e formato únicos:** seis meses de um formato; o meta muda com novas coleções e
  banimentos (duas cartas do catálogo foram banidas depois do período).
- **Decks de torneio:** listas otimizadas e parecidas entre si favorecem o popular.
- **Sem feedback negativo:** decklists não dizem quais cartas foram rejeitadas.

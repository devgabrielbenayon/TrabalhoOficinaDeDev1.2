# Exemplo numérico: do perfil ao ranking

Este exemplo usa três cartas reais e cinco atributos simplificados para explicar o
cálculo. **Os valores abaixo são didáticos**: não são as features nem os scores
produzidos pelo TF-IDF/MiniLM do projeto. Imagine que o perfil unitário
`p = [0.8, 0.4, 0.4, 0, 0.2]` foi construído a partir das cartas de treino de um
deck e que as cinco dimensões representam, nesta ordem: sinal textual de dano,
referência a criatura como alvo, referência a jogador como alvo, cor vermelha e
CMC baixo. Não há vazamento: esse perfil representa apenas cartas de treino.

Para facilitar a conta, os vetores abaixo já têm norma L2 igual a 1:

| Carta | Vetor unitário (5 dimensões) |
|---|---|
| Lightning Bolt | `[0.8, 0.4, 0, 0.4, 0.2]` |
| Shock | `[0.7, 0.5, 0, 0.45, 0.2398]` |
| Lava Spike | `[0.5, 0, 0.8, 0.3, 0.1414]` |

O score base é o cosseno; como os vetores estão normalizados, basta multiplicar
componente a componente e somar:

| Carta | Cálculo `p · x` | Cosseno |
|---|---|---:|
| Lightning Bolt | `0.8×0.8 + 0.4×0.4 + 0.4×0 + 0×0.4 + 0.2×0.2` | `0.8400` |
| Shock | `0.8×0.7 + 0.4×0.5 + 0.4×0 + 0×0.45 + 0.2×0.2398` | `0.8080` |
| Lava Spike | `0.8×0.5 + 0.4×0 + 0.4×0.8 + 0×0.3 + 0.2×0.1414` | `0.7483` |

O ranking é **Lightning Bolt**, **Shock**, **Lava Spike**. Se uma carta estiver no
treino do deck, o sistema a exclui antes de apresentar recomendações; aqui as três
cartas são tratadas como candidatas apenas para demonstrar a ordenação. Com MMR,
depois da primeira escolha, uma candidata muito parecida pode perder posição pela
penalidade de redundância.

No sistema real, os vetores combinam texto e metadados; o score é calculado sobre a
representação usada pelo ranking, e o perfil é uma média ponderada normalizada das
cartas de treino. A demonstração de cinco dimensões não pretende reconstruir os
vetores reais.

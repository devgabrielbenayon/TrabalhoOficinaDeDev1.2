# Exemplo numérico: do perfil ao ranking

Exemplo com **cartas reais e valores reais** da matriz de itens do projeto (pesos dos
blocos escolhidos na validação: texto 1,0, cores 0,25, keywords 0,25, tipos 0, numérico 0).
Para caber num slide, usamos só **5 das 3.526 features**; no fim mostramos o resultado
com o vetor completo, que é o que o sistema realmente usa.

## 1. O deck (somente cartas de treino)

| Carta de treino | Cópias | Peso `min(cópias, 4) / 4` |
|---|---:|---:|
| Lightning Bolt | 4 | 1,0 |
| Rift Bolt | 2 | 0,5 |

Candidatas: **Lightning Helix**, **Skewer the Critics** e **Counterspell**.

## 2. Vetores das cartas (5 features)

Features escolhidas: `texto: "deal 3"`, `texto: "suspend"`, `texto: "gain 3"`,
`texto: "counter target"` e `cor: red`. Os valores vêm da matriz real (TF-IDF com stemming
e multi-hot de cor, já com os pesos dos blocos); cada vetor foi normalizado (norma L2 = 1)
dentro dessas 5 dimensões.

| Carta | deal 3 | suspend | gain 3 | counter target | red |
|---|---:|---:|---:|---:|---:|
| Lightning Bolt | 0,9293 | 0 | 0 | 0 | 0,3694 |
| Rift Bolt | 0,5952 | 0,7504 | 0 | 0 | 0,2876 |
| Lightning Helix | 0,6793 | 0 | 0,6793 | 0 | 0,2776 |
| Skewer the Critics | 0,8514 | 0 | 0 | 0 | 0,5245 |
| Counterspell | 0 | 0 | 0 | 1,0000 | 0 |

## 3. Perfil do deck

Média ponderada das cartas de treino, seguida de normalização L2 (dividir pela soma
dos pesos não muda a direção, então basta somar):

```
soma   = 1,0 × Bolt + 0,5 × Rift Bolt
       = [0,9293 + 0,2976,  0 + 0,3752,  0,  0,  0,3694 + 0,1438]
       = [1,2269, 0,3752, 0, 0, 0,5132]        (norma = 1,3818)
perfil = soma / 1,3818 = [0,8879, 0,2715, 0, 0, 0,3714]
```

## 4. Cosseno com cada candidata

Com vetores unitários, o cosseno é o produto escalar: soma das parcelas
`perfil[k] × carta[k]`. Essas parcelas são exatamente o que a explicação mostra como
"features que mais contribuíram".

| Candidata | Parcelas (deal 3 + … + red) | Cosseno |
|---|---|---:|
| Skewer the Critics | 0,7560 + 0 + 0 + 0 + 0,1948 | **0,9508** |
| Lightning Helix | 0,6032 + 0 + 0 + 0 + 0,1031 | **0,7062** |
| Counterspell | 0 + 0 + 0 + 0 + 0 | **0,0000** |

Ranking com 5 features: **Skewer the Critics → Lightning Helix → Counterspell**.
Explicação de Skewer the Critics: `texto: "deal 3"` responde por 80% do score
(0,7560 / 0,9508) e `cor: red` por 20%.

## 5. O mesmo deck com o vetor completo (o que o sistema faz)

| Candidata | Cosseno com todas as 3.526 features |
|---|---:|
| Lightning Helix | 0,7344 |
| Skewer the Critics | 0,6513 |
| Counterspell | 0,0387 |

A ordem das duas primeiras **se inverte**. As duas cartas dividem com o deck o texto de
dano ("deal 3", "3 damag", "damag target", "cardname deal"), mas Skewer the Critics tem
várias features próprias de peso alto ("spectacl", "spectacl mana_r", "mana_r cardname",
keyword *spectacle*) que o deck não tem; como cada vetor tem norma 1, esse peso "fora do
deck" reduz o cosseno. É por isso que o sistema usa o vetor completo; o recorte de
5 features serve só para ilustrar a conta. Counterspell continua em último: não divide
texto de dano nem cor com o deck.

Os itens do treino (Lightning Bolt e Rift Bolt) recebem score `−∞` antes do ranking e
nunca são recomendados.

*Valores gerados a partir de `data/processed/block_*.npz` e dos pesos de
`results/params.json`.*

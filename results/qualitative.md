# Análise qualitativa (cbf_tfidf, Top-10)

Gerado por `python -m src.evaluate`. ✔ marca acerto (carta do teste do deck).


## Maior nDCG: deck 1442, cores WR (nDCG@10 = 0.967)

**Treino (13 cartas):** 4x Cleansing Wildfire, 4x Field of Ruin, 4x Flagstones of Trokair, 4x High Noon, 4x Path to Exile, 4x Solitude, 4x White Orchid Phantom, 4x Wrath of the Skies, 3x Demolition Field, 3x Sacred Foundry, 2x Sunken Citadel, 1x Flashback, 1x Wear // Tear

**Teste (3 cartas):** Erode, Ghost Quarter, Price of Freedom

**Top-10** (3 acertos):

1. ✔ **Erode** (score 0.734)
   - features: texto: "librari basic" (6%); texto: "may search" (6%); texto: "control may" (5%); identidade de cor: white (5%); texto: "tap shuffl" (5%)
   - mais parecida no deck: Path to Exile (cosseno 0.78)
   - em comum com o deck: palavras-chave nenhuma; tipos instant
2. ✔ **Ghost Quarter** (score 0.696)
   - features: texto: "land" (10%); texto: "librari basic" (6%); texto: "may search" (5%); texto: "control may" (5%); texto: "basic land" (4%)
   - mais parecida no deck: Demolition Field (cosseno 0.85)
   - em comum com o deck: palavras-chave nenhuma; tipos land
3. **Geomancer's Gambit** (score 0.681)
   - features: texto: "land" (7%); texto: "librari basic" (7%); texto: "may search" (6%); texto: "control may" (5%); texto: "basic land" (5%)
   - mais parecida no deck: Cleansing Wildfire (cosseno 0.93)
   - em comum com o deck: palavras-chave nenhuma; tipos sorcery
4. ✔ **Price of Freedom** (score 0.656)
   - features: texto: "land" (7%); texto: "librari basic" (6%); texto: "may search" (6%); texto: "control may" (5%); texto: "tap shuffl" (5%)
   - mais parecida no deck: Cleansing Wildfire (cosseno 0.81)
   - em comum com o deck: palavras-chave nenhuma; tipos sorcery
5. **Avengers Disassembled** (score 0.619)
   - features: texto: "land" (7%); texto: "librari basic" (6%); texto: "may search" (6%); texto: "control may" (5%); texto: "tap shuffl" (5%)
   - mais parecida no deck: Cleansing Wildfire (cosseno 0.76)
   - em comum com o deck: palavras-chave nenhuma; tipos sorcery
6. **Sundering Eruption // Volcanic Fissure** (score 0.592)
   - features: texto: "land" (8%); texto: "tap" (5%); texto: "librari basic" (5%); texto: "may search" (5%); texto: "control may" (4%)
   - mais parecida no deck: Cleansing Wildfire (cosseno 0.65)
   - em comum com o deck: palavras-chave nenhuma; tipos land, sorcery
7. **Assassin's Trophy** (score 0.590)
   - features: texto: "librari basic" (8%); texto: "may search" (7%); texto: "control may" (6%); texto: "basic land" (6%); texto: "basic" (6%)
   - mais parecida no deck: Demolition Field (cosseno 0.69)
   - em comum com o deck: palavras-chave nenhuma; tipos instant
8. **Sakura-Tribe Elder** (score 0.562)
   - features: texto: "librari basic" (9%); texto: "tap shuffl" (7%); texto: "basic land" (7%); texto: "basic" (7%); texto: "card put" (6%)
   - mais parecida no deck: Path to Exile (cosseno 0.73)
   - em comum com o deck: palavras-chave nenhuma; tipos creature
9. **Urza's Cave** (score 0.550)
   - features: texto: "land" (9%); texto: "tap" (9%); texto: "tap shuffl" (7%); texto: "card put" (6%); texto: "put onto" (5%)
   - mais parecida no deck: Demolition Field (cosseno 0.60)
   - em comum com o deck: palavras-chave nenhuma; tipos land, cave
10. **Simulacrum Shaper** (score 0.549)
   - features: texto: "librari basic" (8%); texto: "may search" (7%); texto: "tap shuffl" (6%); texto: "basic land" (6%); texto: "basic" (6%)
   - mais parecida no deck: Path to Exile (cosseno 0.68)
   - em comum com o deck: palavras-chave nenhuma; tipos creature

**Análise:** _(a preencher pela equipe)_


## nDCG mediano entre os decks com acerto: deck 4207, cores WU (nDCG@10 = 0.195)

**Treino (14 cartas):** 4x Guide of Souls, 4x Ocelot Pride, 4x Phelia, Exuberant Shepherd, 4x Quantum Riddler, 4x Solitude, 4x White Orchid Phantom, 4x Witch Enchanter // Witch-Blessed Meadow, 3x Ephemerate, 3x Flagstones of Trokair, 3x Hallowed Fountain, 3x Prismatic Ending, 2x Arid Mesa, 2x Meticulous Archive, 2x The Wondrous Wasp

**Teste (4 cartas):** Flooded Strand, Ranger-Captain of Eos, Sacred Foundry, Teferi, Time Raveler

**Top-10** (1 acertos):

1. **Path to Exile** (score 0.424)
   - features: identidade de cor: white (13%); cor: white (11%); texto: "battlefield" (4%); texto: "tap shuffl" (4%); texto: "may search" (4%)
   - mais parecida no deck: White Orchid Phantom (cosseno 0.58)
   - em comum com o deck: palavras-chave nenhuma; tipos instant
2. **Godless Shrine** (score 0.415)
   - features: texto: "enter" (16%); identidade de cor: white (11%); texto: "life enter" (9%); texto: "may pay" (8%); texto: "land enter" (6%)
   - mais parecida no deck: Hallowed Fountain (cosseno 0.98)
   - em comum com o deck: palavras-chave nenhuma; tipos land, plains
3. ✔ **Sacred Foundry** (score 0.415)
   - features: texto: "enter" (16%); identidade de cor: white (11%); texto: "life enter" (9%); texto: "may pay" (8%); texto: "land enter" (6%)
   - mais parecida no deck: Hallowed Fountain (cosseno 0.98)
   - em comum com o deck: palavras-chave nenhuma; tipos land, plains
4. **Temple Garden** (score 0.415)
   - features: texto: "enter" (16%); identidade de cor: white (11%); texto: "life enter" (9%); texto: "may pay" (8%); texto: "land enter" (6%)
   - mais parecida no deck: Hallowed Fountain (cosseno 0.98)
   - em comum com o deck: palavras-chave nenhuma; tipos land, plains
5. **Erode** (score 0.399)
   - features: identidade de cor: white (14%); cor: white (12%); texto: "battlefield" (4%); texto: "tap shuffl" (4%); texto: "may search" (4%)
   - mais parecida no deck: White Orchid Phantom (cosseno 0.60)
   - em comum com o deck: palavras-chave nenhuma; tipos instant
6. **Razorgrass Ambush // Razorgrass Field** (score 0.396)
   - features: identidade de cor: white (14%); cor: white (12%); texto: "enter" (9%); texto: "tap" (6%); texto: "life enter" (5%)
   - mais parecida no deck: Witch Enchanter // Witch-Blessed Meadow (cosseno 0.55)
   - em comum com o deck: palavras-chave nenhuma; tipos land, instant
7. **Flickering Hound** (score 0.387)
   - features: identidade de cor: white (15%); cor: white (12%); texto: "control" (7%); texto: "battlefield owner" (6%); texto: "one target" (6%)
   - mais parecida no deck: Ephemerate (cosseno 0.58)
   - em comum com o deck: palavras-chave nenhuma; tipos creature, dog
8. **Breeding Pool** (score 0.381)
   - features: texto: "enter" (17%); texto: "life enter" (9%); texto: "may pay" (9%); texto: "land enter" (7%); texto: "life" (7%)
   - mais parecida no deck: Hallowed Fountain (cosseno 0.98)
   - em comum com o deck: palavras-chave nenhuma; tipos land, island
9. **Steam Vents** (score 0.381)
   - features: texto: "enter" (17%); texto: "life enter" (9%); texto: "may pay" (9%); texto: "land enter" (7%); texto: "life" (7%)
   - mais parecida no deck: Hallowed Fountain (cosseno 0.98)
   - em comum com o deck: palavras-chave nenhuma; tipos land, island
10. **Watery Grave** (score 0.381)
   - features: texto: "enter" (17%); texto: "life enter" (9%); texto: "may pay" (9%); texto: "land enter" (7%); texto: "life" (7%)
   - mais parecida no deck: Hallowed Fountain (cosseno 0.98)
   - em comum com o deck: palavras-chave nenhuma; tipos land, island

**Análise:** _(a preencher pela equipe)_


## nDCG = 0 (caso de erro): deck 0, cores URG (nDCG@10 = 0.000)

**Treino (20 cartas):** 4x Arboreal Grazer, 4x Gruul Turf, 4x Primeval Titan, 4x Scapeshift, 4x Spelunking, 4x Urza's Saga, 3x Boseiju, Who Endures, 3x Green Sun's Zenith, 3x Malevolent Rumble, 2x Lotus Field, 2x Simic Growth Chamber, 2x Summoner's Pact, 1x Aftermath Analyst, 1x Echoing Deeps, 1x Hanweir Battlements … (+5)

**Teste (5 cartas):** Amulet of Vigor, Crumbling Vestige, Cultivator Colossus, Otawara, Soaring City, Shifting Woodland

**Top-10** (0 acertos):

1. **Demolition Field** (score 0.551)
   - features: texto: "land" (17%); texto: "onto battlefield" (6%); texto: "land card" (6%); texto: "onto" (6%); cor: colorless (5%)
   - mais parecida no deck: Urza's Cave (cosseno 0.60)
   - em comum com o deck: palavras-chave nenhuma; tipos land
2. **Elvish Reclaimer** (score 0.549)
   - features: texto: "land" (17%); texto: "land card" (9%); identidade de cor: green (8%); texto: "tap" (7%); cor: green (6%)
   - mais parecida no deck: Urza's Cave (cosseno 0.68)
   - em comum com o deck: palavras-chave nenhuma; tipos creature, elf
3. **Ghost Quarter** (score 0.526)
   - features: texto: "land" (18%); texto: "tap" (7%); cor: colorless (5%); texto: "onto battlefield" (5%); texto: "land card" (5%)
   - mais parecida no deck: Urza's Cave (cosseno 0.59)
   - em comum com o deck: palavras-chave nenhuma; tipos land
4. **Sakura-Tribe Scout** (score 0.509)
   - features: texto: "land" (11%); texto: "onto battlefield" (9%); texto: "land card" (9%); texto: "onto" (9%); identidade de cor: green (9%)
   - mais parecida no deck: Arboreal Grazer (cosseno 0.70)
   - em comum com o deck: palavras-chave nenhuma; tipos creature
5. **Selesnya Sanctuary** (score 0.500)
   - features: texto: "land" (22%); texto: "tap" (9%); texto: "land enter" (8%); identidade de cor: green (7%); texto: "enter" (6%)
   - mais parecida no deck: Simic Growth Chamber (cosseno 0.89)
   - em comum com o deck: palavras-chave nenhuma; tipos land
6. **Golgari Rot Farm** (score 0.494)
   - features: texto: "land" (22%); texto: "tap" (9%); texto: "land enter" (8%); identidade de cor: green (7%); texto: "enter" (6%)
   - mais parecida no deck: Gruul Turf (cosseno 0.85)
   - em comum com o deck: palavras-chave nenhuma; tipos land
7. **Sakura-Tribe Elder** (score 0.491)
   - features: identidade de cor: green (9%); texto: "land" (8%); texto: "onto battlefield" (7%); texto: "land card" (7%); texto: "onto" (7%)
   - mais parecida no deck: Scapeshift (cosseno 0.49)
   - em comum com o deck: palavras-chave nenhuma; tipos creature
8. **Lessons from Life** (score 0.487)
   - features: texto: "land" (9%); texto: "onto battlefield" (8%); texto: "land card" (8%); texto: "onto" (8%); texto: "battlefield tap" (7%)
   - mais parecida no deck: Arboreal Grazer (cosseno 0.65)
   - em comum com o deck: palavras-chave nenhuma; tipos sorcery
9. **Growth Spiral** (score 0.479)
   - features: texto: "land" (11%); texto: "onto battlefield" (9%); texto: "land card" (9%); texto: "onto" (9%); texto: "put land" (7%)
   - mais parecida no deck: Arboreal Grazer (cosseno 0.66)
   - em comum com o deck: palavras-chave nenhuma; tipos instant
10. **Simulacrum Shaper** (score 0.478)
   - features: identidade de cor: green (9%); texto: "land" (7%); cor: green (7%); texto: "onto battlefield" (6%); texto: "land card" (6%)
   - mais parecida no deck: Primeval Titan (cosseno 0.47)
   - em comum com o deck: palavras-chave nenhuma; tipos creature, elf

**Análise:** _(a preencher pela equipe)_

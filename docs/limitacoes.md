# Limitações

- **Superespecialização:** a média do deck favorece atributos recorrentes e pode
  recomendar variações muito parecidas, reduzindo a exploração de estratégias novas.
  MMR penaliza redundância entre itens, mas não modela sinergias de jogo.
- **Sem feedback negativo:** uma decklist registra cartas escolhidas, não cartas
  rejeitadas. O sistema não aprende preferências por exclusão.
- **Sinergias não capturadas:** as recomendações comparam itens individualmente com o
  perfil. Combos, curvas, papéis no deck e interações entre várias cartas não são
  representados diretamente.
- **Viés do meta:** as decklists vêm de torneios, formato e janela temporal
  configurados. Resultados refletem esse recorte e podem desfavorecer cartas casuais,
  recentes ou pouco usadas.
- **Deck vazio (cold start):** sem cartas de treino não há perfil que represente
  preferências; a interface pede ao usuário que selecione ao menos uma carta.
- **Metadados:** qualidade e atualização do texto, tipos, cores e keywords afetam as
  features e as explicações. Erros de associação entre nomes de decklist e catálogo
  também removem interações.
- **Relevância binária e split:** uma carta do teste vale como acerto, sem distinguir
  quantidade ou função. Decks pequenos e o recorte temporal limitam a generalização.
- **Embeddings gerais:** MiniLM é um modelo de sentença generalista, não treinado
  especificamente para regras de Magic. Suas dimensões latentes não são interpretáveis
  como palavras; para explicações semânticas, o sistema mostra seus índices e os
  metadados compartilhados separadamente.
- **Custo e latência inicial:** download do catálogo/modelo e geração inicial de
  embeddings consomem tempo e armazenamento. O cache acelera repetições; o tempo
  observado depende de hardware, rede e tamanho do catálogo.

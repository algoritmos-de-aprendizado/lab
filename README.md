## Pipeline de Análise de Fidelidade Semântica (Orlando Furioso) - MLP Autoassociativa

### RA: 1353225

Este pipeline foi desenvolvido para **quantificar e visualizar a fidelidade semântica** de múltiplas traduções do "Orlando Furioso" (italiano para inglês) em relação ao texto original. A metodologia integra Processamento de Linguagem Natural (PLN) com uma rede neural MLP autoassociativa (autoencoder) para extrair representações de texto significativas e compará-las.

### Ferramentas Principais

* **Geração de Embeddings**: `SentenceTransformers` (modelos multilíngues) para criar vetores semânticos das estrofes.
* **Rede Neural**: `PyTorch` para a implementação e treinamento da MLP Autoassociativa.
* **Análise e Visualização**: `pandas`, `matplotlib`, `seaborn` para estatísticas e gráficos comparativos.

### A Rede Neural MLP Autoassociativa

A rede neural utilizada é uma **MLP Autoassociativa**, também conhecida como um tipo de **Autoencoder**.

**Motivação para a Escolha:**

A escolha da MLP Autoassociativa ocorre por alguns motivos:

1.  **Aprendizado de Representações Semânticas Refinadas:** Embora sejam utilizados embeddings multilíngues que capturam a semântica, a MLP autoassociativa pode refinar ainda mais essas representações. Ela aprende a mapear a entrada (os embeddings da estrofe) para uma versão mais densa e potencialmente mais discriminativa, que pode ser mais eficaz para a medição de similaridade.
2.  **Redução de Dimensionalidade e Extração de Características:** A obra "Orlando Furioso" é extensa e, quando convertida em representações numéricas (como embeddings), resulta em vetores de alta dimensão. A MLP autoassociativa atua como um **redutor de dimensionalidade**. Ao tentar reconstruir sua própria entrada através de uma camada oculta de menor dimensão, ela é forçada a aprender as **características mais essenciais e compactas** dos dados. Essa representação comprimida é o que chamamos de **espaço latente** ou **representação codificada**. Essa compressão ajuda a focar nas informações mais relevantes e a mitigar o "ruído" que poderia estar presente nas representações originais de maior dimensão.
3.  **Flexibilidade:** A arquitetura de uma MLP é flexível e pode ser ajustada (número de camadas, neurônios) para testar e otimizar a codificação de informações.

### Como o Pipeline Funciona

O fluxo de dados começa após a **ingestão e pré-processamento** dos textos de "Orlando Furioso", transformados em um formato limpo e padronizado por **estrofe**. Esses textos são então convertidos em **embeddings multilíngues**, que são vetores numéricos que capturam o significado semântico e são compatíveis entre diferentes idiomas. Em seguida, cada conjunto de embeddings (original e traduções) é alimentado em uma **MLP autoassociativa** separada. Essa rede neural comprime os embeddings, aprendendo representações essenciais otimizadas para cada estrofe. A partir dessas representações, a **fidelidade semântica** é calculada via similaridade de cosseno. Finalmente, os scores de fidelidade são **visualizados e analisados** em gráficos, oferecendo insights sobre a qualidade das traduções.

### Critério de Avaliação de Fidelidade

A fidelidade da tradução é avaliada através do critério de **Fidelidade Semântica**:

A similaridade é medida pela **similaridade de cosseno** entre as **representações essenciais** (saídas da MLP Autoassociativa) das estrofes correspondentes (original vs. tradução). O que permite isso são os embeddings multilíngues, que já alinham semanticamente textos de diferentes idiomas em um espaço compartilhado, tornando a similaridade de cosseno uma métrica direta de quão bem o significado foi preservado.

### Fluxo do Pipeline

O processo ocorre linearmente, garantindo a coerência da análise:

1.  **Preparação de Dados**: Textos são carregados e pré-processados (conversão para minúsculas) para padronização. **A segmentação ocorre em nível de estrofe, mantendo o registro do canto ao qual cada estrofe pertence.**
2.  **Vetorização Semântica**: **Estrofes** são convertidas em embeddings multilíngues (vetores numéricos que capturam significado entre idiomas).
3.  **Refinamento com MLP Autoassociativa**: Os embeddings das estrofes são alimentados em MLPs autoassociativas (uma para cada idioma/tradução) para aprender representações mais compactas e refinadas.
4.  **Avaliação Semântica**: A similaridade de cosseno é calculada entre as representações obtidas da estrofe original e das traduções, gerando scores de fidelidade semântica para **cada estrofe**.
5.  **Análise e Visualização**: Os scores de fidelidade são representados em gráficos para comparar as traduções.
6.  **Conclusões e Sugestões Futuras**: Os resultados e limitações são discutidos e propostas para análises mais aprofundadas são apresentadas.
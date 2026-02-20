# VecLink 🔗

[🇺🇸 English](#-english-version) \| [🇧🇷 Português](#-versão-em-português)

------------------------------------------------------------------------

## 🇺🇸 English Version

A Python toolkit designed to perform robust **record linkage** between
datasets that lack a common primary key.\
VecLink uses **Deep Learning (TensorFlow/Keras)** to generate semantic
text embeddings and **K-Nearest Neighbors (KNN)** to find the closest
matches between distinct databases.

------------------------------------------------------------------------

### 🧠 The Concept: How It Works

Traditional record linkage often relies on exact string matching or
simple fuzzy logic (like Levenshtein distance), which can fail when
dealing with abbreviations, swapped words, or complex string structures.

VecLink solves this using a machine learning pipeline:

1.  **Tokenization**\
    Breaks down text (names, addresses, mixed data) into numerical
    sequences.

2.  **Deep Learning Embeddings**\
    A Sequential Neural Network (using Keras `Embedding` and
    `GlobalAveragePooling1D`) converts these sequences into dense vector
    representations in a multi-dimensional space.

3.  **K-Nearest Neighbors (KNN)**\
    Uses the `scikit-learn` KNN algorithm to calculate the cosine
    distance between vectors from the Reference dataset and the Target
    dataset.

------------------------------------------------------------------------

### 📊 Understanding the Results (Distance Metric)

When running `VecLink.cruzar()`, the output DataFrame includes a crucial
column:

    distancia_similaridade

Because the algorithm uses **Cosine Distance**:

-   **0.0 (or very close to 0)** → Perfect or near-perfect semantic
    match.
-   **Higher values (0.3, 0.5, etc.)** → Lower similarity.

> 💡 Tip: Define a similarity threshold (e.g., distance \< 0.15) for
> high-confidence linkages.

------------------------------------------------------------------------

### 🚀 Quick Start

#### Install Dependencies

``` bash
pip install -r requirements.txt
```

#### Usage Example

``` python
import pandas as pd
from veclink import VecLink

df_reference = pd.read_csv("reference_data.csv", sep=";")
df_target = pd.read_csv("target_data.csv", sep=";")

linker = VecLink(dim_embedding=64)

linker.treinar(
    df_ref=df_reference, col_ref="FULL_ADDRESS",
    df_alvo=df_target, col_alvo="LOCATION_DESC"
)

df_matched = linker.cruzar(
    df_ref=df_reference,
    df_alvo=df_target,
    col_alvo="LOCATION_DESC"
)

df_matched.to_csv("matched_output.csv", sep=";", index=False)
```

------------------------------------------------------------------------

# 🇧🇷 Versão em Português

Uma ferramenta em Python desenvolvida para realizar **Record Linkage**
entre bases de dados que não possuem uma chave primária em comum.

O VecLink utiliza **Deep Learning (TensorFlow/Keras)** para gerar
embeddings semânticos de texto e **K-Nearest Neighbors (KNN)** para
encontrar as correspondências mais próximas entre registros.

------------------------------------------------------------------------

### 🧠 O Conceito: Como Funciona

O VecLink resolve o cruzamento de dados usando um pipeline de Machine
Learning:

1.  **Tokenização**\
    Transforma o texto em sequências numéricas.

2.  **Embeddings com Deep Learning**\
    Utiliza `Embedding` e `GlobalAveragePooling1D` do Keras para gerar
    vetores densos.

3.  **KNN com Distância de Cosseno**\
    Calcula a proximidade semântica entre registros.

------------------------------------------------------------------------

### 📊 Métrica de Distância

A coluna `distancia_similaridade` representa:

-   **0.0 (ou próximo de 0)** → Match quase perfeito.
-   **Valores maiores** → Menor similaridade.

> 💡 Dica: Utilize um threshold como 0.15 para garantir maior precisão.

------------------------------------------------------------------------

### 🚀 Exemplo de Uso

``` python
import pandas as pd
from veclink import VecLink

df_referencia = pd.read_csv("base_referencia.csv", sep=";")
df_alvo = pd.read_csv("base_alvo.csv", sep=";")

linker = VecLink(dim_embedding=64)

linker.treinar(
    df_ref=df_referencia, col_ref="ENDERECO_COMPLETO",
    df_alvo=df_alvo, col_alvo="DESC_LOCAL"
)

df_resultado = linker.cruzar(
    df_ref=df_referencia,
    df_alvo=df_alvo,
    col_alvo="DESC_LOCAL"
)

df_resultado.to_csv("resultado_cruzamento.csv", sep=";", index=False)
```

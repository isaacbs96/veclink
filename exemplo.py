import pandas as pd
from veclink import VecLink

# 1. Carrega as bases
df_referencia = pd.read_csv('base_referencia.csv', sep=';')
df_nova = pd.read_csv('base_nova.csv', sep=';')

# 2. Instancia a classe
relacionador = VecLink(dim_embedding=64)

# 3. Treina o modelo com ambas as bases (para cobrir todo o vocabulário)
relacionador.treinar(
    df_ref=df_referencia, col_ref='COLUNA_NOME_REF',
    df_alvo=df_nova, col_alvo='COLUNA_NOME_NOVA'
)

# 4. Executa o cruzamento
df_resultado = relacionador.cruzar(
    df_ref=df_referencia,
    df_alvo=df_nova,
    col_alvo='COLUNA_NOME_NOVA'
)

# 5. Exporta o resultado
df_resultado.to_csv('resultado_cruzamento.csv', sep=';', index=False)
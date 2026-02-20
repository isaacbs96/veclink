import pandas as pd
import re
import logging
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Embedding # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from sklearn.neighbors import NearestNeighbors

# Configuração básica de log para exibir as mensagens no console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class VecLink:
    def __init__(self, dim_embedding=64, metrica='cosine'):
        self.dim_embedding = dim_embedding
        self.metrica = metrica
        self.tokenizer = Tokenizer()
        self.tam_max = 0
        self.modelo_embedding = None
        self.modelo_knn = NearestNeighbors(n_neighbors=1, metric=self.metrica)
        self.treinado = False
        self.logger = logging.getLogger("VecLink")

    def _limpar_texto(self, texto):
        texto = str(texto)
        texto = re.sub(r'\d', '', texto)
        return texto.lower().strip()

    def _construir_modelo(self):
        self.logger.info("Construindo a arquitetura do modelo de Embedding...")
        self.modelo_embedding = Sequential([
            Input(shape=(self.tam_max,)),
            Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=self.dim_embedding),
            GlobalAveragePooling1D()
        ])

    def treinar(self, df_ref, col_ref, df_alvo=None, col_alvo=None):
        self.logger.info("Iniciando o treinamento do VecLink...")
        
        self.logger.info("Limpando e padronizando os textos de referência...")
        textos_ref = df_ref[col_ref].apply(self._limpar_texto).tolist()
        textos_alvo = df_alvo[col_alvo].apply(self._limpar_texto).tolist() if df_alvo is not None and col_alvo is not None else []

        self.logger.info("Ajustando o Tokenizador com o vocabulário das bases...")
        todos_textos = textos_ref + textos_alvo
        self.tokenizer.fit_on_texts(todos_textos)

        self.logger.info("Convertendo textos para sequências numéricas...")
        seqs_ref = self.tokenizer.texts_to_sequences(textos_ref)
        seqs_alvo = self.tokenizer.texts_to_sequences(textos_alvo) if textos_alvo else []

        self.tam_max = max(
            max([len(x) for x in seqs_ref]) if seqs_ref else 0,
            max([len(x) for x in seqs_alvo]) if seqs_alvo else 0
        )

        self.logger.info(f"Tamanho máximo da sequência definido como: {self.tam_max}")
        x_ref = pad_sequences(seqs_ref, maxlen=self.tam_max, padding='post')
        
        self._construir_modelo()
        
        self.logger.info("Gerando embeddings (vetores) para a base de referência. Isso pode levar alguns instantes...")
        embeddings_ref = self.modelo_embedding.predict(x_ref, verbose=0)

        self.logger.info("Treinando o modelo K-Nearest Neighbors (KNN)...")
        self.modelo_knn.fit(embeddings_ref)
        self.treinado = True
        
        self.logger.info("Treinamento concluído com sucesso!")

    def cruzar(self, df_ref, df_alvo, col_alvo, incluir_distancias=True):
        if not self.treinado:
            self.logger.error("Tentativa de cruzar bases sem treinar o modelo antes.")
            raise ValueError("O modelo precisa ser treinado com treinar() primeiro.")

        self.logger.info("Iniciando o cruzamento das bases...")
        
        self.logger.info("Preparando a base alvo (limpeza e tokenização)...")
        textos_alvo = df_alvo[col_alvo].apply(self._limpar_texto).tolist()
        seqs_alvo = self.tokenizer.texts_to_sequences(textos_alvo)
        x_alvo = pad_sequences(seqs_alvo, maxlen=self.tam_max, padding='post')

        self.logger.info("Gerando embeddings para a base alvo...")
        embeddings_alvo = self.modelo_embedding.predict(x_alvo, verbose=0)
        
        self.logger.info("Buscando as correspondências mais próximas via KNN...")
        distancias, indices = self.modelo_knn.kneighbors(embeddings_alvo)

        self.logger.info("Montando o DataFrame com os resultados finais...")
        df_res = df_alvo.copy()
        df_res['idx_match_ref'] = indices.flatten()

        if incluir_distancias:
            df_res['distancia_similaridade'] = distancias.flatten()

        df_ref_reset = df_ref.reset_index(drop=False).rename(columns={'index': 'idx_match_ref'})
        df_final = pd.merge(df_res, df_ref_reset, on='idx_match_ref', how='left', suffixes=('_ALVO', '_REF'))
        df_final.drop(columns=['idx_match_ref'], inplace=True)

        self.logger.info("Cruzamento finalizado!")
        return df_final
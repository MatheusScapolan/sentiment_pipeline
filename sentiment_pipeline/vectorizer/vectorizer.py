"""
Vetorização configurável de textos. Transforma strings processadas em
matrizes numéricas prontas para classificadores.

Estratégias suportadas:
    bow       : Bag of Words (CountVectorizer)
    tfidf     : TF-IDF (TfidfVectorizer)
    tfidf_svd : TF-IDF + redução dimensional via TruncatedSVD (LSA)
    word2vec  : média dos vetores GloVe pré-treinados ou treinados no corpus
"""

import logging
import os

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)


class Vectorizer:
    """
    Vetoriza textos com a estratégia definida na configuração.

    Parâmetros comuns:
        strategy (str)     : "bow", "tfidf", "tfidf_svd" ou "word2vec".
        max_features (int) : tamanho máximo do vocabulário.

    Parâmetros por estratégia:
        bow        -> binary (bool)
        tfidf      -> sublinear_tf (bool), ngram_range (tuple), norm (str)
        tfidf_svd  -> mesmos do tfidf + svd_components (int: 50, 100 ou 300)
        word2vec   -> glove_path (str|None), embedding_dim (int)

    Exemplo:
        vec = Vectorizer({"strategy": "tfidf", "ngram_range": (1, 2), "max_features": 50000})
        X_treino = vec.fit_transform(textos_treino)
        X_teste  = vec.transform(textos_teste)
    """

    def __init__(self, config: dict):
        self.config = config
        self.strategy = config.get("strategy", "tfidf")
        self._vec = None
        self._svd = None
        self._w2v = None
        self._glove = None
        self._fitted = False

    # --- Construtores internos ---

    def _bow(self):
        return CountVectorizer(
            binary=self.config.get("binary", False),
            max_features=self.config.get("max_features", 50000),
            strip_accents="unicode",
        )

    def _tfidf(self):
        return TfidfVectorizer(
            sublinear_tf=self.config.get("sublinear_tf", True),
            ngram_range=self.config.get("ngram_range", (1, 2)),
            max_features=self.config.get("max_features", 50000),
            norm=self.config.get("norm", "l2"),
            strip_accents="unicode",
            min_df=self.config.get("min_df", 2),
        )

    # --- Word2Vec / GloVe ---

    def _carregar_glove(self, caminho: str) -> dict:
        logger.info("Carregando GloVe de %s...", caminho)
        vetores = {}
        with open(caminho, encoding="utf-8") as f:
            for linha in f:
                partes = linha.rstrip().split(" ")
                try:
                    vetores[partes[0]] = np.array(partes[1:], dtype=np.float32)
                except ValueError:
                    continue
        logger.info("GloVe carregado: %d vetores.", len(vetores))
        return vetores

    def _media_vetores(self, textos: list, dicionario: dict, dim: int) -> np.ndarray:
        resultado = []
        for texto in textos:
            vecs = [dicionario[t] for t in texto.split() if t in dicionario]
            resultado.append(np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32))
        return np.array(resultado, dtype=np.float32)

    # --- Interface pública ---

    def fit_transform(self, textos: list) -> np.ndarray:
        """Ajusta o vetorizador e retorna a matriz de features do conjunto de treino."""
        if self.strategy == "bow":
            self._vec = self._bow()
            result = self._vec.fit_transform(textos)

        elif self.strategy == "tfidf":
            self._vec = self._tfidf()
            result = self._vec.fit_transform(textos)

        elif self.strategy == "tfidf_svd":
            self._vec = self._tfidf()
            n = self.config.get("svd_components", 100)
            self._svd = TruncatedSVD(n_components=n, random_state=42)
            result = self._svd.fit_transform(self._vec.fit_transform(textos))

        elif self.strategy == "word2vec":
            glove_path = self.config.get("glove_path")
            dim = self.config.get("embedding_dim", 100)
            if glove_path and os.path.isfile(glove_path):
                self._glove = self._carregar_glove(glove_path)
                result = self._media_vetores(textos, self._glove, dim)
            else:
                try:
                    from gensim.models import Word2Vec
                except ImportError:
                    raise ImportError("Execute: pip install gensim")
                tokenizado = [t.split() for t in textos]
                logger.info("Treinando Word2Vec no corpus (dim=%d)...", dim)
                self._w2v = Word2Vec(tokenizado, vector_size=dim, window=5, min_count=2, seed=42)
                result = self._media_vetores(textos, self._w2v.wv, dim)
        else:
            raise ValueError(f"Estratégia desconhecida: '{self.strategy}'")

        self._fitted = True
        return result

    def transform(self, textos: list) -> np.ndarray:
        """Transforma novos textos usando o vetorizador já ajustado."""
        if not self._fitted:
            raise RuntimeError("Execute fit_transform antes de transform.")

        if self.strategy == "bow":
            return self._vec.transform(textos)
        if self.strategy == "tfidf":
            return self._vec.transform(textos)
        if self.strategy == "tfidf_svd":
            return self._svd.transform(self._vec.transform(textos))
        if self.strategy == "word2vec":
            dim = self.config.get("embedding_dim", 100)
            dicionario = self._glove if self._glove is not None else self._w2v.wv
            return self._media_vetores(textos, dicionario, dim)

        raise ValueError(f"Estratégia desconhecida: '{self.strategy}'")

    def get_feature_names(self) -> list | None:
        if self._vec and hasattr(self._vec, "get_feature_names_out"):
            return list(self._vec.get_feature_names_out())
        return None

    def get_config(self) -> dict:
        return self.config.copy()

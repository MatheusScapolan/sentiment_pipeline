"""
Orquestrador central do pipeline de análise de sentimentos.
Integra Preprocessor, Vectorizer e Classifier em um único objeto.

O pipeline é agnóstico de dataset e idioma: funciona com qualquer
coleção de textos e rótulos binários, em qualquer língua suportada
pelo NLTK.

Uso mínimo
    from sentiment_pipeline import SentimentPipeline

    pipeline = SentimentPipeline({
        "preprocessor": {
            "language": "portuguese",
            "lowercase": True,
            "remove_stopwords": "keep_negations",
            "normalization": "stemming",
            "handle_negations": True,
        },
        "vectorizer": {
            "strategy": "tfidf",
            "sublinear_tf": True,
            "ngram_range": (1, 2),
            "max_features": 50000,
        },
        "classifier": {
            "model": "logistic_regression",
            "search": "random",
            "cv": 3,
        },
    })
    pipeline.fit(X_treino, y_treino)
    metricas = pipeline.evaluate(X_val, y_val)
    print(pipeline.predict_texto("Não gostei nada desse filme!"))
"""

import logging
import time

import numpy as np

from sentiment_pipeline.preprocessor.preprocessor import Preprocessor
from sentiment_pipeline.vectorizer.vectorizer import Vectorizer
from sentiment_pipeline.classifier.classifier import Classifier
from sentiment_pipeline.utils.utils import avaliar

logger = logging.getLogger(__name__)


class SentimentPipeline:
    """
    Pipeline completo: pré-processamento + vetorização + classificação.

    Parâmetros
        config (dict)           chaves obrigatórias "preprocessor", "vectorizer"
                                e "classifier".
        nome (str)              identificador para logs e relatórios.
        nomes_classes (list)    nomes das classes na ordem [classe_0, classe_1].
                                Padrão ["negativo", "positivo"]. Usado nos relatórios.

    Após fit(), os atributos tempo_treino e metricas_treino ficam disponíveis.
    """

    def __init__(self, config: dict, nome: str = "pipeline", nomes_classes: list = None):
        self.config = config
        self.nome = nome
        self.nomes_classes = nomes_classes or ["negativo", "positivo"]
        self.tempo_treino = None
        self.metricas_treino = None
        self._fitted = False

        # Repassa a estratégia de vetorização ao classifier (necessário para NB)
        clf_cfg = config.get("classifier", {}).copy()
        clf_cfg["vectorizer_strategy"] = config.get("vectorizer", {}).get("strategy", "tfidf")
        clf_cfg["binary"] = config.get("vectorizer", {}).get("binary", False)

        self._pre = Preprocessor(config.get("preprocessor", {}))
        self._vec = Vectorizer(config.get("vectorizer", {}))
        self._clf = Classifier(clf_cfg)

    def fit(self, textos: list, rotulos) -> "SentimentPipeline":
        """
        Treina o pipeline completo: pré-processa, vetoriza e classifica.

        Parâmetros
            textos   lista de strings brutas (qualquer idioma).
            rotulos  array-like de inteiros 0 e 1.
        """
        logger.info("[%s] Iniciando treinamento...", self.nome)
        t0 = time.time()

        logger.info("[%s] 1/3 Pré-processamento (idioma=%s)",
                    self.nome, self._pre.idioma)
        textos_proc = self._pre.transform(textos)

        logger.info("[%s] 2/3 Vetorização (%s)",
                    self.nome, self.config.get("vectorizer", {}).get("strategy"))
        X = self._vec.fit_transform(textos_proc)

        logger.info("[%s] 3/3 Classificação (%s | busca=%s)",
                    self.nome,
                    self.config.get("classifier", {}).get("model"),
                    self.config.get("classifier", {}).get("search", "manual"))
        self._clf.fit(X, rotulos)

        self.tempo_treino = round(time.time() - t0, 2)
        self.metricas_treino = avaliar(
            rotulos, self._clf.predict(X),
            nomes_classes=self.nomes_classes,
        )
        self._fitted = True

        logger.info(
            "[%s] Concluído em %.1fs | F1-macro (treino): %.4f",
            self.nome, self.tempo_treino, self.metricas_treino["f1_macro"],
        )
        return self

    def predict(self, textos: list) -> np.ndarray:
        """Classifica novos textos brutos e retorna array de classes."""
        self._checar()
        return self._clf.predict(
            self._vec.transform(self._pre.transform(textos))
        )

    def predict_proba(self, textos: list):
        """Retorna probabilidades por classe quando o classificador suporta."""
        self._checar()
        return self._clf.predict_proba(
            self._vec.transform(self._pre.transform(textos))
        )

    def evaluate(self, textos: list, rotulos) -> dict:
        """
        Avalia o pipeline em um conjunto externo.
        Use validação para seleção de configuração e teste apenas na avaliação final.
        """
        self._checar()
        y_pred = self.predict(textos)
        y_proba = self.predict_proba(textos)
        proba_pos = (
            y_proba[:, 1] if y_proba is not None and y_proba.ndim == 2 else y_proba
        )
        return avaliar(rotulos, y_pred, proba_pos, nomes_classes=self.nomes_classes)

    def predict_texto(self, texto: str) -> dict:
        """
        Classifica um único texto e retorna resultado legível.

        Retorna
            {"classe": int, "rotulo": str, "confianca": float|None}
        """
        self._checar()
        classe = int(self.predict([texto])[0])
        proba = self.predict_proba([texto])
        confianca = (
            round(float(proba[0, classe]), 4)
            if proba is not None and proba.ndim == 2
            else None
        )
        nome_classe = (
            self.nomes_classes[classe]
            if classe < len(self.nomes_classes)
            else str(classe)
        )
        return {"classe": classe, "rotulo": nome_classe, "confianca": confianca}

    def resumo(self) -> str:
        """Retorna string legível com a configuração e métricas do pipeline."""
        pre = self.config.get("preprocessor", {})
        vec = self.config.get("vectorizer", {})
        clf = self.config.get("classifier", {})
        linhas = [
            f"Pipeline : {self.nome}",
            f"  Idioma : {pre.get('language', 'english')}",
            f"  Pre    : lower={pre.get('lowercase')} sw={pre.get('remove_stopwords')} "
            f"norm={pre.get('normalization')} neg={pre.get('handle_negations')}",
            f"  Vec    : {vec.get('strategy')} max_feat={vec.get('max_features')} "
            f"ngram={vec.get('ngram_range')}",
            f"  Clf    : {clf.get('model')} busca={clf.get('search')}",
        ]
        if self._fitted:
            linhas += [
                f"  Params : {self._clf.get_best_params()}",
                f"  Tempo  : {self.tempo_treino}s",
                f"  F1-treino : {self.metricas_treino['f1_macro']}",
            ]
        return "\n".join(linhas)

    def get_config(self) -> dict:
        return self.config.copy()

    def _checar(self):
        if not self._fitted:
            raise RuntimeError("Execute pipeline.fit() antes de predict() ou evaluate().")

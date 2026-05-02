"""
Classificação configurável com suporte a busca de hiperparâmetros.

Modelos suportados
    naive_bayes         MultinomialNB, BernoulliNB ou GaussianNB
    logistic_regression Regressão Logística
    linear_svc          LinearSVC (SVM linear, eficiente para texto)
    random_forest       Random Forest com votação por maioria
    lightgbm            LightGBM, gradient boosting eficiente

Modalidades de busca
    manual  hiperparâmetros fixos em "params"
    grid    GridSearchCV para espaços pequenos
    random  RandomizedSearchCV para espaços maiores
    optuna  otimização bayesiana via Optuna (requer pip install optuna)
"""

import logging
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

logger = logging.getLogger(__name__)

_GRIDS = {
    "naive_bayes":         {"alpha": [0.01, 0.1, 0.5, 1.0, 2.0]},
    "logistic_regression": {"C": [0.01, 0.1, 1.0, 5.0, 10.0], "max_iter": [1000]},
    "linear_svc":          {"estimator__C": [0.01, 0.1, 1.0, 5.0, 10.0]},
    "random_forest":       {"n_estimators": [100, 200, 300], "max_depth": [None, 20, 40]},
    "lightgbm":            {"n_estimators": [100, 200, 300],
                            "learning_rate": [0.05, 0.1, 0.2],
                            "num_leaves": [31, 63, 127]},
}

_DISTRIBUTIONS = {
    "naive_bayes":         {"alpha": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
    "logistic_regression": {"C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0],
                            "max_iter": [1000],
                            "solver": ["lbfgs", "saga"]},
    "linear_svc":          {"estimator__C": [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
    "random_forest":       {"n_estimators": [100, 150, 200, 300, 400],
                            "max_depth": [None, 10, 20, 30, 40],
                            "min_samples_leaf": [1, 2, 4],
                            "max_features": ["sqrt", "log2"]},
    "lightgbm":            {"n_estimators": [100, 200, 300, 400, 500],
                            "learning_rate": [0.01, 0.05, 0.1, 0.15, 0.2],
                            "num_leaves": [15, 31, 63, 127],
                            "min_child_samples": [10, 20, 50],
                            "subsample": [0.7, 0.8, 0.9, 1.0],
                            "colsample_bytree": [0.7, 0.8, 0.9, 1.0]},
}


class Classifier:
    """
    Encapsula um classificador clássico com busca opcional de hiperparâmetros.

    Parâmetros na configuração
        model (str)               nome do modelo.
        search (str)              "manual", "grid", "random" ou "optuna".
        params (dict)             hiperparâmetros fixos quando search for "manual".
        vectorizer_strategy (str) informa o pipeline sobre qual variante do NB usar.
        cv (int)                  folds de validação cruzada, padrão 3.
        n_iter (int)              iterações para "random", padrão 20.
        n_trials (int)            trials para "optuna", padrão 30.

    Exemplo
        clf = Classifier({"model": "logistic_regression", "search": "random",
                          "cv": 3, "n_iter": 20})
        clf.fit(X_treino, y_treino)
        previsoes = clf.predict(X_val)
    """

    def __init__(self, config: dict):
        self.config = config
        self.model_name = config.get("model", "logistic_regression")
        self.search = config.get("search", "manual")
        self.cv = config.get("cv", 3)
        self.n_iter = config.get("n_iter", 20)
        self.n_trials = config.get("n_trials", 30)
        self.vec_strategy = config.get("vectorizer_strategy", "tfidf")
        self._estimator = None
        self._best_params = None

    def _denso(self) -> bool:
        return self.vec_strategy in ("tfidf_svd", "word2vec")

    def _base(self, p: dict = None):
        """Instancia o modelo base com os parâmetros fornecidos."""
        p = p or {}

        if self.model_name == "naive_bayes":
            if self._denso():
                from sklearn.naive_bayes import GaussianNB
                return GaussianNB()
            if self.vec_strategy == "bow" and self.config.get("binary"):
                return BernoulliNB(alpha=p.get("alpha", 1.0))
            return MultinomialNB(alpha=p.get("alpha", 1.0))

        if self.model_name == "logistic_regression":
            return LogisticRegression(
                C=p.get("C", 1.0),
                max_iter=p.get("max_iter", 1000),
                solver=p.get("solver", "lbfgs"),
                random_state=42,
                n_jobs=-1,
            )

        if self.model_name == "linear_svc":
            base = LinearSVC(C=p.get("C", 1.0), max_iter=2000, random_state=42)
            return CalibratedClassifierCV(base, cv=3)

        if self.model_name == "random_forest":
            return RandomForestClassifier(
                n_estimators=p.get("n_estimators", 200),
                max_depth=p.get("max_depth"),
                min_samples_leaf=p.get("min_samples_leaf", 1),
                max_features=p.get("max_features", "sqrt"),
                random_state=42,
                n_jobs=-1,
            )

        if self.model_name == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except ImportError:
                raise ImportError("Execute pip install lightgbm")
            return LGBMClassifier(
                n_estimators=p.get("n_estimators", 200),
                learning_rate=p.get("learning_rate", 0.1),
                num_leaves=p.get("num_leaves", 63),
                min_child_samples=p.get("min_child_samples", 20),
                subsample=p.get("subsample", 0.8),
                colsample_bytree=p.get("colsample_bytree", 0.8),
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        raise ValueError(f"Modelo desconhecido '{self.model_name}'")

    def _buscar(self, X, y, SearchClass, espaco, **kwargs):
        """Executa busca de hiperparâmetros e armazena o melhor estimador."""
        search = SearchClass(
            self._base(), espaco,
            cv=self.cv, scoring="f1_macro",
            n_jobs=-1, refit=True, **kwargs,
        )
        search.fit(X, y)
        self._best_params = search.best_params_
        logger.info("%s melhores params %s F1-macro CV %.4f",
                    SearchClass.__name__, self._best_params, search.best_score_)
        self._estimator = search.best_estimator_

    def _optuna(self, X, y):
        """Otimização bayesiana com Optuna."""
        try:
            import optuna
        except ImportError:
            raise ImportError("Execute pip install optuna")
        from sklearn.model_selection import cross_val_score
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objetivo(trial):
            if self.model_name == "logistic_regression":
                m = LogisticRegression(
                    C=trial.suggest_float("C", 1e-3, 50, log=True),
                    solver=trial.suggest_categorical("solver", ["lbfgs", "saga"]),
                    max_iter=1000, random_state=42,
                )
            elif self.model_name == "linear_svc":
                m = CalibratedClassifierCV(
                    LinearSVC(C=trial.suggest_float("C", 1e-3, 20, log=True),
                              max_iter=2000, random_state=42), cv=3,
                )
            elif self.model_name == "random_forest":
                m = RandomForestClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    max_depth=trial.suggest_categorical("max_depth", [None, 10, 20, 40]),
                    min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                    random_state=42,
                )
            elif self.model_name == "lightgbm":
                from lightgbm import LGBMClassifier
                m = LGBMClassifier(
                    n_estimators=trial.suggest_int("n_estimators", 100, 500),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                    num_leaves=trial.suggest_int("num_leaves", 15, 127),
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
                    random_state=42, verbose=-1,
                )
            else:
                return 0.0
            return cross_val_score(m, X, y, cv=self.cv, scoring="f1_macro").mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objetivo, n_trials=self.n_trials, show_progress_bar=False)
        self._best_params = study.best_params
        logger.info("Optuna melhores params %s F1-macro CV %.4f",
                    self._best_params, study.best_value)
        self._estimator = self._base(self._best_params)
        self._estimator.fit(X, y)

    def fit(self, X, y) -> "Classifier":
        """Treina o classificador. Executa busca de hiperparâmetros se configurada."""
        if self.search == "manual":
            self._estimator = self._base(self.config.get("params", {}))
            self._estimator.fit(X, y)
            self._best_params = self.config.get("params", {})
        elif self.search == "grid":
            self._buscar(X, y, GridSearchCV, _GRIDS.get(self.model_name, {}))
        elif self.search == "random":
            self._buscar(X, y, RandomizedSearchCV,
                         _DISTRIBUTIONS.get(self.model_name, {}),
                         n_iter=self.n_iter, random_state=42)
        elif self.search == "optuna":
            self._optuna(X, y)
        else:
            raise ValueError(f"Modalidade de busca desconhecida '{self.search}'")
        return self

    def predict(self, X) -> np.ndarray:
        self._checar()
        return self._estimator.predict(X)

    def predict_proba(self, X) -> np.ndarray | None:
        self._checar()
        if hasattr(self._estimator, "predict_proba"):
            return self._estimator.predict_proba(X)
        return None

    def get_best_params(self) -> dict | None:
        return self._best_params

    def get_config(self) -> dict:
        return self.config.copy()

    def _checar(self):
        if self._estimator is None:
            raise RuntimeError("Execute fit() antes de predict().")

"""
Testes do projeto sentiment_pipeline.
Cobre Preprocessor (inglês e português), Vectorizer, Classifier e SentimentPipeline.

    pytest tests/ -v
    pytest tests/ -v --cov=sentiment_pipeline --cov-report=term-missing
"""

import numpy as np
import pytest
import scipy.sparse
from sklearn.datasets import make_classification

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentiment_pipeline import SentimentPipeline, Preprocessor, Vectorizer, Classifier
from sentiment_pipeline.utils import carregar_dataset, _inferir_mapeamento


# Fixtures

@pytest.fixture
def corpus_en():
    return [
        "this film was absolutely great and i loved every moment of it",
        "terrible movie boring and awful nothing good about it at all",
        "amazing story wonderful acting superb direction brilliant cast",
        "worst film ever made completely unwatchable horrid and dreadful",
        "brilliant cinema deeply moving emotional outstanding performances",
        "cinematography was beautiful and the music was simply perfect",
        "i hated every single moment of this dreadful terrible production",
        "a masterpiece of modern storytelling highly recommended to all",
        "predictable and derivative nothing original here disappointing",
        "fantastic performances from everyone in the entire brilliant cast",
    ]

@pytest.fixture
def corpus_pt():
    return [
        "esse filme foi absolutamente incrível adorei cada momento",
        "história emocionante e atuações brilhantes recomendo muito",
        "cinema de qualidade com direção impecável e roteiro perfeito",
        "uma das melhores experiências que já tive num cinema excelente",
        "simplesmente maravilhoso uma obra prima do cinema nacional",
        "filme terrível perda de tempo total não gostei nada disso",
        "péssima atuação roteiro sem sentido completamente decepcionante",
        "não recomendo para ninguém foi horrível do começo ao fim",
        "entediante e previsível nunca mais assisto um filme assim",
        "horroroso jamais vi algo tão ruim na minha vida que desperdício",
    ]

@pytest.fixture
def sentimentos_en():
    textos = [
        "I loved this movie, it was fantastic and brilliant",
        "Brilliant and touching story, highly recommended film",
        "Amazing performances, truly a masterpiece of cinema",
        "Great film with wonderful characters and story",
        "Outstanding cinema experience, loved every single moment",
        "Worst film ever, complete waste of time and money",
        "Terrible acting and boring story, hated this movie",
        "Awful movie, nothing makes any sense at all here",
        "Dull and predictable, very disappointing experience",
        "Horrible film, would not recommend to anyone ever",
    ]
    return textos, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

@pytest.fixture
def sentimentos_pt():
    textos = [
        "Adorei esse filme foi fantástico e brilhante",
        "História tocante altamente recomendado",
        "Atuações incríveis uma verdadeira obra prima",
        "Ótimo filme com personagens maravilhosos",
        "Experiência incrível adorei cada momento",
        "Pior filme já visto completa perda de tempo",
        "Péssima atuação história horrível detestei",
        "Filme terrível nada faz sentido aqui",
        "Entediante e previsível muito decepcionante",
        "Horrível não recomendaria para ninguém",
    ]
    return textos, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

@pytest.fixture
def dados_densos():
    X, y = make_classification(n_samples=200, n_features=20, n_informative=10,
                               n_classes=2, random_state=42)
    return X.astype(np.float32), y


# Preprocessor — Inglês

class TestPreprocessorIngles:
    def test_lowercase(self):
        pre = Preprocessor({"language": "english", "lowercase": True})
        assert pre.transform(["Hello World"])[0] == pre.transform(["hello world"])[0]

    def test_remove_url(self):
        pre = Preprocessor({"language": "english", "remove_urls": True})
        assert "http" not in pre.transform(["Visit https://example.com"])[0]

    def test_emoticon_smile(self):
        pre = Preprocessor({"language": "english", "normalize_emojis": True})
        assert "smile" in pre.transform(["Great film :)"])[0]

    def test_keep_negations_en(self):
        pre = Preprocessor({"language": "english", "lowercase": True,
                            "remove_stopwords": "keep_negations"})
        for neg in ["not", "never", "no", "nor"]:
            assert neg in pre.transform([f"I {neg} liked it"])[0]

    def test_handle_negations_en(self):
        pre = Preprocessor({"language": "english", "lowercase": True,
                            "remove_stopwords": False, "normalization": None,
                            "handle_negations": True})
        resultado = pre.transform(["I did not love this film"])[0]
        assert "love_neg" in resultado or "love_NEG" in resultado

    def test_janela_negacao(self):
        pre = Preprocessor({"language": "english", "lowercase": True,
                            "remove_stopwords": False, "normalization": None,
                            "handle_negations": True})
        resultado = pre.transform(["not one two three four five six seven"])[0]
        neg_tokens = [t for t in resultado.split() if "_NEG" in t or "_neg" in t]
        assert len(neg_tokens) <= 5

    def test_stemming_en(self):
        pre = Preprocessor({"language": "english", "lowercase": True,
                            "normalization": "stemming", "remove_stopwords": False})
        tokens = pre.transform(["plays playing played"])[0].split()
        assert len(set(tokens)) <= 2


# Preprocessor — Português

class TestPreprocessorPortugues:
    def test_lowercase_pt(self):
        pre = Preprocessor({"language": "portuguese", "lowercase": True})
        assert pre.transform(["ÓTIMO Filme"])[0] == pre.transform(["ótimo filme"])[0]

    def test_keep_negations_pt(self):
        pre = Preprocessor({"language": "portuguese", "lowercase": True,
                            "remove_stopwords": "keep_negations"})
        for neg in ["não", "nunca", "jamais", "nem"]:
            resultado = pre.transform([f"eu {neg} gostei do filme"])[0]
            assert neg in resultado, f"Negação '{neg}' deveria ser preservada"

    def test_handle_negations_pt(self):
        pre = Preprocessor({"language": "portuguese", "lowercase": True,
                            "remove_stopwords": False, "normalization": None,
                            "handle_negations": True})
        resultado = pre.transform(["Não gostei nada desse filme"])[0]
        assert "gostei_NEG" in resultado or "gostei_neg" in resultado

    def test_stemming_pt(self):
        pre = Preprocessor({"language": "portuguese", "lowercase": True,
                            "normalization": "stemming", "remove_stopwords": False})
        resultado = pre.transform(["filmando filmes filmado"])[0]
        tokens = resultado.split()
        assert len(set(tokens)) <= 2

    def test_negation_words_customizadas(self):
        pre = Preprocessor({
            "language": "portuguese",
            "lowercase": True,
            "remove_stopwords": "keep_negations",
            "negation_words": ["nunca", "jamais", "sequer"],
        })
        assert "nunca" in pre.transform(["eu nunca gostei"])[0]
        assert pre.get_negation_words() == {"nunca", "jamais", "sequer"}

    def test_pipeline_completo_pt(self, corpus_pt):
        pre = Preprocessor({
            "language": "portuguese",
            "lowercase": True,
            "remove_stopwords": "keep_negations",
            "normalization": "stemming",
            "handle_negations": True,
            "remove_urls": True,
            "normalize_emojis": True,
        })
        resultado = pre.transform(corpus_pt)
        assert len(resultado) == len(corpus_pt)
        assert all(isinstance(r, str) and len(r) > 0 for r in resultado)


# Preprocessor — geral

class TestPreprocessorGeral:
    def test_lista_vazia(self):
        assert Preprocessor({"language": "english"}).transform([]) == []

    def test_none_tolerado(self):
        assert isinstance(Preprocessor({}).transform([None])[0], str)

    def test_get_config_copia(self):
        cfg = {"language": "english"}
        pre = Preprocessor(cfg)
        copia = pre.get_config()
        copia["extra"] = True
        assert "extra" not in pre.get_config()

    def test_get_negation_words_retorna_set(self):
        pre = Preprocessor({"language": "english"})
        assert isinstance(pre.get_negation_words(), set)


# Vectorizer

class TestVectorizer:
    def test_bow_sparse(self, corpus_en):
        X = Vectorizer({"strategy": "bow", "max_features": 100}).fit_transform(corpus_en)
        assert scipy.sparse.issparse(X)

    def test_bow_binario(self, corpus_en):
        X = Vectorizer({"strategy": "bow", "binary": True,
                        "max_features": 100}).fit_transform(corpus_en)
        assert set(np.unique(X.toarray())).issubset({0, 1})

    def test_tfidf_shape(self, corpus_en):
        X = Vectorizer({"strategy": "tfidf", "max_features": 100,
                        "min_df": 1}).fit_transform(corpus_en)
        assert X.shape[0] == len(corpus_en)

    def test_tfidf_norma_l2(self, corpus_en):
        X = Vectorizer({"strategy": "tfidf", "norm": "l2", "max_features": 200,
                        "min_df": 1}).fit_transform(corpus_en).toarray()
        normas = np.linalg.norm(X, axis=1)
        assert np.allclose(normas[normas > 0], 1.0, atol=1e-5)

    def test_tfidf_svd_denso(self, corpus_en):
        X = Vectorizer({"strategy": "tfidf_svd", "max_features": 100,
                        "svd_components": 5, "min_df": 1}).fit_transform(corpus_en)
        assert isinstance(X, np.ndarray) and X.shape == (len(corpus_en), 5)

    def test_transform_sem_fit_erro(self, corpus_en):
        with pytest.raises(RuntimeError):
            Vectorizer({"strategy": "tfidf"}).transform(corpus_en)

    def test_transform_novos_textos(self, corpus_en):
        vec = Vectorizer({"strategy": "tfidf", "max_features": 50, "min_df": 1})
        X_tr = vec.fit_transform(corpus_en)
        X_novo = vec.transform(["amazing new film here"])
        assert X_novo.shape[1] == X_tr.shape[1]

    def test_estrategia_invalida(self):
        with pytest.raises(ValueError):
            Vectorizer({"strategy": "invalida"}).fit_transform(["texto"])

    def test_word2vec_corpus(self, corpus_en):
        pytest.importorskip("gensim")
        X = Vectorizer({"strategy": "word2vec", "glove_path": None,
                        "embedding_dim": 20}).fit_transform(corpus_en)
        assert X.shape == (len(corpus_en), 20)


# Classifier

class TestClassifier:
    @pytest.mark.parametrize("modelo", ["logistic_regression", "linear_svc", "random_forest"])
    def test_fit_predict(self, dados_densos, modelo):
        X, y = dados_densos
        clf = Classifier({"model": modelo, "search": "manual", "params": {}})
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == y.shape and set(np.unique(pred)).issubset({0, 1})

    def test_naive_bayes_sparse(self):
        X = scipy.sparse.random(100, 50, density=0.3, format="csr")
        X.data = np.abs(X.data)
        y = np.random.randint(0, 2, 100)
        Classifier({"model": "naive_bayes", "search": "manual",
                    "vectorizer_strategy": "tfidf"}).fit(X, y).predict(X)

    def test_predict_proba_logreg(self, dados_densos):
        X, y = dados_densos
        clf = Classifier({"model": "logistic_regression", "search": "manual",
                          "params": {"C": 1.0}})
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba is not None and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_grid_search(self, dados_densos):
        X, y = dados_densos
        clf = Classifier({"model": "logistic_regression", "search": "grid", "cv": 2})
        clf.fit(X, y)
        assert clf.get_best_params() is not None

    def test_random_search(self, dados_densos):
        X, y = dados_densos
        Classifier({"model": "logistic_regression", "search": "random",
                    "cv": 2, "n_iter": 5}).fit(X, y).predict(X)

    def test_modelo_invalido(self, dados_densos):
        X, y = dados_densos
        with pytest.raises(ValueError):
            Classifier({"model": "inexistente", "search": "manual"}).fit(X, y)

    def test_busca_invalida(self, dados_densos):
        X, y = dados_densos
        with pytest.raises(ValueError):
            Classifier({"model": "logistic_regression", "search": "invalida"}).fit(X, y)

    def test_sem_fit_erro(self, dados_densos):
        X, _ = dados_densos
        with pytest.raises(RuntimeError):
            Classifier({"model": "logistic_regression"}).predict(X)


# SentimentPipeline

class TestPipeline:
    @pytest.fixture
    def cfg_en(self):
        return {
            "preprocessor": {"language": "english", "lowercase": True},
            "vectorizer":   {"strategy": "tfidf", "max_features": 300,
                             "ngram_range": (1, 1), "min_df": 1},
            "classifier":   {"model": "logistic_regression", "search": "manual",
                             "params": {"C": 1.0, "max_iter": 200}},
        }

    @pytest.fixture
    def cfg_pt(self):
        return {
            "preprocessor": {"language": "portuguese", "lowercase": True,
                             "remove_stopwords": "keep_negations",
                             "normalization": "stemming", "handle_negations": True},
            "vectorizer":   {"strategy": "tfidf", "max_features": 300,
                             "ngram_range": (1, 2), "min_df": 1},
            "classifier":   {"model": "logistic_regression", "search": "manual",
                             "params": {"C": 1.0, "max_iter": 200}},
        }

    def test_fit_predict_en(self, cfg_en, sentimentos_en):
        textos, rotulos = sentimentos_en
        p = SentimentPipeline(cfg_en)
        p.fit(textos, rotulos)
        pred = p.predict(textos)
        assert len(pred) == len(textos) and set(np.unique(pred)).issubset({0, 1})

    def test_fit_predict_pt(self, cfg_pt, sentimentos_pt):
        textos, rotulos = sentimentos_pt
        p = SentimentPipeline(cfg_pt, nomes_classes=["negativo", "positivo"])
        p.fit(textos, rotulos)
        pred = p.predict(textos)
        assert len(pred) == len(textos)

    def test_evaluate_metricas(self, cfg_en, sentimentos_en):
        textos, rotulos = sentimentos_en
        p = SentimentPipeline(cfg_en)
        p.fit(textos, rotulos)
        m = p.evaluate(textos, rotulos)
        assert all(k in m for k in ["acuracia", "f1_macro", "relatorio"])
        assert 0 <= m["acuracia"] <= 1

    def test_predict_texto_individual(self, cfg_en, sentimentos_en):
        textos, rotulos = sentimentos_en
        p = SentimentPipeline(cfg_en)
        p.fit(textos, rotulos)
        r = p.predict_texto("This movie was amazing!")
        assert r["classe"] in (0, 1) and isinstance(r["rotulo"], str)

    def test_nomes_classes_customizados(self, cfg_pt, sentimentos_pt):
        textos, rotulos = sentimentos_pt
        p = SentimentPipeline(cfg_pt, nomes_classes=["negativo", "positivo"])
        p.fit(textos, rotulos)
        r = p.predict_texto("Adorei esse filme!")
        assert r["rotulo"] in ("negativo", "positivo")
        assert r["rotulo"] in ("negativo", "positivo")

    def test_erro_sem_fit(self, cfg_en):
        p = SentimentPipeline(cfg_en)
        with pytest.raises(RuntimeError):
            p.predict(["texto"])
        with pytest.raises(RuntimeError):
            p.evaluate(["texto"], [1])

    def test_serializar_e_carregar(self, cfg_en, sentimentos_en, tmp_path):
        from sentiment_pipeline.utils import salvar_pipeline, carregar_pipeline
        textos, rotulos = sentimentos_en
        p = SentimentPipeline(cfg_en)
        p.fit(textos, rotulos)
        caminho = str(tmp_path / "pipeline_teste.pkl")
        salvar_pipeline(p, caminho)
        p2 = carregar_pipeline(caminho)
        assert np.array_equal(p.predict(textos), p2.predict(textos))

    def test_tempo_treino_registrado(self, cfg_en, sentimentos_en):
        textos, rotulos = sentimentos_en
        p = SentimentPipeline(cfg_en)
        p.fit(textos, rotulos)
        assert p.tempo_treino is not None and p.tempo_treino >= 0

    @pytest.mark.parametrize("strategy", ["bow", "tfidf"])
    def test_multiplas_estrategias(self, strategy, sentimentos_en):
        textos, rotulos = sentimentos_en
        cfg = {
            "preprocessor": {"language": "english", "lowercase": True},
            "vectorizer":   {"strategy": strategy, "max_features": 100, "min_df": 1},
            "classifier":   {"model": "logistic_regression", "search": "manual",
                             "params": {"C": 1.0, "max_iter": 200}},
        }
        p = SentimentPipeline(cfg)
        p.fit(textos, rotulos)
        assert len(p.predict(textos[:3])) == 3


# Utils — carregar_dataset e _inferir_mapeamento

class TestCarregarDataset:
    def test_inferir_mapeamento_ingles(self):
        m = _inferir_mapeamento(np.array(["positive", "negative"]))
        assert m["positive"] == 1 and m["negative"] == 0

    def test_inferir_mapeamento_portugues(self):
        m = _inferir_mapeamento(np.array(["positivo", "negativo"]))
        assert m["positivo"] == 1 and m["negativo"] == 0

    def test_inferir_mapeamento_binario(self):
        m = _inferir_mapeamento(np.array(["1", "0"]))
        assert m["1"] == 1 and m["0"] == 0

    def test_carregar_csv_customizado(self, tmp_path):
        from sentiment_pipeline.utils import carregar_dataset
        import pandas as pd
        df = pd.DataFrame({
            "texto": ["Ótimo filme!", "Péssimo filme.", "Adorei!", "Odiei."],
            "classe": ["positivo", "negativo", "positivo", "negativo"],
        })
        csv_path = str(tmp_path / "teste.csv")
        df.to_csv(csv_path, index=False)

        textos, rotulos = carregar_dataset(
            csv_path,
            coluna_texto="texto",
            coluna_rotulo="classe",
            mapeamento_classes={"positivo": 1, "negativo": 0},
        )
        assert len(textos) == 4
        assert list(rotulos) == [1, 0, 1, 0]

    def test_coluna_invalida_levanta_erro(self, tmp_path):
        from sentiment_pipeline.utils import carregar_dataset
        import pandas as pd
        df = pd.DataFrame({"review": ["texto"], "sentiment": ["positive"]})
        csv_path = str(tmp_path / "teste.csv")
        df.to_csv(csv_path, index=False)
        with pytest.raises(ValueError, match="inexistente"):
            carregar_dataset(csv_path, coluna_texto="inexistente")

    def test_arquivo_inexistente_levanta_erro(self):
        from sentiment_pipeline.utils import carregar_dataset
        with pytest.raises(FileNotFoundError):
            carregar_dataset("arquivo_que_nao_existe.csv")

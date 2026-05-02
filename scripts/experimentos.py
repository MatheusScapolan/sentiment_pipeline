"""
Grade completa de experimentos comparativos.

Executa 10 combinações de pré-processamento, vetorização e classificação,
avalia cada uma nos conjuntos de validação e teste, e salva o melhor
pipeline em disco.

O pipeline é agnóstico de dataset: funciona com qualquer CSV em qualquer
idioma, bastando informar as colunas corretas.

Uso com IMDB (padrão)
    python scripts/experimentos.py --dataset data/IMDB_Dataset.csv

Uso com dataset customizado em português
    python scripts/experimentos.py \\
        --dataset data/meu_dataset.csv \\
        --coluna_texto texto \\
        --coluna_rotulo classe \\
        --classe_positiva positivo \\
        --classe_negativa negativo \\
        --idioma portuguese

Flags completas
    --dataset          caminho para o CSV (obrigatório)
    --coluna_texto     nome da coluna com os textos. Padrão "review"
    --coluna_rotulo    nome da coluna com os rótulos. Padrão "sentiment"
    --classe_positiva  valor do rótulo positivo. Padrão "positive"
    --classe_negativa  valor do rótulo negativo. Padrão "negative"
    --idioma           idioma para pré-processamento. Padrão "english"
    --output           pasta de saída. Padrão "outputs/"
    --subset           número de amostras para teste rápido
    --seed             semente aleatória. Padrão 42
    --glove            caminho para arquivo GloVe .txt pré-treinado
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentiment_pipeline import SentimentPipeline
from sentiment_pipeline.utils import (
    carregar_dataset, dividir_dataset,
    salvar_pipeline, gerar_relatorio, configurar_logging,
)

configurar_logging("INFO")
logger = logging.getLogger(__name__)


def grade_experimentos(idioma: str = "english", glove_path=None) -> list:
    """
    Retorna os 10 experimentos configurados para o idioma informado.
    As negações e stopwords se adaptam automaticamente ao idioma via Preprocessor.
    """

    # Configuração de pré-processamento base com tratamento completo de negações
    pre_completo = {
        "language": idioma,
        "lowercase": True,
        "remove_stopwords": "keep_negations",
        "normalization": "lemmatization" if idioma == "english" else "stemming",
        "handle_negations": True,
        "remove_urls": True,
        "normalize_emojis": True,
    }

    # Pré-processamento leve sem remoção de stopwords
    pre_leve = {
        "language": idioma,
        "lowercase": True,
        "remove_stopwords": False,
        "normalization": None,
        "handle_negations": True,
        "remove_urls": True,
        "normalize_emojis": True,
    }

    # Pré-processamento com stemming e remoção simples de stopwords
    pre_stemming = {
        "language": idioma,
        "lowercase": True,
        "remove_stopwords": True,
        "normalization": "stemming",
        "handle_negations": False,
        "remove_urls": True,
        "normalize_emojis": False,
    }

    experimentos = [

        # 1. Baseline: BoW simples + Naive Bayes com busca em grid
        {
            "nome": "bow_naive_bayes",
            "preprocessor": {**pre_leve, "normalize_emojis": False},
            "vectorizer": {"strategy": "bow", "binary": False, "max_features": 30000},
            "classifier": {"model": "naive_bayes", "search": "grid", "cv": 3},
        },

        # 2. BoW binário + BernoulliNB com busca em grid
        {
            "nome": "bow_binario_bernoulli",
            "preprocessor": pre_stemming,
            "vectorizer": {"strategy": "bow", "binary": True, "max_features": 40000},
            "classifier": {"model": "naive_bayes", "search": "grid", "cv": 3},
        },

        # 3. TF-IDF bigramas + Regressão Logística manual
        {
            "nome": "tfidf_bigrama_logreg_manual",
            "preprocessor": pre_completo,
            "vectorizer": {
                "strategy": "tfidf", "sublinear_tf": True,
                "ngram_range": (1, 2), "max_features": 50000, "norm": "l2",
            },
            "classifier": {
                "model": "logistic_regression", "search": "manual",
                "params": {"C": 1.0, "max_iter": 1000},
            },
        },

        # 4. TF-IDF bigramas + LinearSVC com busca aleatória
        {
            "nome": "tfidf_bigrama_linearsvc",
            "preprocessor": pre_completo,
            "vectorizer": {
                "strategy": "tfidf", "sublinear_tf": True,
                "ngram_range": (1, 2), "max_features": 80000, "norm": "l2",
            },
            "classifier": {"model": "linear_svc", "search": "random", "cv": 3, "n_iter": 10},
        },

        # 5. TF-IDF unigramas + Regressão Logística com busca aleatória
        # Sem remover stopwords para capturar contexto completo
        {
            "nome": "tfidf_unigram_logreg_random",
            "preprocessor": pre_leve,
            "vectorizer": {
                "strategy": "tfidf", "sublinear_tf": True,
                "ngram_range": (1, 1), "max_features": 60000, "norm": "l2",
            },
            "classifier": {
                "model": "logistic_regression", "search": "random",
                "cv": 3, "n_iter": 15,
            },
        },

        # 6. TF-IDF trigramas + LinearSVC manual
        # Trigramas capturam expressões de 3 palavras como "not very good"
        {
            "nome": "tfidf_trigrama_linearsvc",
            "preprocessor": {**pre_completo, "normalize_emojis": False},
            "vectorizer": {
                "strategy": "tfidf", "sublinear_tf": True,
                "ngram_range": (1, 3), "max_features": 100000, "norm": "l2",
            },
            "classifier": {
                "model": "linear_svc", "search": "manual", "params": {"C": 1.0},
            },
        },

        # 7. TF-IDF unigramas + LightGBM com busca aleatória
        {
            "nome": "tfidf_unigram_lightgbm",
            "preprocessor": {**pre_completo, "normalize_emojis": False},
            "vectorizer": {
                "strategy": "tfidf", "sublinear_tf": True,
                "ngram_range": (1, 1), "max_features": 40000, "norm": "l2",
            },
            "classifier": {"model": "lightgbm", "search": "random", "cv": 3, "n_iter": 10},
        },

        # 8. TF-IDF + SVD 100 dimensões (LSA) + Regressão Logística com grid
        {
            "nome": "tfidf_svd100_logreg",
            "preprocessor": pre_stemming,
            "vectorizer": {
                "strategy": "tfidf_svd", "sublinear_tf": True,
                "ngram_range": (1, 1), "max_features": 80000,
                "norm": "l2", "svd_components": 100,
            },
            "classifier": {"model": "logistic_regression", "search": "grid", "cv": 3},
        },

        # 9. TF-IDF + SVD 300 dimensões + Random Forest manual
        {
            "nome": "tfidf_svd300_rf",
            "preprocessor": pre_stemming,
            "vectorizer": {
                "strategy": "tfidf_svd", "sublinear_tf": True,
                "ngram_range": (1, 1), "max_features": 80000,
                "norm": "l2", "svd_components": 300,
            },
            "classifier": {
                "model": "random_forest", "search": "manual",
                "params": {"n_estimators": 200},
            },
        },

        # 10. Word2Vec com GloVe ou treinado no corpus
        {
            "nome": "glove_logreg" if (glove_path and os.path.isfile(str(glove_path))) else "word2vec_corpus_logreg",
            "preprocessor": {**pre_leve, "normalize_emojis": False},
            "vectorizer": {
                "strategy": "word2vec", "glove_path": glove_path, "embedding_dim": 100,
            },
            "classifier": {
                "model": "logistic_regression", "search": "manual",
                "params": {"C": 1.0, "max_iter": 1000},
            },
        },
    ]

    return experimentos


def main():
    parser = argparse.ArgumentParser(
        description="Grade de experimentos de análise de sentimentos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset",         required=True,
                        help="Caminho para o arquivo CSV.")
    parser.add_argument("--coluna_texto",    default="review",
                        help="Nome da coluna com os textos. Padrão: review")
    parser.add_argument("--coluna_rotulo",   default="sentiment",
                        help="Nome da coluna com os rótulos. Padrão: sentiment")
    parser.add_argument("--classe_positiva", default="positive",
                        help="Valor do rótulo positivo. Padrão: positive")
    parser.add_argument("--classe_negativa", default="negative",
                        help="Valor do rótulo negativo. Padrão: negative")
    parser.add_argument("--idioma",          default="english",
                        help="Idioma para pré-processamento. Padrão: english")
    parser.add_argument("--separador",       default=",",
                        help="Separador do CSV. Padrão: ,")
    parser.add_argument("--encoding",        default="utf-8",
                        help="Encoding do CSV. Padrão: utf-8")
    parser.add_argument("--output",          default="outputs")
    parser.add_argument("--subset",          type=int, default=None)
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--glove",           default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    mapeamento = {
        args.classe_positiva: 1,
        args.classe_negativa: 0,
    }

    textos, rotulos = carregar_dataset(
        args.dataset,
        coluna_texto=args.coluna_texto,
        coluna_rotulo=args.coluna_rotulo,
        mapeamento_classes=mapeamento,
        separador=args.separador,
        codificacao=args.encoding,
    )

    if args.subset:
        textos  = textos.iloc[:args.subset]
        rotulos = rotulos.iloc[:args.subset]
        logger.info("Usando subconjunto de %d amostras.", args.subset)

    X_tr, X_val, X_te, y_tr, y_val, y_te = dividir_dataset(
        textos, rotulos, seed=args.seed
    )

    nomes_classes = [args.classe_negativa, args.classe_positiva]
    experimentos = grade_experimentos(idioma=args.idioma, glove_path=args.glove)
    resultados, melhor_f1, melhor_pipeline, melhor_nome = [], -1, None, None

    logger.info("Iniciando %d experimentos (idioma=%s)...", len(experimentos), args.idioma)
    sep = "=" * 58

    for i, cfg in enumerate(experimentos, 1):
        nome = cfg["nome"]
        print(f"\n{sep}\nExperimento {i}/{len(experimentos)}: {nome}\n{sep}")

        try:
            pipeline = SentimentPipeline(
                {"preprocessor": cfg["preprocessor"],
                 "vectorizer":   cfg["vectorizer"],
                 "classifier":   cfg["classifier"]},
                nome=nome,
                nomes_classes=nomes_classes,
            )
            pipeline.fit(X_tr, y_tr)
            m_val  = pipeline.evaluate(X_val, y_val)
            m_test = pipeline.evaluate(X_te, y_te)

            print(f"\n  Validação  F1-macro {m_val['f1_macro']}  Acurácia {m_val['acuracia']}")
            print(f"  Teste      F1-macro {m_test['f1_macro']}  Acurácia {m_test['acuracia']}")
            if m_test.get("auc_roc"):
                print(f"  AUC-ROC    {m_test['auc_roc']}")

            resultados.append({
                "nome": nome,
                "config": cfg,
                "metricas_val":   m_val,
                "metricas_teste": m_test,
            })

            if m_val["f1_macro"] > melhor_f1:
                melhor_f1, melhor_pipeline, melhor_nome = m_val["f1_macro"], pipeline, nome

        except Exception as exc:
            logger.error("Experimento '%s' falhou: %s", nome, exc, exc_info=True)
            resultados.append({
                "nome": nome, "config": cfg,
                "metricas_val": {}, "metricas_teste": {},
            })

    csv_path = os.path.join(args.output, "resultados_experimentos.csv")
    df = gerar_relatorio(resultados, caminho_csv=csv_path)

    print(f"\n\n{sep}\nRANKING FINAL por F1-macro na validação\n{sep}")
    print(df[["experimento", "vetorizador", "classificador",
              "f1_macro_val", "acuracia_val", "f1_macro_teste"]].to_string(index=False))

    if melhor_pipeline:
        pkl_path = os.path.join(args.output, "melhor_pipeline.pkl")
        salvar_pipeline(melhor_pipeline, pkl_path)
        print(f"\nMelhor pipeline: '{melhor_nome}'  F1-macro val {melhor_f1:.4f}")
        print(f"Pipeline salvo em: {pkl_path}")

    print(f"Relatório CSV salvo em: {csv_path}")
    print(f"\nPara visualizar no dashboard execute:")
    print(f"  python scripts/dashboard.py")


if __name__ == "__main__":
    main()

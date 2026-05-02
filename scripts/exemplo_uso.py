"""
Valida a instalação e demonstra o uso do pipeline com dados sintéticos.
Roda sem precisar do dataset IMDB.

    python scripts/exemplo_uso.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentiment_pipeline import SentimentPipeline
from sentiment_pipeline.utils import avaliar, formatar_resultado, configurar_logging

configurar_logging("INFO")

# Dados em inglês
TEXTOS_EN = [
    "This movie was absolutely fantastic, I loved every minute of it!",
    "A masterpiece of cinema, truly moving and beautifully crafted.",
    "Outstanding performances from the cast, highly recommended.",
    "One of the best films I have seen this year, simply brilliant.",
    "Wonderful storytelling and a deeply emotional journey.",
    "This was a terrible film, boring and poorly written.",
    "What a waste of time, the plot made no sense at all.",
    "Awful acting and dreadful story, I regret watching this.",
    "The worst movie I have ever seen, completely unwatchable.",
    "Dull and predictable from start to finish, very disappointing.",
]
ROTULOS_EN = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Dados em português
TEXTOS_PT = [
    "Filme incrível, adorei cada momento da história!",
    "Uma obra prima do cinema, profundamente emocionante.",
    "Atuações fantásticas, recomendo muito esse filme.",
    "Um dos melhores filmes que já assisti, simplesmente brilhante.",
    "Narrativa maravilhosa e uma jornada muito emocionante.",
    "Filme horrível, entediante e muito mal escrito.",
    "Que desperdício de tempo, o enredo não faz sentido algum.",
    "Péssima atuação e história terrível, me arrependo de ter assistido.",
    "Pior filme que já vi, completamente impossível de assistir.",
    "Previsível e sem originalidade do começo ao fim, decepcionante.",
]
ROTULOS_PT = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]


def rodar_pipeline(textos, rotulos, idioma, nome):
    print(f"\n{'='*55}")
    print(f"Pipeline em {idioma.upper()}: {nome}")
    print('='*55)

    config = {
        "preprocessor": {
            "language": idioma,
            "lowercase": True,
            "remove_stopwords": "keep_negations",
            "normalization": "lemmatization" if idioma == "english" else "stemming",
            "handle_negations": True,
            "remove_urls": True,
            "normalize_emojis": True,
        },
        "vectorizer": {
            "strategy": "tfidf",
            "sublinear_tf": True,
            "ngram_range": (1, 2),
            "max_features": 5000,
            "norm": "l2",
            "min_df": 1,
        },
        "classifier": {
            "model": "logistic_regression",
            "search": "manual",
            "params": {"C": 1.0, "max_iter": 500},
        },
    }

    nomes_classes = (
        ["negative", "positive"] if idioma == "english"
        else ["negativo", "positivo"]
    )

    pipeline = SentimentPipeline(config, nome=nome, nomes_classes=nomes_classes)
    pipeline.fit(textos, rotulos)

    print(pipeline.resumo())

    metricas = pipeline.evaluate(textos, rotulos, )
    print(f"\nMétricas (treino sintético):")
    print(formatar_resultado(metricas))
    return pipeline


def main():
    p_en = rodar_pipeline(TEXTOS_EN, ROTULOS_EN, "english", "exemplo_en")
    p_pt = rodar_pipeline(TEXTOS_PT, ROTULOS_PT, "portuguese", "exemplo_pt")

    print("\n" + "="*55)
    print("Classificação de textos individuais")
    print("="*55)

    testes = [
        (p_en, "I absolutely loved this film, it was spectacular!"),
        (p_en, "This movie was a complete disaster, I hated it."),
        (p_pt, "Adorei esse filme, foi simplesmente espetacular!"),
        (p_pt, "Não gostei nada desse filme, foi um desastre completo."),
    ]
    for pipeline, texto in testes:
        r = pipeline.predict_texto(texto)
        print(f"  [{r['rotulo'].upper():12}] conf={r['confianca']} | {texto[:60]}")

    print("\nPipeline funcionando. Para rodar com o IMDB:")
    print("  python scripts/experimentos.py --dataset data/IMDB_Dataset.csv --subset 5000")
    print("\nPara rodar com dataset em português:")
    print("  python scripts/experimentos.py \\")
    print("      --dataset data/meu_dataset.csv \\")
    print("      --coluna_texto texto --coluna_rotulo classe \\")
    print("      --classe_positiva positivo --classe_negativa negativo \\")
    print("      --idioma portuguese")


if __name__ == "__main__":
    main()

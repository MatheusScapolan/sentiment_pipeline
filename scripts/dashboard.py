"""
Inicia o dashboard Gradio do Sentiment Pipeline.

Uso
    python scripts/dashboard.py

Com caminhos personalizados
    python scripts/dashboard.py \\
        --csv outputs/resultados_experimentos.csv \\
        --pipeline outputs/melhor_pipeline.pkl

Para compartilhar via link público temporário
    python scripts/dashboard.py --share

O dashboard abre automaticamente no navegador em http://localhost:7860
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sentiment_pipeline.dashboard import criar_dashboard


def main():
    parser = argparse.ArgumentParser(description="Dashboard Gradio do Sentiment Pipeline.")
    parser.add_argument(
        "--csv",
        default="outputs/resultados_experimentos.csv",
        help="Caminho para o CSV de resultados dos experimentos.",
    )
    parser.add_argument(
        "--pipeline",
        default="outputs/melhor_pipeline.pkl",
        help="Caminho para o pipeline treinado (.pkl).",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Gera link público temporário via servidor Gradio.",
    )
    args = parser.parse_args()

    print("Iniciando dashboard em http://localhost:7860")
    criar_dashboard(
        caminho_csv=args.csv,
        caminho_pkl=args.pipeline,
        compartilhar=args.share,
    )


if __name__ == "__main__":
    main()

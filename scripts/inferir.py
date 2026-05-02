"""
Classifica textos com um pipeline treinado e salvo em disco.

Modo interativo (digita textos no terminal)
    python scripts/inferir.py --pipeline outputs/melhor_pipeline.pkl

Modo lote (um texto por linha em arquivo)
    python scripts/inferir.py --pipeline outputs/melhor_pipeline.pkl --arquivo textos.txt

Saída CSV redirecionada para arquivo
    python scripts/inferir.py --pipeline outputs/melhor_pipeline.pkl --arquivo textos.txt > saida.csv
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sentiment_pipeline.utils import carregar_pipeline


def main():
    parser = argparse.ArgumentParser(description="Inferência com pipeline treinado.")
    parser.add_argument("--pipeline", required=True,
                        help="Caminho para o arquivo .pkl do pipeline.")
    parser.add_argument("--arquivo",  default=None,
                        help="Arquivo com um texto por linha (modo lote).")
    parser.add_argument("--encoding", default="utf-8",
                        help="Codificação do arquivo de entrada. Padrão utf-8.")
    args = parser.parse_args()

    if not os.path.isfile(args.pipeline):
        print(f"Arquivo não encontrado: {args.pipeline}")
        sys.exit(1)

    pipeline = carregar_pipeline(args.pipeline)
    print(f"Pipeline '{getattr(pipeline, 'nome', 'desconhecido')}' carregado.")
    print(f"Idioma: {pipeline.config.get('preprocessor', {}).get('language', 'english')}")
    print(f"Classes: {pipeline.nomes_classes}\n")

    if args.arquivo:
        with open(args.arquivo, encoding=args.encoding) as f:
            linhas = [l.strip() for l in f if l.strip()]
        print("texto,classe,rotulo,confianca")
        for linha in linhas:
            r = pipeline.predict_texto(linha)
            texto_curto = linha[:80].replace('"', "'")
            print(f'"{texto_curto}",{r["classe"]},{r["rotulo"]},{r["confianca"]}')
    else:
        print("Digite um texto e pressione Enter para classificar.")
        print("Linha vazia ou Ctrl+C encerra.\n")
        while True:
            try:
                texto = input("Texto: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nEncerrando.")
                break
            if not texto:
                break
            r = pipeline.predict_texto(texto)
            print(f"  Resultado  : {r['rotulo'].upper()}")
            print(f"  Confiança  : {r['confianca']}\n")


if __name__ == "__main__":
    main()

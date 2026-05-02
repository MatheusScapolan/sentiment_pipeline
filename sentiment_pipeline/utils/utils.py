"""
Funções utilitárias compartilhadas: carregamento de dados agnóstico de dataset,
avaliação de modelos, serialização e geração de relatórios comparativos.

O carregamento de dados foi generalizado para aceitar qualquer CSV,
com configuração explícita de colunas, idioma e mapeamento de classes.
"""

import logging
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def carregar_dataset(
    caminho: str,
    coluna_texto: str = "review",
    coluna_rotulo: str = "sentiment",
    mapeamento_classes: dict = None,
    codificacao: str = "utf-8",
    separador: str = ",",
) -> tuple:
    """
    Carrega qualquer CSV e retorna (textos, rótulos) prontos para uso no pipeline.
    Agnóstico de dataset: funciona com IMDB, datasets em português, ou qualquer outro.

    Parâmetros
        caminho (str)              caminho para o arquivo CSV.
        coluna_texto (str)         nome da coluna com os textos. Padrão "review".
        coluna_rotulo (str)        nome da coluna com os rótulos. Padrão "sentiment".
        mapeamento_classes (dict)  mapeamento de valores da coluna para 0 e 1.
                                   Se None, tenta inferir automaticamente.
                                   Exemplos
                                       {"positive": 1, "negative": 0}
                                       {"positivo": 1, "negativo": 0}
                                       {"5": 1, "1": 0}
        codificacao (str)          codificação do arquivo. Padrão "utf-8".
        separador (str)            separador de colunas. Padrão ",".

    Retorna
        Tupla (pd.Series textos, pd.Series rótulos inteiros 0/1).

    Exemplos de uso

        Dataset IMDB padrão
            textos, rotulos = carregar_dataset("data/IMDB_Dataset.csv")

        Dataset customizado em português
            textos, rotulos = carregar_dataset(
                "data/meu_dataset.csv",
                coluna_texto="texto",
                coluna_rotulo="classe",
                mapeamento_classes={"positivo": 1, "negativo": 0},
            )

        Dataset com separador ponto-e-vírgula
            textos, rotulos = carregar_dataset(
                "data/dados.csv",
                coluna_texto="comentario",
                coluna_rotulo="sentimento",
                separador=";",
            )
    """
    if not os.path.isfile(caminho):
        raise FileNotFoundError(
            f"Dataset não encontrado em '{caminho}'.\n"
            "Verifique o caminho e tente novamente."
        )

    df = pd.read_csv(
        caminho,
        sep=separador,
        encoding=codificacao,
        engine="python",
        on_bad_lines="skip",
    )

    logger.info("Colunas disponíveis: %s", list(df.columns))

    if coluna_texto not in df.columns:
        raise ValueError(
            f"Coluna de texto '{coluna_texto}' não encontrada. "
            f"Colunas disponíveis: {list(df.columns)}"
        )
    if coluna_rotulo not in df.columns:
        if coluna_rotulo == "label" and "overall_rating" in df.columns:
            logger.info("Criando coluna 'label' a partir de 'overall_rating'.")
            df = df[df["overall_rating"] != 3]
            df["label"] = df["overall_rating"].apply(lambda x: 1 if x >= 4 else 0)
            logger.info("Colunas após tratamento: %s", list(df.columns))
        else:
            raise ValueError(
                f"Coluna de rótulo '{coluna_rotulo}' não encontrada. "
                f"Colunas disponíveis: {list(df.columns)}"
            )

    df = df.dropna(subset=[coluna_texto, coluna_rotulo])

    # Inferência automática do mapeamento quando não fornecido
    if mapeamento_classes is None:
        valores_unicos = df[coluna_rotulo].astype(str).str.strip().unique()
        mapeamento_classes = _inferir_mapeamento(valores_unicos)
        logger.info("Mapeamento de classes inferido automaticamente: %s", mapeamento_classes)

    rotulo_str = df[coluna_rotulo].astype(str).str.strip()
    mascara_valida = rotulo_str.isin(mapeamento_classes)
    if not mascara_valida.all():
        invalidos = rotulo_str[~mascara_valida].unique()
        logger.warning(
            "%d amostras descartadas por rótulo desconhecido: %s",
            (~mascara_valida).sum(), invalidos,
        )
        df = df[mascara_valida].reset_index(drop=True)
        rotulo_str = rotulo_str[mascara_valida].reset_index(drop=True)

    rotulos = rotulo_str.map(mapeamento_classes).astype(int)

    logger.info(
        "Dataset carregado de '%s': %d amostras (classe 0: %d, classe 1: %d)",
        os.path.basename(caminho), len(df),
        (rotulos == 0).sum(), (rotulos == 1).sum(),
    )
    return df[coluna_texto].reset_index(drop=True), rotulos.reset_index(drop=True)


def _inferir_mapeamento(valores: np.ndarray) -> dict:
    """
    Tenta inferir o mapeamento binário automaticamente.
    Reconhece padrões comuns em inglês, português e espanhol.
    Se não reconhecer, mapeia alfabeticamente: menor valor = 0, maior = 1.
    """
    valores_lower = {str(v).lower().strip() for v in valores}

    padroes_conhecidos = [
        {"positive", "negative"},
        {"positivo", "negativo"},
        {"pos", "neg"},
        {"bom", "ruim"},
        {"good", "bad"},
        {"1", "0"},
        {"true", "false"},
        {"yes", "no"},
        {"sim", "não"},
        {"happy", "sad"},
    ]
    positivos_por_padrao = {
        frozenset({"positive", "negative"}): "positive",
        frozenset({"positivo", "negativo"}): "positivo",
        frozenset({"pos", "neg"}): "pos",
        frozenset({"bom", "ruim"}): "bom",
        frozenset({"good", "bad"}): "good",
        frozenset({"1", "0"}): "1",
        frozenset({"true", "false"}): "true",
        frozenset({"yes", "no"}): "yes",
        frozenset({"sim", "não"}): "sim",
        frozenset({"happy", "sad"}): "happy",
    }

    for padrao in padroes_conhecidos:
        if valores_lower == padrao:
            chave = frozenset(padrao)
            positivo = positivos_por_padrao[chave]
            negativo = (padrao - {positivo}).pop()
            return {positivo: 1, negativo: 0}

    # Fallback: ordena alfabeticamente e mapeia
    ordenados = sorted(str(v) for v in valores)
    if len(ordenados) == 2:
        logger.warning(
            "Mapeamento automático por ordem alfabética: '%s'=0, '%s'=1. "
            "Para garantir a ordem correta, forneça 'mapeamento_classes' explicitamente.",
            ordenados[0], ordenados[1],
        )
        return {ordenados[0]: 0, ordenados[1]: 1}

    raise ValueError(
        f"Não foi possível inferir mapeamento binário para os valores: {sorted(valores)}. "
        "Forneça 'mapeamento_classes' explicitamente, ex: {'positivo': 1, 'negativo': 0}."
    )


def carregar_imdb(caminho: str) -> tuple:
    """
    Atalho para carregar o dataset IMDB com as colunas padrão.
    Mantido por compatibilidade com scripts existentes.
    """
    return carregar_dataset(
        caminho,
        coluna_texto="review",
        coluna_rotulo="sentiment",
        mapeamento_classes={"positive": 1, "negative": 0},
    )


def dividir_dataset(textos, rotulos, treino=0.70, val=0.15, seed=42) -> tuple:
    """
    Divide os dados em treino / validação / teste com estratificação.
    O conjunto de teste nunca deve ser consultado durante desenvolvimento.

    Retorna
        X_treino, X_val, X_teste, y_treino, y_val, y_teste (todos como listas Python)
    """
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        textos, rotulos,
        test_size=1 - treino,
        stratify=rotulos,
        random_state=seed,
    )
    val_rel = val / (val + (1 - treino - val))
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - val_rel,
        stratify=y_tmp,
        random_state=seed,
    )
    logger.info(
        "Divisão concluída: treino=%d val=%d teste=%d",
        len(X_tr), len(X_val), len(X_te),
    )
    return list(X_tr), list(X_val), list(X_te), list(y_tr), list(y_val), list(y_te)


def avaliar(y_true, y_pred, y_proba=None, nomes_classes=None) -> dict:
    """
    Calcula acurácia, F1-macro, F1-ponderado, AUC-ROC e relatório completo.

    Parâmetros
        y_true        rótulos verdadeiros.
        y_pred        rótulos preditos.
        y_proba       probabilidades da classe positiva (opcional, para AUC-ROC).
        nomes_classes lista com nomes das classes na ordem [classe_0, classe_1].
                      Padrão ["negativo", "positivo"].
    """
    if nomes_classes is None:
        nomes_classes = ["negativo", "positivo"]

    metricas = {
        "acuracia":     round(accuracy_score(y_true, y_pred), 4),
        "f1_macro":     round(f1_score(y_true, y_pred, average="macro"), 4),
        "f1_ponderado": round(f1_score(y_true, y_pred, average="weighted"), 4),
        "relatorio":    classification_report(
            y_true, y_pred, target_names=nomes_classes, zero_division=0
        ),
    }
    if y_proba is not None:
        try:
            metricas["auc_roc"] = round(roc_auc_score(y_true, y_proba), 4)
        except Exception:
            metricas["auc_roc"] = None
    return metricas


def formatar_resultado(m: dict) -> str:
    linhas = [
        f"  Acuracia     : {m.get('acuracia', '-')}",
        f"  F1-macro     : {m.get('f1_macro', '-')}",
        f"  F1-ponderado : {m.get('f1_ponderado', '-')}",
    ]
    if m.get("auc_roc"):
        linhas.append(f"  AUC-ROC      : {m['auc_roc']}")
    linhas.append("")
    linhas.append(m.get("relatorio", ""))
    return "\n".join(linhas)


def salvar_pipeline(pipeline, caminho: str):
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(pipeline, f)
    logger.info("Pipeline salvo em '%s'.", caminho)


def carregar_pipeline(caminho: str):
    with open(caminho, "rb") as f:
        return pickle.load(f)


def gerar_relatorio(resultados: list, caminho_csv=None) -> pd.DataFrame:
    """Gera DataFrame comparativo dos experimentos, ordenado por F1-macro na validação."""
    linhas = [{
        "experimento":    r["nome"],
        "vetorizador":    r["config"]["vectorizer"].get("strategy", "-"),
        "classificador":  r["config"]["classifier"].get("model", "-"),
        "f1_macro_val":   r["metricas_val"].get("f1_macro"),
        "acuracia_val":   r["metricas_val"].get("acuracia"),
        "f1_macro_teste": r["metricas_teste"].get("f1_macro"),
        "acuracia_teste": r["metricas_teste"].get("acuracia"),
        "auc_roc_teste":  r["metricas_teste"].get("auc_roc"),
    } for r in resultados]
    df = pd.DataFrame(linhas).sort_values("f1_macro_val", ascending=False)
    if caminho_csv:
        os.makedirs(os.path.dirname(caminho_csv) or ".", exist_ok=True)
        df.to_csv(caminho_csv, index=False)
        logger.info("Relatório salvo em '%s'.", caminho_csv)
    return df


def configurar_logging(nivel: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, nivel.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

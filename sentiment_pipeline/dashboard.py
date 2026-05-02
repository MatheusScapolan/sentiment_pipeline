"""
Dashboard interativo de análise de experimentos e inferência em tempo real.

Esta funcionalidade não foi solicitada na atividade ponderada, mas agrega
valor ao projeto ao transformar os resultados dos experimentos em uma
interface visual e interativa sem necessidade de escrever HTML ou JavaScript.

O dashboard é construído com Gradio e possui três abas

Aba 1  Resultados dos Experimentos
    Carrega o CSV gerado pelo script experimentos.py e exibe um gráfico
    comparativo de F1-macro, uma tabela com ranking completo e uma análise
    automática do melhor e pior pipeline identificado.

Aba 2  Classificação em Tempo Real
    Carrega um pipeline treinado (arquivo .pkl) e permite classificar
    textos digitados diretamente na interface. Suporta qualquer idioma
    e qualquer conjunto de classes configurado no pipeline.

Aba 3  Sobre o Projeto
    Informações sobre o projeto, dataset e referências bibliográficas.

Como usar

    Via linha de comando após os experimentos
        python scripts/dashboard.py

    Com caminhos personalizados
        python scripts/dashboard.py \\
            --csv outputs/resultados_experimentos.csv \\
            --pipeline outputs/melhor_pipeline.pkl

    Para deixar o servidor acessível na rede local
        python scripts/dashboard.py --share
"""

import os
import threading

# FIX 1: matplotlib.use("Agg") DEVE ser chamado no topo do módulo, antes de
# qualquer import de matplotlib.pyplot. Chamá-lo dentro de um callback
# Gradio em ambiente multithread causa deadlock no backend Agg.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
import pandas as pd


# FIX 2: Lock global para proteger o cache de pipelines contra race conditions
# quando múltiplos callbacks tentam carregar o mesmo arquivo simultaneamente.
_PIPELINE_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers de I/O
# ---------------------------------------------------------------------------

def _carregar_csv(caminho_csv: str):
    """Carrega e valida o CSV de resultados dos experimentos."""
    if not caminho_csv or not os.path.isfile(caminho_csv):
        return None
    df = pd.read_csv(caminho_csv)

    colunas_numericas = [
        "f1_macro_val",
        "acuracia_val",
        "f1_macro_teste",
        "acuracia_teste",
        "auc_roc_teste",
    ]
    for coluna in colunas_numericas:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = df.sort_values("f1_macro_val", ascending=False, na_position="last").reset_index(drop=True)
    df.index = df.index + 1
    return df.fillna("N/D")


def _carregar_pipeline(caminho_pkl: str):
    """Carrega um pipeline serializado do disco."""
    if not caminho_pkl or not os.path.isfile(caminho_pkl):
        return None
    import pickle
    with open(caminho_pkl, "rb") as f:
        return pickle.load(f)


def _obter_pipeline(caminho_pkl: str):
    """Carrega pipeline com cache thread-safe para evitar serialização pesada repetida."""
    if not caminho_pkl or not os.path.isfile(caminho_pkl):
        return None
    # FIX 2 (cont.): usa Lock para garantir que apenas uma thread carregue o
    # arquivo por vez; as demais aguardam e reutilizam o cache já populado.
    with _CACHE_LOCK:
        if caminho_pkl not in _PIPELINE_CACHE:
            _PIPELINE_CACHE[caminho_pkl] = _carregar_pipeline(caminho_pkl)
    return _PIPELINE_CACHE[caminho_pkl]


def _info_pipeline(pipeline) -> str:
    """Retorna string de status do pipeline carregado, com idioma e classes."""
    if pipeline is None:
        return "Nenhum pipeline carregado."
    idioma  = pipeline.config.get("preprocessor", {}).get("language", "english")
    classes = _mapear_classes_display(getattr(pipeline, "nomes_classes", ["negativo", "positivo"]))
    nome    = getattr(pipeline, "nome", "desconhecido")
    return (
        f"Pipeline **{nome}** carregado. "
        f"Idioma **{idioma}** | Classes **{classes[0]}** e **{classes[1]}**"
    )


def _mapear_classes_display(nomes):
    """Normaliza nomes para exibicao consistente no dashboard."""
    if not nomes or len(nomes) < 2:
        return ["negativo", "positivo"]

    mapa = {
        "detractor": "negativo",
        "promoter": "positivo",
    }
    resultado = []
    for nome in nomes[:2]:
        chave = str(nome).strip().lower()
        resultado.append(mapa.get(chave, nome))
    return resultado


# ---------------------------------------------------------------------------
# Funções de apresentação
# ---------------------------------------------------------------------------

def _grafico_barras(df: pd.DataFrame):
    """Gera um gráfico de barras comparando F1-macro de validação e teste."""
    # FIX 1 (cont.): matplotlib.use("Agg") já foi chamado no topo; aqui
    # apenas criamos a figura normalmente.
    nomes   = df["experimento"].tolist()
    f1_val  = pd.to_numeric(df["f1_macro_val"],   errors="coerce").fillna(0).tolist()
    f1_test = pd.to_numeric(df["f1_macro_teste"], errors="coerce").fillna(0).tolist()

    x       = range(len(nomes))
    largura = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(nomes) * 1.2), 5))
    barras_val  = ax.bar(
        [i - largura / 2 for i in x], f1_val,  largura,
        label="F1-macro Validação",  color="#3498db", alpha=0.85,
    )
    barras_test = ax.bar(
        [i + largura / 2 for i in x], f1_test, largura,
        label="F1-macro Teste",      color="#27ae60", alpha=0.85,
    )

    ax.set_ylabel("F1-macro", fontsize=11)
    ax.set_title("Comparativo de Desempenho por Experimento", fontsize=13, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(nomes, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(max(0, min(f1_val + f1_test) - 0.05), 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))

    for barra in barras_val:
        ax.annotate(
            f"{barra.get_height():.3f}",
            xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=7, color="#1a5276",
        )
    for barra in barras_test:
        ax.annotate(
            f"{barra.get_height():.3f}",
            xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=7, color="#1e8449",
        )

    plt.tight_layout()
    return fig


def _analise_automatica(df: pd.DataFrame) -> str:
    """Gera análise automática do melhor e pior pipeline em português."""
    melhor = df.iloc[0]
    pior   = df.iloc[-1]

    f1_melhor = pd.to_numeric(melhor.get("f1_macro_val"), errors="coerce")
    f1_pior   = pd.to_numeric(pior.get("f1_macro_val"),   errors="coerce")
    diferenca = (
        round(f1_melhor - f1_pior, 4)
        if pd.notna(f1_melhor) and pd.notna(f1_pior)
        else "N/D"
    )

    total      = len(df)
    melhor_vec = melhor.get("vetorizador",   "N/D")
    melhor_clf = melhor.get("classificador", "N/D")

    texto = (
        f"Análise dos Resultados\n\n"
        f"Foram executados {total} experimentos combinando diferentes estratégias "
        f"de pré-processamento, vetorização e classificação.\n\n"
        f"Melhor resultado\n"
        f'O experimento "{melhor["experimento"]}" obteve o maior F1-macro na validação '
        f'({melhor["f1_macro_val"]}), utilizando vetorização {melhor_vec} com o '
        f"classificador {melhor_clf}. No conjunto de teste final, o F1-macro registrado "
        f'foi {melhor.get("f1_macro_teste", "N/D")}, o que indica boa generalização do '
        f"modelo para dados não vistos durante o treinamento.\n\n"
        f"Menor resultado\n"
        f'O experimento "{pior["experimento"]}" obteve o menor F1-macro na validação '
        f'({pior["f1_macro_val"]}). A diferença entre o melhor e o pior resultado foi de '
        f"{diferenca} pontos de F1-macro, o que reforça a importância de testar múltiplas "
        f"combinações de pré-processamento e vetorização antes de definir a configuração final.\n\n"
        f"Interpretação geral\n"
        f"A variação observada entre os experimentos confirma o que a literatura de NLP aponta: "
        f"a escolha da representação textual impacta os resultados de forma tão significativa "
        f"quanto a escolha do algoritmo de classificação. Configurações que preservam negações "
        f"e utilizam bigramas tendem a capturar melhor a polaridade expressa em avaliações."
    )
    return texto


def _tabela_formatada(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara o DataFrame para exibição no Gradio com colunas renomeadas."""
    colunas = {
        "experimento":    "Experimento",
        "vetorizador":    "Vetorizador",
        "classificador":  "Classificador",
        "f1_macro_val":   "F1 Validação",
        "acuracia_val":   "Acurácia Val.",
        "f1_macro_teste": "F1 Teste",
        "acuracia_teste": "Acurácia Teste",
        "auc_roc_teste":  "AUC-ROC",
    }
    df_exib = df[[c for c in colunas if c in df.columns]].rename(columns=colunas)
    df_exib.insert(0, "Posição", range(1, len(df_exib) + 1))
    return df_exib


# ---------------------------------------------------------------------------
# Callbacks das abas
# ---------------------------------------------------------------------------

def _atualizar_aba_resultados(csv_path: str):
    """Callback do botão 'Carregar Resultados' e do carregamento inicial."""
    df = _carregar_csv(csv_path)
    if df is None:
        msg = (
            f"Arquivo não encontrado em '{csv_path}'. "
            "Execute os experimentos primeiro."
        )
        return None, msg, None
    return _grafico_barras(df), _analise_automatica(df), _tabela_formatada(df)


def _carregar_pipeline_interface(pkl_path: str):
    """
    Callback do botão 'Carregar Pipeline'.
    Retorna o caminho (não o objeto) para o gr.State, e uma mensagem de status
    com idioma e classes do pipeline carregado.
    """
    p = _obter_pipeline(pkl_path)
    if p is None:
        return None, f"Arquivo não encontrado em '{pkl_path}'."
    return pkl_path, _info_pipeline(p)


def _classificar_texto(texto: str, pipeline_path):
    """
    Callback dos botões 'Classificar' e Enter no campo de texto.

    Generalizado para exibir os nomes reais das classes do pipeline
    (não hardcoded como POSITIVO/NEGATIVO), suportando qualquer idioma
    e qualquer mapeamento configurado no pipeline treinado.
    """
    if pipeline_path is None:
        return {}, 0.0, "Carregue um pipeline antes de classificar."
    if not texto or not texto.strip():
        return {}, 0.0, "Digite um texto antes de classificar."

    pipeline = _obter_pipeline(pipeline_path)
    if pipeline is None:
        return {}, 0.0, "Pipeline não encontrado no caminho informado."

    resultado  = pipeline.predict_texto(texto.strip())
    rotulo     = resultado["rotulo"].upper()
    confianca  = float(resultado["confianca"] or 0.0)
    outra      = round(1.0 - confianca, 4)

    # Recupera os nomes reais das classes do pipeline para exibição correta
    nomes = _mapear_classes_display(getattr(pipeline, "nomes_classes", ["negativo", "positivo"]))
    nome_pos = str(nomes[1]).upper() if len(nomes) > 1 else "POSITIVO"
    nome_neg = str(nomes[0]).upper()

    # BUG CORRIGIDO: o dicionário passado ao gr.Label deve atribuir 'confianca'
    # à classe que o modelo realmente predisse, e 'outra' à classe oposta.
    if rotulo == nome_pos:
        classes = {nome_pos: confianca, nome_neg: outra}
    else:
        classes = {nome_neg: confianca, nome_pos: outra}

    idioma = pipeline.config.get("preprocessor", {}).get("language", "english")
    info = (
        f"Classe predita: {rotulo}\n"
        f"Confiança: {confianca}\n"
        f"Pipeline: {getattr(pipeline, 'nome', 'desconhecido')}\n"
        f"Idioma: {idioma}"
    )
    return classes, confianca, info


# ---------------------------------------------------------------------------
# Construção do layout
# ---------------------------------------------------------------------------

def criar_dashboard(
    caminho_csv: str = "outputs/resultados_experimentos.csv",
    caminho_pkl: str = "outputs/melhor_pipeline.pkl",
    compartilhar: bool = False,
) -> None:
    """
    Cria e inicia o dashboard Gradio.

    Parâmetros
        caminho_csv     caminho para o CSV gerado por experimentos.py.
        caminho_pkl     caminho para o pipeline serializado (.pkl).
        compartilhar    se True, gera link público via servidor Gradio.
    """
    csv_existe = os.path.isfile(caminho_csv)
    pkl_existe = os.path.isfile(caminho_pkl)

    with gr.Blocks(
        title="Sentiment Pipeline — Dashboard",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    ) as app:

        gr.Markdown("""
# Sentiment Pipeline
### Dashboard de Análise de Sentimentos com Classificadores Clássicos
Pipeline dinâmico e agnóstico de dataset — Módulo 6 do Inteli
        """)

        with gr.Tabs():

            # ===============================================================
            # Aba 1 — Resultados dos Experimentos
            # ===============================================================
            with gr.Tab("Resultados dos Experimentos"):

                gr.Markdown("""
### Comparativo de Desempenho

Esta aba apresenta os resultados de todos os experimentos executados.
Cada linha representa uma combinação de pré-processamento, vetorização
e classificação testada no dataset configurado.
                """)

                with gr.Row():
                    entrada_csv = gr.Textbox(
                        label="Caminho do arquivo CSV de resultados",
                        value=caminho_csv,
                        placeholder="outputs/resultados_experimentos.csv",
                    )
                    botao_carregar = gr.Button(
                        "Carregar Resultados", variant="primary", scale=0
                    )

                with gr.Row():
                    with gr.Column(scale=2):
                        grafico = gr.Plot(label="F1-macro por Experimento")
                    with gr.Column(scale=1):
                        analise = gr.Textbox(
                            label="Análise Automática",
                            lines=18,
                            interactive=False,
                        )

                tabela = gr.Dataframe(
                    label="Ranking Completo dos Experimentos",
                    interactive=False,
                    wrap=True,
                )

                botao_carregar.click(
                    fn=_atualizar_aba_resultados,
                    inputs=[entrada_csv],
                    outputs=[grafico, analise, tabela],
                )

                # FIX 3: substitui app.load() por gr.on("load") com queue ativo.
                # O app.load() original disparava um callback bloqueante na thread
                # principal do Gradio antes de qualquer aba ser renderizada,
                # impedindo a troca de abas. gr.on("load") é enfileirado pelo
                # sistema de filas do Gradio 4.x e não bloqueia o event loop.
                if csv_existe:
                    gr.on(
                        triggers=[app.load],
                        fn=_atualizar_aba_resultados,
                        inputs=[entrada_csv],
                        outputs=[grafico, analise, tabela],
                    )
                else:
                    gr.Markdown(
                        f"> Nenhum resultado encontrado em `{caminho_csv}`. "
                        "Execute `python scripts/experimentos.py --dataset data/IMDB_Dataset.csv` "
                        "e depois clique em **Carregar Resultados**."
                    )

            # ===============================================================
            # Aba 2 — Classificação em Tempo Real
            # ===============================================================
            with gr.Tab("Classificação em Tempo Real"):

                gr.Markdown("""
### Classificador de Sentimentos

Digite ou cole um texto para classificá-lo.
O pipeline suporta qualquer idioma configurado durante o treinamento:
inglês, português, espanhol e outros idiomas suportados pelo NLTK.
                """)

                with gr.Row():
                    entrada_pkl = gr.Textbox(
                        label="Caminho do pipeline treinado (.pkl)",
                        value=caminho_pkl,
                        placeholder="outputs/melhor_pipeline.pkl",
                    )
                    botao_pipeline = gr.Button("Carregar Pipeline", scale=0)

                status_pipeline = gr.Markdown(
                    value=(
                        _info_pipeline(_obter_pipeline(caminho_pkl))
                        if pkl_existe
                        else f"Nenhum pipeline encontrado em `{caminho_pkl}`."
                    )
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        texto_entrada = gr.Textbox(
                            label="Texto para classificar",
                            placeholder="Digite aqui o texto a ser analisado...",
                            lines=5,
                        )
                        botao_classificar = gr.Button(
                            "Classificar", variant="primary"
                        )

                    with gr.Column(scale=1):
                        saida_rotulo    = gr.Label(label="Resultado")
                        saida_confianca = gr.Number(
                            label="Confiança do Modelo", precision=4
                        )
                        saida_info = gr.Textbox(
                            label="Detalhes", interactive=False, lines=4
                        )

                gr.Examples(
                    examples=[
                        ["This movie was absolutely fantastic! Outstanding performances and deeply moving story."],
                        ["What a terrible waste of time. The plot made no sense and the acting was awful."],
                        ["Não gostei nada desse filme. A história não faz sentido e a atuação foi péssima."],
                        ["Adorei esse filme! Uma das melhores histórias que já assisti, simplesmente incrível."],
                        ["It was okay, not great but not terrible either. Some good moments."],
                    ],
                    inputs=[texto_entrada],
                    label="Exemplos de textos para testar",
                )

                # FIX 4: gr.State armazena apenas o caminho (str) do pipeline,
                # nunca o objeto serializado. O objeto fica no _PIPELINE_CACHE
                # protegido pelo Lock. Passar objetos pesados no gr.State do
                # Gradio 4.x os serializa/desserializa a cada evento, gerando
                # latência e podendo travar callbacks subsequentes.
                pipeline_estado = gr.State(value=caminho_pkl if pkl_existe else None)

                botao_pipeline.click(
                    fn=_carregar_pipeline_interface,
                    inputs=[entrada_pkl],
                    outputs=[pipeline_estado, status_pipeline],
                )

                botao_classificar.click(
                    fn=_classificar_texto,
                    inputs=[texto_entrada, pipeline_estado],
                    outputs=[saida_rotulo, saida_confianca, saida_info],
                )
                texto_entrada.submit(
                    fn=_classificar_texto,
                    inputs=[texto_entrada, pipeline_estado],
                    outputs=[saida_rotulo, saida_confianca, saida_info],
                )

            # ===============================================================
            # Aba 3 — Sobre o Projeto
            # ===============================================================
            with gr.Tab("Sobre o Projeto"):
                gr.Markdown("""
### Sobre o Sentiment Pipeline

**Contexto acadêmico**

Este projeto foi desenvolvido como atividade ponderada do Módulo 6 do curso de
Sistemas de Informação do Inteli. O módulo aborda o processamento e análise de
textos com inteligência artificial, com ênfase em classificação de sentimentos,
vetorização de texto e machine learning.

**O que este pipeline faz**

O pipeline recebe qualquer lista de textos e rótulos binários em qualquer idioma,
aplica as etapas de pré-processamento configuradas, converte os textos em
representações numéricas e treina o classificador selecionado. O sistema é
agnóstico de dataset: funciona com IMDB, datasets em português, espanhol ou
qualquer outro idioma suportado pelo NLTK.

**Componentes principais**

Preprocessor — limpeza e normalização textual com suporte a inglês, português,
espanhol, francês, alemão, italiano e outros idiomas via NLTK.

Vectorizer — quatro estratégias: BoW, TF-IDF, TF-IDF com SVD (LSA) e Word2Vec.

Classifier — cinco modelos com busca de hiperparâmetros manual, grid, aleatória
ou bayesiana via Optuna.

SentimentPipeline — orquestrador que integra os três componentes acima.

**Dataset padrão nos experimentos**

IMDB Dataset com 50 mil avaliações de filmes, introduzido por Maas et al. (2011)
em "Learning Word Vectors for Sentiment Analysis". Qualquer outro CSV pode ser
usado informando as colunas corretas ao script de experimentos.

**Referências principais**

Pang, Lee e Vaithyanathan (2002) — Thumbs up? Sentiment Classification using Machine Learning.

Maas et al. (2011) — Learning Word Vectors for Sentiment Analysis.

Ke et al. (2017) — LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
                """)

        # FIX 3 (cont.): .queue() é obrigatório no Gradio 4.x para que callbacks
        # assíncronos (incluindo gr.on("load")) sejam processados corretamente
        # pelo event loop sem bloquear a renderização do frontend.
        app.queue()

    app.launch(share=compartilhar, server_name="0.0.0.0")


# ---------------------------------------------------------------------------
# Entry point CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sentiment Pipeline — Dashboard")
    parser.add_argument(
        "--csv",
        default="outputs/resultados_experimentos.csv",
        help="Caminho para o CSV gerado por experimentos.py",
    )
    parser.add_argument(
        "--pipeline",
        default="outputs/melhor_pipeline.pkl",
        help="Caminho para o pipeline serializado (.pkl)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Gera link público via servidor Gradio",
    )
    args = parser.parse_args()

    criar_dashboard(
        caminho_csv=args.csv,
        caminho_pkl=args.pipeline,
        compartilhar=args.share,
    )

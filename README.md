# Pipeline Dinâmico para Classificadores Clássicos de Sentimentos

Atividade Ponderada do Módulo 6 — Sistemas de Informação — Inteli

---

## Por que esta atividade existe

O Módulo 6 do Inteli é dedicado à construção de sistemas de análise de textos com inteligência artificial. Um dos fundamentos centrais do módulo é a percepção de que a escolha de como preparar e representar o texto numericamente frequentemente determina mais o resultado do modelo do que a troca do algoritmo de classificação.

Pang, Lee e Vaithyanathan (2002), em "Thumbs up? Sentiment Classification using Machine Learning Techniques", demonstraram que variações aparentemente simples na representação do texto, como usar presença binária de palavras em vez de frequência absoluta, produzem diferenças expressivas de desempenho mesmo quando o classificador utilizado é exatamente o mesmo. Maas et al. (2011), ao introduzir o dataset IMDB com 50 mil avaliações que é utilizado por padrão neste projeto, reforçaram esse argumento ao mostrar que a qualidade das representações vetoriais é o componente mais determinante em tarefas de análise de sentimentos.

---

## Como esta atividade se encaixa no módulo

O módulo organiza o aprendizado em quatro eixos: matemática, computação, UX e negócios.

Do ponto de vista matemático, o pipeline operacionaliza vetores e matrizes nas representações textuais, probabilidade e Teorema de Bayes no Naive Bayes, gradiente descendente na Regressão Logística com otimização via LBFGS ou SAGA, e redução dimensional via decomposição SVD na Análise Semântica Latente.

Do ponto de vista computacional, o projeto aplica machine learning clássico com Scikit-learn, com ênfase em engenharia de software: código modular, testado com pytest e documentado para ser reutilizável em outros contextos.

Do ponto de vista de UX e visualização de dados, o projeto entrega um dashboard interativo construído com Gradio que apresenta gráficos comparativos dos experimentos e um classificador em tempo real acessível diretamente no navegador.

---

## O que foi construído

O projeto entrega um pacote Python instalável chamado `sentiment_pipeline`, composto por quatro componentes principais.

O `Preprocessor` aplica etapas configuráveis de limpeza e normalização textual e suporta múltiplos idiomas. O `Vectorizer` converte textos processados em representações numéricas com quatro estratégias distintas. O `Classifier` treina um classificador clássico com suporte a busca de hiperparâmetros em quatro modalidades. O `SentimentPipeline` integra os três componentes anteriores em um único objeto com interface `fit`, `predict` e `evaluate`.

O pipeline é **agnóstico de dataset e idioma**: aceita qualquer CSV informando as colunas de texto e rótulo, suporta inglês, português, espanhol, francês, alemão, italiano e outros idiomas do NLTK, e permite configurar os nomes das classes de acordo com o dataset.

---

## Estrutura do projeto

```
sentiment_pipeline/
|
|-- sentiment_pipeline/              Pacote Python principal
|   |-- __init__.py
|   |-- pipeline.py                  SentimentPipeline
|   |-- dashboard.py                 Dashboard interativo com Gradio
|   |-- preprocessor/
|   |   |-- preprocessor.py          Preprocessor com suporte multilíngue
|   |-- vectorizer/
|   |   |-- vectorizer.py            Vectorizer com 4 estratégias
|   |-- classifier/
|   |   |-- classifier.py            Classifier com 5 modelos e 4 buscas
|   |-- utils/
|       |-- utils.py                 Carregamento agnóstico, avaliação, serialização
|
|-- scripts/
|   |-- experimentos.py              Grade com os 10 experimentos
|   |-- dashboard.py                 Ponto de entrada do dashboard
|   |-- inferir.py                   Inferência com pipeline salvo
|   |-- exemplo_uso.py               Demonstração em inglês e português
|
|-- tests/
|   |-- test_pipeline.py             Testes unitários e de integração
|   |-- conftest.py
|
|-- data/                            Coloque aqui o CSV do dataset
|-- outputs/                         Gerado automaticamente pelos scripts
|-- requirements.txt
|-- setup.py
|-- README.md
```

---

## Tecnologias utilizadas

**Scikit-learn** é a base de todo o pipeline. Fornece os vetorizadores `CountVectorizer` e `TfidfVectorizer`, o `TruncatedSVD` para redução dimensional via LSA, todos os classificadores clássicos, e as ferramentas de busca `GridSearchCV` e `RandomizedSearchCV`. A biblioteca foi escolhida por sua maturidade e pela interface unificada que torna simples trocar qualquer componente sem alterar o restante do código.

**NLTK** fornece os recursos linguísticos para pré-processamento em múltiplos idiomas: tokenização com suporte a idioma via `word_tokenize`, listas de stopwords para inglês, português, espanhol e outros idiomas, o stemmer `PorterStemmer` para inglês, o `SnowballStemmer` para inglês e outros oito idiomas, e o `WordNetLemmatizer` para inglês. A distinção entre stemming e lematização é relevante. O stemmer aplica regras heurísticas e é rápido, mas pode produzir raízes que não correspondem a palavras reais. O lematizador usa um dicionário morfológico e produz a forma de base correta, sendo preferível quando a interpretabilidade das features importa.

**LightGBM** é incluído como alternativa eficiente ao Random Forest para representações de alta dimensionalidade. Ke et al. (2017) demonstraram que o modelo reduz o custo computacional via amostragem baseada em gradiente e agrupamento exclusivo de features, sendo adequado para TF-IDF com dezenas de milhares de dimensões.

**Gensim** é utilizado opcionalmente para treinar embeddings Word2Vec no próprio corpus. Mikolov et al. (2013) demonstraram que representações distribuídas capturam relações semânticas que as representações esparsas de BoW e TF-IDF não conseguem. No pipeline, cada texto é representado pela média dos vetores de seus tokens.

**Gradio** é a biblioteca do dashboard interativo. Permite criar interfaces web sem HTML, CSS ou JavaScript. Foi escolhido por estar alinhado ao eixo de visualização de dados do módulo, por ser amplamente utilizado na comunidade de ML para demonstração de modelos, e por suportar execução local sem infraestrutura de servidor.

**Matplotlib** é usado internamente pelo dashboard para gerar o gráfico comparativo de F1-macro. A escolha mantém as dependências ao mínimo necessário sem adicionar bibliotecas de visualização JavaScript.

**Optuna** é oferecido como dependência opcional para busca bayesiana de hiperparâmetros via o algoritmo TPE (Tree-structured Parzen Estimator), que aprende quais regiões do espaço de parâmetros são mais promissoras a partir dos trials anteriores.

---

## Escolhas técnicas detalhadas

### Pré-processamento multilíngue

O `Preprocessor` foi projetado para funcionar com qualquer idioma suportado pelo NLTK. O parâmetro `language` controla três aspectos simultaneamente: a tokenização via `word_tokenize`, as stopwords carregadas do corpus do NLTK, e o stemmer instanciado via `SnowballStemmer`.

Para inglês, a lematização via `WordNetLemmatizer` está disponível como alternativa ao stemming. Para os demais idiomas, o sistema usa `SnowballStemmer` automaticamente quando `normalization="lemmatization"` é configurado, e registra um aviso no log explicando a substituição. Isso evita erros silenciosos ao mudar de idioma.

As palavras de negação são definidas por idioma no dicionário `NEGATION_WORDS_POR_IDIOMA`. Para português, o conjunto inclui "não", "nunca", "jamais", "nem", "nada", "ninguém", "tampouco" e "sequer". Para inglês, inclui "not", "never", "nor", "neither" e as formas contraídas como "can't", "won't" e "didn't". O usuário pode sobrescrever esse conjunto passando uma lista customizada via `negation_words` na configuração.

A preservação de negações com `keep_negations` é particularmente importante. Pang et al. (2002) demonstraram que remover palavras como "not" e "never" piora o desempenho porque invertem a polaridade das expressões ao redor. A expressão "not good" tem polaridade oposta a "good", mas um modelo que elimina "not" trata as duas como equivalentes.

O tratamento ativo de negações com `handle_negations` marca os tokens que seguem uma negação com o sufixo `_NEG`. Com isso, o vetorizador trata "good" e "good_NEG" como features completamente distintas. A janela de negação é limitada a cinco tokens e encerra em pontuação, evitando propagação indefinida além do escopo gramatical.

O módulo `_normalizar_emojis` preserva caracteres acentuados ao verificar a categoria Unicode de cada caractere: apenas os da categoria `So` (símbolos), `Cs` (substitutos) e `Co` (uso privado) são substituídos por palavras. Caracteres acentuados comuns em português e espanhol, que ficam acima de `ord > 127` mas pertencem à categoria `Ll` (letra minúscula) ou `Lu` (letra maiúscula), são preservados intactos.

### Carregamento agnóstico de dataset

A função `carregar_dataset` em `utils.py` aceita qualquer CSV com configuração explícita de colunas, mapeamento de classes e codificação. Ela inclui inferência automática do mapeamento quando não fornecido, reconhecendo padrões comuns como `{"positive", "negative"}`, `{"positivo", "negativo"}`, `{"1", "0"}`, `{"bom", "ruim"}` e outros. Quando o padrão não é reconhecido, o mapeamento é feito por ordem alfabética com aviso no log.

Para lidar com CSVs reais e inconsistentes, o carregamento usa o parser Python do pandas e ignora linhas mal formatadas. Isso torna a leitura mais robusta, mesmo que algumas linhas sejam descartadas.

Quando `coluna_rotulo` é `"label"` e a coluna `overall_rating` existe, o pipeline cria automaticamente o rótulo binário: notas 4 e 5 viram 1 (positivo), notas 1 e 2 viram 0 (negativo) e notas 3 são removidas.

A função `carregar_imdb` é mantida como atalho de compatibilidade, delegando para `carregar_dataset` com os parâmetros padrão do IMDB.

### Avaliação com nomes de classe customizados

A função `avaliar` em `utils.py` aceita o parâmetro `nomes_classes` para usar os nomes corretos no relatório de classificação. O `SentimentPipeline` armazena `nomes_classes` como atributo e o repassa para `avaliar` em todos os pontos de avaliação. O método `predict_texto` usa `nomes_classes` para retornar o rótulo legível correspondente à classe predita.

### Vetorização

O **Bag of Words** é eficaz para textos longos onde a presença de certas palavras já é altamente informativa. Com `binary=True`, indica presença ou ausência do termo, adequado para `BernoulliNB`.

O **TF-IDF** com `sublinear_tf=True` aplica `1 + log(tf)` no lugar da frequência bruta, reduzindo o peso de tokens muito frequentes. A configuração `ngram_range=(1, 2)` captura expressões como "not bad" ou "nunca mais" que o unigrama isolado perderia.

O **TF-IDF com SVD** implementa a Análise Semântica Latente. Deerwester et al. (1990) demonstraram que a decomposição SVD da matriz termo-documento captura associações semânticas entre palavras que não co-ocorrem diretamente. O pipeline testa 100 e 300 componentes.

O **Word2Vec** representa cada texto como a média dos vetores de seus tokens, produzindo uma representação densa adequada para classificadores que se beneficiam de features contínuas.

### Classificadores e seleção automática do Naive Bayes

O `Classifier` seleciona automaticamente a variante correta do Naive Bayes de acordo com a estratégia de vetorização. O `MultinomialNB` é usado para BoW e TF-IDF, o `BernoulliNB` para BoW binário, e o `GaussianNB` para representações densas como TF-IDF com SVD e Word2Vec.

### Divisão dos dados

O dataset é dividido em 70% treino, 15% validação e 15% teste com estratificação pela classe-alvo, seguindo a prática descrita em Hastie, Tibshirani e Friedman (2009): o conjunto de teste deve permanecer invisível durante o desenvolvimento para que a estimativa de desempenho seja confiável.

---

## Como rodar em menos de 10 minutos

### 1. Instalar

```bash
git clone <url-do-repositorio>
cd sentiment_pipeline
pip install -e .
```

### 2. Validar a instalação (sem dataset externo, menos de 1 minuto)

```bash
python scripts/exemplo_uso.py
```

Demonstra o pipeline em inglês e em português com dados sintéticos.

### 3. Baixar o dataset IMDB

Acesse [kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), baixe `IMDB_Dataset.csv` e coloque em `data/`.

### 4. Rodar os experimentos

```bash
# Teste rápido com 5 mil amostras (3 a 5 minutos)
python scripts/experimentos.py --dataset data/IMDB_Dataset.csv --subset 5000

# Todos os experimentos (30 a 60 minutos com 50 mil amostras)
python scripts/experimentos.py --dataset data/IMDB_Dataset.csv
```

### 5. Rodar os testes

```bash
pytest tests/ -v
pytest tests/ -v --cov=sentiment_pipeline --cov-report=term-missing
```

### 6. Abrir o dashboard

```bash
python scripts/dashboard.py
# Acesse http://localhost:7860
```

---

## Usando com qualquer dataset

O pipeline aceita qualquer CSV binário em qualquer idioma suportado pelo NLTK.

### Dataset em português

```bash
python scripts/experimentos.py \
    --dataset data/meu_dataset_pt.csv \
    --coluna_texto texto \
    --coluna_rotulo sentimento \
    --classe_positiva positivo \
    --classe_negativa negativo \
    --idioma portuguese
```

### Dataset com separador ponto-e-vírgula

```bash
python scripts/experimentos.py \
    --dataset data/dados.csv \
    --coluna_texto comentario \
    --coluna_rotulo classe \
    --classe_positiva bom \
    --classe_negativa ruim \
    --idioma portuguese \
    --separador ";"
```

### Dataset com encoding Latin-1

```bash
python scripts/experimentos.py \
    --dataset data/dados_latin1.csv \
    --coluna_texto comentario \
    --coluna_rotulo classe \
    --classe_positiva bom \
    --classe_negativa ruim \
    --idioma portuguese \
    --separador ";" \
    --encoding latin1
```

### B2W Reviews (label automático)

```bash
python scripts/experimentos.py \
    --dataset data/B2W-Reviews01.csv \
    --coluna_texto review_text \
    --coluna_rotulo label \
    --classe_positiva 1 \
    --classe_negativa 0 \
    --idioma portuguese \
    --separador "," \
    --encoding latin1
```

### Como biblioteca Python

```python
from sentiment_pipeline import SentimentPipeline
from sentiment_pipeline.utils import carregar_dataset, dividir_dataset

# Carregamento agnóstico
textos, rotulos = carregar_dataset(
    "data/meu_dataset.csv",
    coluna_texto="texto",
    coluna_rotulo="sentimento",
    mapeamento_classes={"positivo": 1, "negativo": 0},
)

X_tr, X_val, X_te, y_tr, y_val, y_te = dividir_dataset(textos, rotulos)

# Pipeline em português
config = {
    "preprocessor": {
        "language": "portuguese",
        "lowercase": True,
        "remove_stopwords": "keep_negations",
        "normalization": "stemming",
        "handle_negations": True,
        "remove_urls": True,
        "normalize_emojis": True,
    },
    "vectorizer": {
        "strategy": "tfidf",
        "sublinear_tf": True,
        "ngram_range": (1, 2),
        "max_features": 50000,
        "norm": "l2",
    },
    "classifier": {
        "model": "logistic_regression",
        "search": "random",
        "cv": 3,
        "n_iter": 20,
    },
}

pipeline = SentimentPipeline(
    config,
    nome="pipeline_pt",
    nomes_classes=["negativo", "positivo"],
)
pipeline.fit(X_tr, y_tr)

metricas = pipeline.evaluate(X_val, y_val)
print(metricas["f1_macro"])

resultado = pipeline.predict_texto("Não gostei nada desse filme!")
# {"classe": 0, "rotulo": "negativo", "confianca": 0.8821}
```

### Negações customizadas

```python
pre = Preprocessor({
    "language": "portuguese",
    "lowercase": True,
    "remove_stopwords": "keep_negations",
    "negation_words": ["não", "nunca", "jamais", "sequer", "tampouco"],
    "handle_negations": True,
})
resultado = pre.transform(["Jamais voltaria a esse restaurante"])
```

---

## Referência de configuração

### Preprocessor

| Parâmetro | Valores aceitos | Descrição |
|-----------|----------------|-----------|
| `language` | `"english"`, `"portuguese"`, `"spanish"`, `"french"`, `"german"`, `"italian"` | Idioma para tokenização, stopwords e stemming |
| `lowercase` | `True` ou `False` | Converte para minúsculas |
| `remove_stopwords` | `True`, `False` ou `"keep_negations"` | Remove stopwords; `keep_negations` preserva negações |
| `negation_words` | lista de strings | Sobrescreve as negações padrão do idioma |
| `normalization` | `"stemming"`, `"lemmatization"` (inglês) ou `None` | Redução morfológica |
| `handle_negations` | `True` ou `False` | Marca tokens após negação com `_NEG` |
| `remove_urls` | `True` ou `False` | Remove URLs |
| `normalize_emojis` | `True` ou `False` | Substitui emoticons por palavras, preserva acentos |

### Vectorizer

| Estratégia | Parâmetros principais |
|------------|-----------------------|
| `"bow"` | `binary`, `max_features` |
| `"tfidf"` | `sublinear_tf`, `ngram_range`, `max_features`, `norm`, `min_df` |
| `"tfidf_svd"` | mesmos do tfidf mais `svd_components` (50, 100 ou 300) |
| `"word2vec"` | `glove_path` (ou `None` para treinar no corpus), `embedding_dim` |

### Classifier

| Parâmetro | Valores aceitos |
|-----------|----------------|
| `model` | `"naive_bayes"`, `"logistic_regression"`, `"linear_svc"`, `"random_forest"`, `"lightgbm"` |
| `search` | `"manual"`, `"grid"`, `"random"`, `"optuna"` |
| `params` | dicionário de hiperparâmetros fixos quando `search` for `"manual"` |
| `cv` | número de folds, padrão 3 |
| `n_iter` | iterações para busca aleatória, padrão 20 |
| `n_trials` | trials para Optuna, padrão 30 |

### SentimentPipeline

| Parâmetro | Descrição |
|-----------|-----------|
| `config` | dicionário com chaves `preprocessor`, `vectorizer` e `classifier` |
| `nome` | identificador para logs e relatórios |
| `nomes_classes` | lista `[nome_classe_0, nome_classe_1]`, padrão `["negativo", "positivo"]` |

---

## Experimentos realizados

| # | Nome | Vetorizador | Classificador | Busca |
|---|------|-------------|---------------|-------|
| 1 | bow_naive_bayes | BoW | Naive Bayes | Grid |
| 2 | bow_binario_bernoulli | BoW binário | BernoulliNB | Grid |
| 3 | tfidf_bigrama_logreg_manual | TF-IDF (1,2) | Regressão Logística | Manual |
| 4 | tfidf_bigrama_linearsvc | TF-IDF (1,2) | LinearSVC | Aleatória |
| 5 | tfidf_unigram_logreg_random | TF-IDF (1,1) | Regressão Logística | Aleatória |
| 6 | tfidf_trigrama_linearsvc | TF-IDF (1,3) | LinearSVC | Manual |
| 7 | tfidf_unigram_lightgbm | TF-IDF (1,1) | LightGBM | Aleatória |
| 8 | tfidf_svd100_logreg | TF-IDF + SVD 100d | Regressão Logística | Grid |
| 9 | tfidf_svd300_rf | TF-IDF + SVD 300d | Random Forest | Manual |
| 10 | word2vec_corpus_logreg | Word2Vec | Regressão Logística | Manual |

---

## Funcionalidade extra: Dashboard Gradio

Esta funcionalidade não foi solicitada na atividade ponderada, mas resolve um problema real de usabilidade: após executar dez experimentos, comparar resultados em uma tabela CSV não é intuitivo nem revelador. O dashboard transforma esses resultados em uma interface visual interativa acessível pelo navegador.

O módulo `dashboard.py` é construído inteiramente com Gradio e organizado em três abas.

A primeira aba, **Resultados dos Experimentos**, carrega o CSV gerado pelo script de experimentos e exibe um gráfico de barras comparando F1-macro de validação e de teste lado a lado para cada experimento, uma tabela com ranking completo e posições numeradas, e uma análise automática em texto português do melhor e do pior pipeline identificado.

A segunda aba, **Classificação em Tempo Real**, permite carregar qualquer pipeline serializado em `.pkl` e classificar textos digitados diretamente na interface. O resultado inclui a classe predita, a confiança do modelo e o nome do pipeline em uso. Cinco exemplos de textos estão disponíveis para facilitar o teste.

A terceira aba, **Sobre o Projeto**, apresenta um resumo contextualizado do projeto para quem acessa o dashboard sem ter lido a documentação completa.

O Gradio foi escolhido por estar alinhado com o eixo de visualização de dados do módulo, por ser amplamente utilizado na comunidade de machine learning para demonstração de modelos, e por não exigir conhecimento de desenvolvimento web para produzir uma interface funcional e apresentável.

O módulo `dashboard.py` é construído com Gradio 4.x e implementa quatro correções técnicas relevantes.

O **FIX 1** garante que `matplotlib.use("Agg")` seja chamado no topo do módulo, antes de qualquer importação de `pyplot`. Chamá-lo dentro de um callback Gradio em ambiente multithread causa deadlock no backend Agg.

O **FIX 2** implementa um cache de pipelines com `threading.Lock` para evitar race conditions quando múltiplos callbacks tentam carregar o mesmo arquivo simultaneamente.

O **FIX 3** substitui `app.load()` por `gr.on("load")` com `app.queue()` ativo. O `app.load()` original disparava um callback bloqueante antes de qualquer aba ser renderizada, impedindo a navegação entre abas no Gradio 4.x.

O **FIX 4** armazena apenas o caminho (string) do pipeline no `gr.State`, nunca o objeto serializado. Passar objetos pesados no `gr.State` do Gradio 4.x os serializa e desserializa a cada evento, gerando latência e podendo travar callbacks subsequentes.

O dashboard exibe os nomes reais das classes do pipeline na aba de classificação em tempo real, suportando qualquer idioma e qualquer mapeamento configurado durante o treinamento.

### Como iniciar

```bash
python scripts/dashboard.py
# http://localhost:7860

python scripts/dashboard.py --share
# Gera link público temporário
```

---

## Conclusão

O desenvolvimento deste pipeline tornou evidente que a representação do texto é tão determinante para o resultado quanto o algoritmo de classificação. A generalização para múltiplos idiomas e datasets demonstra que um pipeline bem projetado não precisa ser reescrito quando o problema muda: basta configurar o idioma, as colunas e o mapeamento de classes.

---

## Referências

- PANG, B.; LEE, L.; VAITHYANATHAN, S. Thumbs up? Sentiment Classification using Machine Learning Techniques. Proceedings of EMNLP, 2002.
- PANG, B.; LEE, L. A Sentimental Education. Proceedings of ACL, 2004.
- MAAS, A. L. et al. Learning Word Vectors for Sentiment Analysis. Proceedings of ACL, 2011.
- MIKOLOV, T. et al. Efficient Estimation of Word Representations in Vector Space. ICLR, 2013.
- DEERWESTER, S. et al. Indexing by Latent Semantic Analysis. Journal of the American Society for Information Science, 1990.
- KE, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Proceedings of NeurIPS, 2017.
- HASTIE, T.; TIBSHIRANI, R.; FRIEDMAN, J. The Elements of Statistical Learning. 2. ed. Springer, 2009.

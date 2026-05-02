"""
Pré-processamento configurável de textos para análise de sentimentos.
Cada etapa é controlada por um dicionário de configuração.

Suporta múltiplos idiomas via parâmetro "language". Os idiomas disponíveis
dependem dos dados do NLTK instalados: "english", "portuguese", "spanish",
"french", "german", "italian" e outros suportados pelo NLTK Stopwords.
"""

import re
import unicodedata
import logging
import nltk

logger = logging.getLogger(__name__)

# Palavras de negação por idioma. Podem ser sobrescritas via "negation_words" na config.
NEGATION_WORDS_POR_IDIOMA = {
    "english": {
        "no", "not", "nor", "never", "neither", "none", "nobody", "nothing",
        "nowhere", "cannot", "can't", "won't", "don't", "doesn't", "didn't",
        "isn't", "wasn't", "weren't", "hasn't", "haven't", "hadn't",
        "wouldn't", "shouldn't", "couldn't",
    },
    "portuguese": {
        "não", "nunca", "jamais", "nem", "nenhum", "nenhuma", "nada",
        "ninguém", "tampouco", "sequer",
    },
    "spanish": {
        "no", "nunca", "jamás", "tampoco", "ningún", "ninguna", "nadie", "nada", "ni",
    },
    "french": {
        "ne", "pas", "jamais", "rien", "personne", "aucun", "aucune", "ni",
    },
    "german": {
        "nicht", "kein", "keine", "keiner", "keines", "niemals", "nie", "nichts", "niemand", "weder",
    },
    "italian": {
        "non", "mai", "niente", "nessuno", "nessuna", "né", "nemmeno", "neanche",
    },
}

EMOTICON_MAP = {
    ":)": "smile", ":-)": "smile", ":D": "laugh",
    ":(": "sad",   ":-(": "sad",   ":/": "unsure",
    ";)": "wink",  "<3": "love",   ":o": "surprised",
    "xD": "laugh", "=)": "smile",  ":P": "playful",
}

# Idiomas disponíveis no SnowballStemmer do NLTK
SNOWBALL_IDIOMAS = {
    "portuguese": "portuguese", "english": "english",
    "spanish": "spanish", "french": "french",
    "german": "german", "italian": "italian",
    "dutch": "dutch", "swedish": "swedish",
}

# Stopwords embutidas de emergência quando o NLTK não está disponível
_SW_FALLBACK = {
    "english": {
        "i","me","my","we","our","you","your","he","him","his","she","her",
        "it","its","they","them","their","what","which","who","this","that",
        "these","those","am","is","are","was","were","be","been","have","has",
        "had","do","does","did","a","an","the","and","but","if","or","as",
        "of","at","by","for","with","to","from","in","out","on","so","than",
        "too","very","just","will","should","now",
    },
    "portuguese": {
        "a","ao","aos","as","até","com","como","da","das","de","do","dos",
        "e","em","entre","essa","esse","esta","este","eu","já","mas","me",
        "mesmo","muito","na","nas","nos","nós","nossa","nosso","o","os","ou",
        "para","pela","pelo","por","qual","que","quem","se","sem","seu",
        "seus","sua","suas","também","te","tem","ter","teu","tua","um","uma",
        "você","vocês",
    },
}


def _baixar_nltk():
    """Tenta baixar recursos NLTK; continua sem erro se não houver rede."""
    for caminho, nome in [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("tokenizers/punkt",     "punkt"),
        ("corpora/stopwords",    "stopwords"),
        ("corpora/wordnet",      "wordnet"),
    ]:
        try:
            nltk.data.find(caminho)
        except LookupError:
            try:
                nltk.download(nome, quiet=True)
            except Exception:
                pass


_baixar_nltk()


def _tokenizar(texto: str, idioma: str = "english") -> list:
    try:
        from nltk.tokenize import word_tokenize
        return word_tokenize(texto, language=idioma)
    except Exception:
        return texto.split()


def _obter_stopwords(idioma: str) -> set:
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words(idioma))
    except Exception:
        return _SW_FALLBACK.get(idioma, _SW_FALLBACK["english"])


class Preprocessor:
    """
    Pré-processador configurável e agnóstico de idioma.

    Parâmetros aceitos na configuração

        language (str)              idioma para tokenização, stopwords e stemming.
                                    Padrão "english". Use "portuguese" para PT-BR.

        lowercase (bool)            converte para minúsculas. Padrão True.

        remove_stopwords (bool|str) True remove todas as stopwords; "keep_negations"
                                    preserva palavras de negação do idioma configurado.

        negation_words (list)       lista customizada de negações (sobrescreve o padrão).

        normalization (str|None)    "stemming", "lemmatization" (somente inglês) ou None.

        handle_negations (bool)     marca tokens após negação com sufixo _NEG.

        remove_urls (bool)          remove URLs do texto.

        normalize_emojis (bool)     substitui emoticons e emojis unicode por palavras.

    Exemplo em português
        pre = Preprocessor({
            "language": "portuguese",
            "lowercase": True,
            "remove_stopwords": "keep_negations",
            "normalization": "stemming",
            "handle_negations": True,
            "remove_urls": True,
        })
        resultado = pre.transform(["Não gostei nada desse filme!"])

    Exemplo em inglês
        pre = Preprocessor({
            "language": "english",
            "lowercase": True,
            "remove_stopwords": "keep_negations",
            "normalization": "lemmatization",
            "handle_negations": True,
        })
        resultado = pre.transform(["I did NOT love this movie!"])
    """

    def __init__(self, config: dict):
        self.config = config
        self.idioma = config.get("language", "english")
        self._stemmer = None
        self._lemmatizer = None
        self._stopwords = None
        self._negation_words = None
        self._configurar()

    def _configurar(self):
        # Palavras de negação
        custom_neg = self.config.get("negation_words")
        if custom_neg:
            self._negation_words = set(w.lower() for w in custom_neg)
        else:
            self._negation_words = NEGATION_WORDS_POR_IDIOMA.get(
                self.idioma, NEGATION_WORDS_POR_IDIOMA["english"]
            )

        # Normalização morfológica
        norm = self.config.get("normalization")
        if norm == "stemming":
            lang = SNOWBALL_IDIOMAS.get(self.idioma, "english")
            try:
                from nltk.stem import SnowballStemmer
                self._stemmer = SnowballStemmer(lang)
            except Exception:
                from nltk.stem import PorterStemmer
                self._stemmer = PorterStemmer()
        elif norm == "lemmatization":
            if self.idioma != "english":
                logger.warning(
                    "Lematização via WordNet disponível apenas para inglês. "
                    "Usando stemming para o idioma '%s'.", self.idioma
                )
                lang = SNOWBALL_IDIOMAS.get(self.idioma, "english")
                try:
                    from nltk.stem import SnowballStemmer
                    self._stemmer = SnowballStemmer(lang)
                except Exception:
                    pass
            else:
                try:
                    from nltk.stem import WordNetLemmatizer
                    self._lemmatizer = WordNetLemmatizer()
                except Exception:
                    from nltk.stem import PorterStemmer
                    self._stemmer = PorterStemmer()

        # Stopwords
        modo_sw = self.config.get("remove_stopwords", False)
        if modo_sw:
            base = _obter_stopwords(self.idioma)
            self._stopwords = (
                base - self._negation_words if modo_sw == "keep_negations" else base
            )

    def _remover_url(self, texto: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", " ", texto)

    def _normalizar_emojis(self, texto: str) -> str:
        for emoticon, palavra in EMOTICON_MAP.items():
            texto = texto.replace(emoticon, f" {palavra} ")
        resultado = []
        for char in texto:
            if ord(char) > 127:
                categoria = unicodedata.category(char)
                if categoria in ("So", "Cs", "Co"):
                    nome = unicodedata.name(char, "").lower().replace(" ", "_")
                    resultado.append(f" {nome} " if nome else " ")
                else:
                    resultado.append(char)
            else:
                resultado.append(char)
        return "".join(resultado)

    def _marcar_negacoes(self, tokens: list) -> list:
        """Marca até 5 tokens após negação com sufixo _NEG. Funciona para qualquer idioma."""
        resultado, negando, janela = [], False, 0
        pontuacao = {",", ".", "!", "?", ";", ":"}
        for token in tokens:
            if token.lower() in self._negation_words:
                negando, janela = True, 0
                resultado.append(token)
            elif negando:
                if token in pontuacao or janela >= 5:
                    negando = False
                    resultado.append(token)
                else:
                    resultado.append(token + "_NEG")
                    janela += 1
            else:
                resultado.append(token)
        return resultado

    def _normalizar_token(self, token: str) -> str:
        if self._stemmer:
            try:
                return self._stemmer.stem(token)
            except Exception:
                return token
        if self._lemmatizer:
            try:
                return self._lemmatizer.lemmatize(token)
            except LookupError:
                return token
        return token

    def _processar_um(self, texto: str) -> str:
        if not isinstance(texto, str):
            texto = str(texto)
        if self.config.get("remove_urls"):
            texto = self._remover_url(texto)
        if self.config.get("normalize_emojis"):
            texto = self._normalizar_emojis(texto)
        if self.config.get("lowercase", True):
            texto = texto.lower()
        texto = re.sub(r"<[^>]+>", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        tokens = _tokenizar(texto, idioma=self.idioma)
        if self.config.get("handle_negations"):
            tokens = self._marcar_negacoes(tokens)
        if self._stopwords:
            tokens = [t for t in tokens if t.lower() not in self._stopwords]
        tokens = [self._normalizar_token(t) for t in tokens]
        return " ".join(tokens)

    def transform(self, textos: list) -> list:
        """Processa uma lista de textos e retorna a versão limpa."""
        return [self._processar_um(t) for t in textos]

    def get_config(self) -> dict:
        return self.config.copy()

    def get_negation_words(self) -> set:
        """Retorna as palavras de negação em uso."""
        return self._negation_words.copy() if self._negation_words else set()

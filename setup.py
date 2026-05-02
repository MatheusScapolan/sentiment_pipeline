from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sentiment_pipeline",
    version="1.0.0",
    description="Pipeline dinâmico e agnóstico para análise de sentimentos com classificadores clássicos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24", "scipy>=1.10", "scikit-learn>=1.3",
        "pandas>=2.0", "nltk>=3.8", "lightgbm>=4.0",
        "matplotlib>=3.7", "gradio>=4.0",
    ],
    extras_require={
        "word2vec": ["gensim>=4.3"],
        "optuna":   ["optuna>=3.3"],
        "dev":      ["pytest>=7.4", "pytest-cov>=4.1"],
    },
)

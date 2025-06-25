from pathlib import Path
import sys

PASTA_PROJETO = Path(__file__).resolve().parents[2]

# Verifica se a pasta de dados existe
PASTA_DADOS = PASTA_PROJETO / "data"
if not PASTA_DADOS.exists():
    raise FileNotFoundError(f"Pasta de dados não encontrada: {PASTA_DADOS}")

# Caminhos dos arquivos
DADOS_ORIGINAIS = PASTA_DADOS / "ml_project1_data.csv"
DADOS_DROP = PASTA_DADOS / "customers_new_features_and_drop.csv"
DADOS_TRATADOS = PASTA_DADOS / "ml_project1_data.parquet"

# Verifica se o arquivo original existe
if not DADOS_ORIGINAIS.exists():
    raise FileNotFoundError(f"Arquivo de dados original não encontrado: {DADOS_ORIGINAIS}")

# Cria pastas se não existirem
PASTA_MODELOS = PASTA_PROJETO / "modelos"
PASTA_RELATORIOS = PASTA_PROJETO / "reports"
PASTA_IMAGENS = PASTA_RELATORIOS / "images"

for pasta in [PASTA_MODELOS, PASTA_RELATORIOS, PASTA_IMAGENS]:
    pasta.mkdir(parents=True, exist_ok=True)

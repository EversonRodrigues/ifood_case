import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def graficos_elbow_silhouette(x, random_state=42, intervalo_k=(2,11)):
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), tight_layout=True)
    
    elbow = {}
    silhouette = []
    
    k_range = range(*intervalo_k)
    
    # Loop pelos valores de K para calcular a inércia e o score de silhouette
    for i in k_range:
        kmeans = KMeans(n_clusters=i, random_state=random_state, n_init=10)
        kmeans.fit(x)
        elbow[i] = kmeans.inertia_
        labels = kmeans.labels_
        silhouette.append(silhouette_score(x, labels))
    
    # --- DETECÇÃO AUTOMÁTICA DO COTOVELO ---
    k_values = np.array(list(elbow.keys()))
    inertia_values = np.array(list(elbow.values()))
    
    line = np.array([k_values[0], inertia_values[0]]), np.array([k_values[-1], inertia_values[-1]])
    line_vec = line[1] - line[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    
    distances = []
    for i in range(len(k_values)):
        p = np.array([k_values[i], inertia_values[i]])
        vec_from_first = p - line[0]
        scalar_proj = np.dot(vec_from_first, line_vec_norm)
        proj = scalar_proj * line_vec_norm
        perpendicular = vec_from_first - proj
        distances.append(np.linalg.norm(perpendicular))
    
    best_elbow_k = k_values[np.argmax(distances)]
    
    sns.lineplot(x=k_values, y=inertia_values, ax=axs[0])
    axs[0].scatter(best_elbow_k, elbow[best_elbow_k], color='red', s=100)
    axs[0].annotate(f"Best K: {best_elbow_k}", (best_elbow_k, elbow[best_elbow_k]),
                    textcoords="offset points", xytext=(10, -15), ha='left', color='red', fontsize=10)
    axs[0].set_xlabel("K")
    axs[0].set_ylabel("Inertia")
    axs[0].set_title("Elbow Method")
    
    best_silhouette_k = k_range[silhouette.index(max(silhouette))]
    sns.lineplot(x=list(k_range), y=silhouette, ax=axs[1])
    axs[1].scatter(best_silhouette_k, max(silhouette), color='blue', s=100)
    axs[1].annotate(f"Best K: {best_silhouette_k}", (best_silhouette_k, max(silhouette)),
                    textcoords="offset points", xytext=(10, -15), ha='left', color='blue', fontsize=10)
    axs[1].set_xlabel("K")
    axs[1].set_ylabel("Silhouette Score")
    axs[1].set_title("Silhouette Method")
    
    plt.show()

def visualizar_clusters(
    dataframe,
    colunas,
    quantidade_cores,
    centroids,
    mostrar_centroids=True, 
    mostrar_pontos=False,
    coluna_cluster=None,
):

    fig = plt.figure()
    
    ax = fig.add_subplot(111, projection="3d")
    
    cores = plt.cm.tab10.colors[:quantidade_cores]
    cores = ListedColormap(cores)
    
    x = dataframe[colunas[0]]
    y = dataframe[colunas[1]]
    z = dataframe[colunas[2]]
    
    ligar_centroids = mostrar_centroids
    ligar_pontos = mostrar_pontos
    
    for i, centroid in enumerate(centroids):
        if ligar_centroids: 
            ax.scatter(*centroid, s=500, alpha=0.5)
            ax.text(*centroid, f"{i}", fontsize=20, horizontalalignment="center", verticalalignment="center")
    
        if ligar_pontos:
            s = ax.scatter(x, y, z, c=coluna_cluster, cmap=cores)
            ax.legend(*s.legend_elements(), bbox_to_anchor=(1.3, 1))
    
    ax.set_xlabel(colunas[0])
    ax.set_ylabel(colunas[1])
    ax.set_zlabel(colunas[2])
    ax.set_title("Clusters")
    
    plt.show()

def inspect_outliers(dataframe, column, whisker_width=1.5):
    """
    Inspecionar Outliers de uma Coluna Numérica de Forma Robusta e Didática.

    Esta função identifica possíveis outliers (valores fora do padrão esperado)
    em uma coluna numérica de um DataFrame, usando o método estatístico chamado IQR (Intervalo Interquartílico).

    Parâmetros
    ----------
    dataframe : pandas.DataFrame
        O conjunto de dados que contém a coluna a ser analisada.

    column : str
        O nome da coluna (como string) onde serão inspecionados os outliers.
        Exemplo: "Income", "Age", etc.

    whisker_width : float, opcional (padrão=1.5)
        O multiplicador usado para definir os limites inferior e superior.
        Um valor padrão de 1.5 significa que serão considerados outliers os pontos
        que estão além de 1.5 vezes o intervalo interquartílico (IQR).
        Para detecção mais ou menos sensível, ajuste esse valor.

    Retorno
    -------
    pandas.DataFrame
        Um novo DataFrame contendo apenas os registros que foram identificados como outliers.

    Exemplo de uso
    --------------
    outliers_income = inspect_outliers(df, "Income", whisker_width=1.5)
    print(outliers_income)
    """

    # ------------------------------
    # ✅ Validação de parâmetros
    # ------------------------------

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("O parâmetro 'dataframe' precisa ser um pandas.DataFrame.")

    if not isinstance(column, str):
        raise TypeError("O parâmetro 'column' precisa ser uma string com o nome da coluna.")

    if column not in dataframe.columns:
        raise ValueError(f"A coluna '{column}' não existe no DataFrame.")

    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise TypeError(f"A coluna '{column}' precisa conter valores numéricos para análise de outliers.")

    # ------------------------------
    # ✅ Cálculo do IQR
    # ------------------------------
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr

    # ------------------------------
    # ✅ Filtrar Outliers
    # ------------------------------
    outliers_df = dataframe[
        (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
    ]

    # ------------------------------
    # ✅ Feedback amigável
    # ------------------------------
    print(f"\nAnalisando a coluna '{column}':")
    print(f"Q1 (25%): {q1}")
    print(f"Q3 (75%): {q3}")
    print(f"IQR: {iqr}")
    print(f"Limite inferior: {lower_bound}")
    print(f"Limite superior: {upper_bound}")
    print(f"Total de outliers encontrados: {len(outliers_df)}")

    return outliers_df

import pandas as pd

def inspect_all_outliers(dataframe, whisker_width=1.5):
    """
    Detecta Outliers em Todas as Colunas Numéricas de um DataFrame.

    Parâmetros
    ----------
    dataframe : pandas.DataFrame
        Seu conjunto de dados.

    whisker_width : float, opcional (padrão=1.5)
        Fator multiplicador do IQR.

    Retorno
    -------
    dict
        Dicionário com os outliers por coluna.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("O parâmetro 'dataframe' precisa ser um pandas.DataFrame.")

    numeric_cols = dataframe.select_dtypes(include=["number"]).columns
    outliers_dict = {}

    for column in numeric_cols:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - whisker_width * iqr
        upper_bound = q3 + whisker_width * iqr

        outliers = dataframe[
            (dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)
        ]

        print(f"\n📌 Coluna: {column}")
        print(f"Q1: {q1}, Q3: {q3}, IQR: {iqr}")
        print(f"Limite Inferior: {lower_bound}, Limite Superior: {upper_bound}")
        print(f"Total de outliers: {len(outliers)}")

        outliers_dict[column] = outliers

    return outliers_dict

import pandas as pd

def remove_outliers(dataframe, column, whisker_width=1.5):
    """
    Remove os Outliers de uma Coluna Numérica Usando o Método IQR.

    Parâmetros
    ----------
    dataframe : pandas.DataFrame
        O DataFrame original com os dados.

    column : str
        Nome da coluna numérica de onde os outliers serão removidos.

    whisker_width : float, opcional (padrão=1.5)
        Multiplicador do IQR para definir os limites de detecção dos outliers.

    Retorno
    -------
    pandas.DataFrame
        Um novo DataFrame com os outliers removidos (as linhas contendo os outliers são excluídas).
    
    Exemplo de uso
    --------------
    df_sem_outliers = remove_outliers(df, "Income", whisker_width=1.5)
    """

    # Validações básicas
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("O parâmetro 'dataframe' precisa ser um pandas.DataFrame.")
    if column not in dataframe.columns:
        raise ValueError(f"A coluna '{column}' não existe no DataFrame.")
    if not pd.api.types.is_numeric_dtype(dataframe[column]):
        raise TypeError(f"A coluna '{column}' precisa ser numérica.")

    # Cálculo do IQR
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - whisker_width * iqr
    upper_bound = q3 + whisker_width * iqr

    # Filtrar os registros que estão DENTRO dos limites (não são outliers)
    cleaned_df = dataframe[
        (dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)
    ].copy()

    print(f"\n✅ Outliers removidos da coluna '{column}':")
    print(f"Antes: {len(dataframe)} linhas | Depois: {len(cleaned_df)} linhas | Removidos: {len(dataframe) - len(cleaned_df)}")

    return cleaned_df

def pairplot(
    dataframe, columns, hue_column=None, alpha=0.5, corner=True, palette="tab10"
):
    """Função para gerar pairplot.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Dataframe com os dados.
    columns : List[str]
        Lista com o nome das colunas (strings) a serem utilizadas.
    hue_column : str, opcional
        Coluna utilizada para hue, por padrão None
    alpha : float, opcional
        Valor de alfa para transparência, por padrão 0.5
    corner : bool, opcional
        Se o pairplot terá apenas a diagonal inferior ou será completo, por padrão True
    palette : str, opcional
        Paleta a ser utilizada, por padrão "tab10"
    """

    analysis = columns.copy() + [hue_column]

    sns.pairplot(
        dataframe[analysis],
        diag_kind="kde",
        hue=hue_column,
        plot_kws=dict(alpha=alpha),
        corner=corner,
        palette=palette,
    )
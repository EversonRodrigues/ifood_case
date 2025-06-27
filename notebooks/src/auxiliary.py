import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import PercentFormatter
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os
import numpy as np

os.environ["OMP_NUM_THREADS"] = "9"

def graficos_elbow_silhouette(x, random_state=42, intervalo_k=(2,11)):
    """
    Gera visualizações dos métodos do Cotovelo (Elbow) e Silhouette para determinar o número ideal de clusters (K)
    em uma clusterização usando KMeans. A função também realiza a detecção automática do "cotovelo" com base
    na maior distância perpendicular entre os pontos e a linha entre o primeiro e o último ponto de inércia.

    Parâmetros
    ----------
    x : array-like or DataFrame
        Matriz de dados numéricos para aplicar o KMeans.

    random_state : int, opcional (default=42)
        Semente para garantir reprodutibilidade dos resultados do KMeans.

    intervalo_k : tuple (int, int), opcional (default=(2, 11))
        Intervalo de valores de K a serem testados (início incluso, fim exclusivo).
        Exemplo: (2, 11) testará de K=2 a K=10.

    Retorno
    -------
    None
        Apenas exibe dois gráficos: um para o método do cotovelo com detecção automática
        e outro para os scores de silhouette. Cada gráfico destaca visualmente o melhor K identificado.

    Notas
    -----
    - O método do cotovelo utiliza a distância perpendicular de cada ponto à linha entre extremos da curva de inércia.
    - O melhor K pelo silhouette é o valor que maximiza o coeficiente de silhouette.
    - Ideal para experimentação visual durante projetos de clusterização com KMeans.
    """
    
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


def plot_columns_percent_by_cluster(
    dataframe,
    columns,
    rows_cols=(2, 3),
    figsize=(15, 8),
    column_cluster="cluster",
    palette="tab10",
):
    """
    Gera gráficos de barras empilhadas com percentual de ocorrência por cluster para variáveis categóricas.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame contendo os dados.
    columns : List[str]
        Lista com nomes das colunas categóricas a serem visualizadas.
    rows_cols : tuple, optional
        (linhas, colunas) no grid de subplots. Default = (2, 3)
    figsize : tuple, optional
        Tamanho da figura matplotlib. Default = (15, 8)
    column_cluster : str, optional
        Nome da coluna de cluster (usualmente 'cluster'). Default = "cluster"
    """
    fig, axs = plt.subplots(
        nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize, sharey=True
    )

    axs = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])

    for ax, col in zip(axs, columns):
        h = sns.histplot(
            x=column_cluster,
            hue=col,
            data=dataframe,
            ax=ax,
            multiple="fill",
            stat="percent",
            discrete=True,
            shrink=0.8,
        )

        # Ocultar spines
        h.spines["top"].set_visible(False)
        h.spines["right"].set_visible(False)
        h.spines["left"].set_visible(False)

        # Formatadores e ajustes visuais
        h.set_title(col, fontsize=12, weight="bold")
        n_clusters = dataframe[column_cluster].nunique()
        h.set_xticks(range(n_clusters))
        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel("")
        h.tick_params(axis="both", which="both", length=0)

        # Inserir rótulos percentuais nas barras
        for bars in h.containers:
            h.bar_label(
                bars,
                label_type="center",
                labels=[f"{b.get_height():.1%}" for b in bars],
                color="white",
                weight="bold",
                fontsize=10,
            )

        # Remover contorno das barras
        for bar in h.patches:
            bar.set_linewidth(0)

    # Ajuste de espaçamento entre subplots
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    plt.show()

def plot_columns_percent_hue_cluster(
    dataframe,
    columns,
    rows_cols=(2, 3),
    figsize=(15, 8),
    column_cluster="cluster",
    palette="tab10",
    mostrar_legenda=False,
):
    """
    Gera gráficos de barras empilhadas com percentual de categorias, com cor por cluster.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame contendo os dados.
    columns : List[str]
        Lista de colunas categóricas a serem visualizadas.
    rows_cols : tuple, optional
        Grid de subplots (linhas, colunas). Default = (2, 3)
    figsize : tuple, optional
        Tamanho da figura. Default = (15, 8)
    column_cluster : str, optional
        Nome da coluna com os clusters. Default = "cluster"
    palette : str, optional
        Paleta de cores. Default = "tab10"
    mostrar_legenda : bool, optional
        Se True, exibe legenda única ao final. Se False, suprime legenda e mantém eixo X. Default = True.
    """
    fig, axs = plt.subplots(
        nrows=rows_cols[0], ncols=rows_cols[1], figsize=figsize, sharey=True
    )

    axs = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])
    legend_handles = None
    legend_labels = None

    for i, (ax, col) in enumerate(zip(axs, columns)):
        h = sns.histplot(
            x=col,
            hue=column_cluster,
            data=dataframe,
            ax=ax,
            multiple="fill",
            stat="percent",
            discrete=True,
            shrink=0.8,
            palette=palette,
        )

        if dataframe[col].dtype != "object":
            h.set_xticks(range(dataframe[col].nunique()))

        h.yaxis.set_major_formatter(PercentFormatter(1))
        h.set_ylabel("")
        h.tick_params(axis="both", which="both", length=0)

        # Remover bordas visuais
        h.spines["top"].set_visible(False)
        h.spines["right"].set_visible(False)
        h.spines["left"].set_visible(False)

        h.set_title(col.title(), fontsize=12, weight="bold")

        # Adicionar rótulos de porcentagem nas barras
        for bars in h.containers:
            h.bar_label(
                bars,
                label_type="center",
                labels=[f"{b.get_height():.1%}" for b in bars],
                color="white",
                weight="bold",
                fontsize=10,
            )

        for bar in h.patches:
            bar.set_linewidth(0)

        # Legenda só se for solicitado
        if mostrar_legenda:
            if h.get_legend():
                legend = h.get_legend()
                legend_handles = legend.legend_handles
                legend_labels = [text.get_text() for text in legend.get_texts()]
                legend.remove()
        else:
            ax.set_xticks([])
            ax.set_xlabel("")
            if h.get_legend():
                h.get_legend().remove()

    # Adicionar legenda fora do grid
    if mostrar_legenda and legend_handles and legend_labels:
        fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),
            ncols=dataframe[column_cluster].nunique(),
            title="Clusters",
            frameon=False,
        )

    plt.subplots_adjust(hspace=0.3, wspace=0.3, bottom=0.2)
    
    plt.show()
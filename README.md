# 🧠 Segmentação e Classificação de Clientes iFood com Ciência de Dados

Este projeto simula um cenário de negócio real onde a empresa precisa entender melhor o perfil de seus clientes para direcionar campanhas de marketing de forma eficiente.

Foram aplicadas técnicas de:

* Análise Exploratória
* Clusterização (KMeans)
* Redução de Dimensionalidade (PCA)
* Classificação supervisionada com validação cruzada
* Otimização de modelos com GridSearchCV

O pipeline completo foi estruturado de forma modular e reutilizável usando boas práticas de engenharia com `Pipeline`, `ColumnTransformer`, reamostragem (`RandomUnderSampler`), validação estratificada e persistência de modelos.

🔗 Baseado no dataset do case de seleção da equipe de dados do [iFood](https://github.com/ifood/ifood-data-business-analyst-test).

---

## 📆 Estrutura do Projeto

```
├── CASES/          <- Explicação do problema e do contexto de negócio
├── DATA/           <- Dados originais e derivados (tratados, balanceados, PCA, etc.)
├── MODELS/         <- Modelos salvos (.joblib) após treinamento e otimização
├── NOTEBOOKS/      <- Etapas de desenvolvimento do projeto (EDA, Clustering, PCA, Classificação)
├── REFERENCE/      <- Dicionário de dados com explicação das features
├── REPORTS/
│   └── IMAGENS/    <- Gráficos salvos e visualizações geradas
└── README.md       <- Documentação principal do projeto
```

---

## 📊 Etapas Executadas

1. 📊 **EDA (Análise Exploratória)**
2. 🔀 **Clusterização e criação de perfis**
3. 🧠 **PCA para visualização dos grupos**
4. ✅ **Classificação dos clientes por cluster**
5. ⚙️ **Otimização com GridSearchCV**
6. 📈 **Avaliação com curvas ROC, PRC e métricas de negócio**

---

## ✅ Resultados Alcançados

* Foram identificados 3 perfis claros de clientes com base em consumo, filhos, idade e aceitação de campanhas.
* A **Regressão Logística** foi o melhor classificador com ótimo equilíbrio entre precision, recall e AP.
* A **precisão média (Average Precision)** aumentou após otimização com GridSearch.
* As métricas e visualizações confirmaram a **capacidade do modelo de gerar valor real para o negócio**.

---

## 🎓 Aprendizados

* Aplicar o ciclo completo de Machine Learning supervisionado e não supervisionado.
* Importância do pré-processamento, seleção de features e reamostragem.
* Uso de pipelines escaláveis com `Pipeline` + `ColumnTransformer` + `GridSearchCV`.
* Avaliação de modelos com métricas robustas como ROC AUC e Average Precision.
* Entrega de insights claros por meio de visualizações e comunicação visual.

---

## 📂 Requisitos

| Biblioteca       | Versão            |
| ---------------- | ----------------- |
| Python           | 3.11+             |
| Pandas           | 2.2.2             |
| Scikit-Learn     | 1.4.2             |
| Imbalanced-Learn | 0.12.3            |
| Seaborn          | 0.13.2            |
| Matplotlib       | 3.8.4             |
| ydata-profiling  | Opcional para EDA |

---

## 📢 Contato

👨‍💻 **Everson Rodrigues**
🔍 Em busca da primeira oportunidade na área de dados
🌐 [LinkedIn](https://www.linkedin.com/in/eversonrodrigues10)
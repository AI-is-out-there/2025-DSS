import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Загрузка и подготовка данных ---
df = pd.read_csv("SD_MSEG_202512141134.csv", sep=";")

# Агрегация данных по PRODUCT_ID
aggregated_data = df.groupby("PRODUCT_ID").agg({
    "REVENUE": ["sum", "mean"],
    "QNTY": ["sum", "mean", "count"]
}).reset_index()

# Переименование колонок
aggregated_data.columns = [
    "PRODUCT_ID",
    "Total_Revenue",
    "Average_Revenue_per_Transaction",
    "Total_Quantity",
    "Average_Quantity_per_Transaction",
    "Number_of_Transactions"
]

# Добавление средней цены за единицу
aggregated_data["Average_Price_per_Unit"] = aggregated_data["Total_Revenue"] / aggregated_data["Total_Quantity"]

# --- 2. Нормализация данных ---
features = aggregated_data[["Total_Revenue", "Total_Quantity", "Average_Price_per_Unit", "Number_of_Transactions"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# --- 3. Agglomerative Clustering ---
agg_clustering = AgglomerativeClustering(n_clusters=5)
aggregated_data["Cluster"] = agg_clustering.fit_predict(scaled_features)

# --- 4. Анализ кластеров ---
# Средние значения по кластерам
cluster_summary = aggregated_data.groupby("Cluster").agg({
    "Total_Revenue": "mean",
    "Total_Quantity": "mean",
    "Average_Price_per_Unit": "mean",
    "Number_of_Transactions": "mean",
    "PRODUCT_ID": "count"
}).rename(columns={"PRODUCT_ID": "Number_of_Products"})

print("Средние значения по кластерам:")
print(cluster_summary)

# --- 5. Визуализация кластеров ---
# t-SNE для визуализации
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_result = tsne.fit_transform(scaled_features)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    x=tsne_result[:, 0],
    y=tsne_result[:, 1],
    hue=aggregated_data["Cluster"],
    palette="viridis",
    s=100,
    alpha=0.8
)
plt.title("Визуализация кластеров (t-SNE)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid()
plt.show()

# --- 6. Топ-5 товаров в каждом кластере по выручке ---
for cluster in sorted(aggregated_data["Cluster"].unique()):
    cluster_products = aggregated_data[aggregated_data["Cluster"] == cluster]
    top_products = cluster_products.sort_values("Total_Revenue", ascending=False).head(5)
    print(f"\nТоп-5 товаров в кластере {cluster} по выручке:")
    print(top_products[["PRODUCT_ID", "Total_Revenue", "Total_Quantity", "Average_Price_per_Unit"]])

# --- 7. Сохранение результатов ---
aggregated_data.to_csv("aggregated_data_with_clusters.csv", index=False)
cluster_summary.to_csv("cluster_summary.csv")

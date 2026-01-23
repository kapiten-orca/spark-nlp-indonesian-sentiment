import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("plots", exist_ok=True)

# mengambil data hasil analisis
df_sentiment = pd.read_csv(
    "data/results/sentiment_distribution/part-00000-1c950864-4628-4ebf-90d1-f61acf394ea0-c000.csv"
)
df_category = pd.read_csv(
    "data/results/category_sentimentAg/part-00000-bad72808-ce5b-4946-a150-2ec78287a72e-c000.csv"
)
df_location = pd.read_csv(
    "data/results/location_sentiment/part-00000-e8320695-9e27-48e7-9403-1f8032511556-c000.csv"
)

# visual distribusi sentimen pie chart
plt.figure()
plt.pie(
    df_sentiment["count"],
    labels=df_sentiment["predicted_sentiment"],
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Sentiment Distribution")
plt.axis("equal")
plt.savefig("plots/sentiment_distribution.png")
plt.close()

# visualisasi data sentimen per kategori dengan bar chart
pivot_category = df_category.pivot(
    index="Category",
    columns="predicted_sentiment",
    values="count"
).fillna(0)

pivot_category.plot(kind="bar", stacked=True, figsize=(10, 6))
plt.title("Category vs Sentiment")
plt.xlabel("Category")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/category_vs_sentiment.png")
plt.close()

# visualisasi data sentimen per lokasi dengan bar chart
df_location_sorted = df_location.sort_values(
    "avg_sentiment_score",
    ascending=False
)

plt.figure(figsize=(10, 5))
plt.bar(
    df_location_sorted["Location"],
    df_location_sorted["avg_sentiment_score"]
)
plt.title("Average Sentiment Score per Location")
plt.xlabel("Location")
plt.ylabel("Average Sentiment Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/location_sentiment_score.png")
plt.close()

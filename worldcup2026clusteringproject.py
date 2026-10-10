
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

sns.set_theme(style="whitegrid", context="notebook")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================

data = pd.read_csv(
    "C:/Users/lenovo/Desktop/worldcup2026/playersdataset.txt",
    sep=None,
    engine="python"
)

# Nettoyage des noms de colonnes
data.columns = data.columns.str.strip()

print("Dimensions initiales :", data.shape)
print(data.head())


# ============================================================
# 2. NETTOYAGE DES DONNEES
# ============================================================

# Conversion des variables utiles en valeurs numeriques
numeric_cols = [
    "games", "minutes", "minutes_90s",
    "goals", "assists", "shots",
    "shots_on_target", "shots_on_target_pct",
    "goals_per_shot", "tackles_won",
    "interceptions", "crosses", "fouled"
]

for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Retirer les gardiens : leurs statistiques ne sont pas comparables
# a celles des joueurs de champ
data = data[
    data["position"].notna()
    & ~data["position"].str.contains("GK", na=False)
].copy()

# Retirer les joueurs sans temps de jeu exploitable
data = data[
    (data["minutes"] >= 90)
    & (data["minutes_90s"] > 0)
].copy()

# Supprimer les doublons eventuels
data = data.drop_duplicates(subset=["player", "team"])

# Verifier que les variables necessaires existent
required_cols = [
    "player", "minutes_90s", "goals", "assists",
    "shots", "shots_on_target", "tackles_won",
    "interceptions", "crosses", "fouled"
]

missing_cols = [col for col in required_cols if col not in data.columns]

if missing_cols:
    raise ValueError(f"Colonnes absentes du dataset : {missing_cols}")


# ============================================================
# 3. CONSTRUCTION DES INDICATEURS D'EFFICACITE
# ============================================================

# On exprime les volumes par 90 minutes jouees
minutes_90s = data["minutes_90s"]

data["goals_per90_calc"] = data["goals"] / minutes_90s
data["assists_per90_calc"] = data["assists"] / minutes_90s
data["shots_on_target_per90_calc"] = (
    data["shots_on_target"] / minutes_90s
)
data["tackles_won_per90"] = data["tackles_won"] / minutes_90s
data["interceptions_per90"] = data["interceptions"] / minutes_90s
data["crosses_per90"] = data["crosses"] / minutes_90s
data["fouled_per90"] = data["fouled"] / minutes_90s

# Indicateurs d'efficacite offensive
# Si un joueur n'a tente aucun tir, son taux est defini a 0
data["shots_on_target_pct_calc"] = (
    100 * data["shots_on_target"] / data["shots"].replace(0, np.nan)
).fillna(0)

data["goals_per_shot_calc"] = (
    data["goals"] / data["shots"].replace(0, np.nan)
).fillna(0)

# Features utilisees par K-Means
features = [
    "goals_per90_calc",
    "assists_per90_calc",
    "shots_on_target_per90_calc",
    "tackles_won_per90",
    "interceptions_per90",
    "crosses_per90",
    "fouled_per90",
    "shots_on_target_pct_calc",
    "goals_per_shot_calc"
]

X = data[features].replace([np.inf, -np.inf], np.nan)

# Remplacer les valeurs manquantes par la mediane
imputer = SimpleImputer(strategy="median")
X_clean = pd.DataFrame(
    imputer.fit_transform(X),
    columns=features,
    index=data.index
)

print("\nNombre de joueurs retenus :", len(X_clean))
print("\nValeurs manquantes apres nettoyage :", X_clean.isna().sum().sum())


# ============================================================
# 4. NORMALISATION
# ============================================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)


# ============================================================
# 5. CHOIX DU NOMBRE DE CLUSTERS
#    Methode du coude + score de silhouette
# ============================================================

k_values = range(2, min(11, len(X_clean)))

inertias = []
silhouette_scores = []

for k in k_values:
    model_test = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels_test = model_test.fit_predict(X_scaled)

    inertias.append(model_test.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels_test)
    )

# Graphique de la methode du coude
plt.figure(figsize=(9, 5))
plt.plot(k_values, inertias, marker="o")
plt.xlabel("Nombre de clusters k")
plt.ylabel("Inertie")
plt.title("Methode du coude")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Graphique du score de silhouette
plt.figure(figsize=(9, 5))
plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Nombre de clusters k")
plt.ylabel("Score de silhouette")
plt.title("Evaluation du clustering par silhouette")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Affichage des scores
print("\nEvaluation des differents k :")
for k, inertia, score in zip(
    k_values, inertias, silhouette_scores
):
    print(f"k={k} | Inertie={inertia:.2f} | Silhouette={score:.4f}")

# Choix initial automatique : silhouette maximale.
# Verifier aussi la methode du coude avant de valider k.
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# Comparaison de plusieurs valeurs de k
print("\nCOMPARAISON DES MODELES")

for k in [2, 3, 4, 5, 6]:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(X_scaled)

    silhouette = silhouette_score(X_scaled, labels)
    calinski = calinski_harabasz_score(X_scaled, labels)
    davies = davies_bouldin_score(X_scaled, labels)

    tailles = pd.Series(labels).value_counts().sort_index()

    print(f"\nK = {k}")
    print(f"Silhouette : {silhouette:.4f}")
    print(f"Calinski-Harabasz : {calinski:.2f}")
    print(f"Davies-Bouldin : {davies:.4f}")
    print("Tailles des clusters :")
    print(tailles.to_dict())


# Choix de k après comparaison
# Si tu retiens 6 clusters pour leur intérêt métier :
best_k = 6

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

data["cluster"] = kmeans.fit_predict(X_scaled)
centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=features
)

print("\nProfils moyens des 6 clusters :")
print(data.groupby("cluster")[features].mean().round(3))

print("\nEffectifs :")
print(data["cluster"].value_counts().sort_index())
# ============================================================
# 6. MODELE K-MEANS FINAL
# ============================================================

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

data["cluster"] = kmeans.fit_predict(X_scaled)

# Centroïdes dans les unites standardisees
centers_scaled = kmeans.cluster_centers_

# Retour aux unites des indicateurs initiaux
centers = pd.DataFrame(
    scaler.inverse_transform(centers_scaled),
    columns=features
)

print("\nCentroïdes des clusters :")
print(centers.round(3))

print("\nNombre de joueurs par cluster :")
print(data["cluster"].value_counts().sort_index())

print("\nProfil moyen de chaque cluster :")
print(
    data.groupby("cluster")[features]
    .mean()
    .round(3)
)

# Associer les noms des joueurs aux clusters
print("\nExtrait des resultats :")
print(data[["player", "team", "position", "minutes", "cluster"]].head(20))


# ============================================================
# 7. VISUALISATION 2D
#    Buts / 90 min vs passes decisives / 90 min
# ============================================================

x_feature = "goals_per90_calc"
y_feature = "assists_per90_calc"

plt.figure(figsize=(11, 7))

for cluster_id in sorted(data["cluster"].unique()):
    subset = data[data["cluster"] == cluster_id]

    plt.scatter(
        subset[x_feature],
        subset[y_feature],
        s=55,
        alpha=0.7,
        label=f"Cluster {cluster_id}"
    )

# Centroïdes : croix rouges avec label
for cluster_id, center in centers.iterrows():
    plt.scatter(
        center[x_feature],
        center[y_feature],
        marker="X",
        s=300,
        color="red",
        edgecolor="black",
        linewidth=1.5,
        zorder=5
    )

    plt.annotate(
        f"Centroide C{cluster_id}",
        (center[x_feature], center[y_feature]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold"
    )

plt.xlabel("Buts par 90 minutes")
plt.ylabel("Passes decisives par 90 minutes")
plt.title("Clusters de joueurs : efficacite offensive")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 8. VISUALISATION 3D
#    Buts / 90 min, passes decisives / 90 min, tirs cadres / 90 min
# ============================================================
# ============================================================
# 8. VISUALISATION 3D SANS CENTROIDES
# ============================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

x_feature = "goals_per90_calc"
y_feature = "assists_per90_calc"
z_feature = "shots_on_target_per90_calc"

for cluster_id in sorted(data["cluster"].unique()):
    subset = data[data["cluster"] == cluster_id]

    ax.scatter(
        subset[x_feature],
        subset[y_feature],
        subset[z_feature],
        s=45,
        alpha=0.7,
        label=f"Cluster {cluster_id}"
    )

ax.set_xlabel("Buts / 90 min")
ax.set_ylabel("Passes decisives / 90 min")
ax.set_zlabel("Tirs cadres / 90 min")

ax.set_title("Clustering 3D des joueurs")
ax.legend()

plt.tight_layout()
plt.show()

# ============================================================
# VISUALISATION DES CLUSTERS K-MEANS
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme(style="whitegrid", context="notebook")

# Indicateurs d'efficacité
features = [
    "goals_per90",
    "assists_per90",
    "shots_on_target_per90"
]

labels = [
    "Buts / 90 min",
    "Passes décisives / 90 min",
    "Tirs cadrés / 90 min"
]

# Préparer les données pour les graphiques
plot_data = data[features + ["cluster"]].copy()

plot_data = plot_data.dropna(subset=features + ["cluster"])

plot_data = plot_data.rename(columns=dict(zip(features, labels)))



# ============================================================
# 1. BOXPLOTS : DISTRIBUTION DES INDICATEURS PAR CLUSTER
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, feature, label in zip(axes, features, labels):

    sns.boxplot(
        data=data,
        x="cluster",
        y=feature,
        hue="cluster",
        palette="tab10",
        legend=False,
        showfliers=False,
        ax=ax
    )

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Valeur par 90 minutes")

plt.suptitle(
    "Distribution des indicateurs par cluster",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()
plt.show()


# ============================================================
# 2. HEATMAP : PROFIL MOYEN DES CLUSTERS
# ============================================================

cluster_profiles = data.groupby("cluster")[features].mean()

# Standardisation des moyennes pour comparer les indicateurs
std = cluster_profiles.std(ddof=0).replace(0, 1)

profiles_scaled = (
    cluster_profiles - cluster_profiles.mean()
) / std

plt.figure(figsize=(10, 6))

sns.heatmap(
    profiles_scaled.rename(columns=dict(zip(features, labels))),
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    center=0,
    linewidths=0.5,
    cbar_kws={"label": "Écart à la moyenne des clusters"}
)

plt.title(
    "Profil moyen des clusters",
    fontsize=15,
    fontweight="bold"
)

plt.xlabel("Indicateur")
plt.ylabel("Cluster")
plt.tight_layout()
plt.show()

print("\nProfils moyens des clusters :")
print(cluster_profiles.round(3))


# ============================================================
# 3. VISUALISATION 3D : SANS CENTROÏDES
# ============================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

palette = sns.color_palette(
    "tab10",
    n_colors=data["cluster"].nunique()
)

for color, cluster in zip(
    palette,
    sorted(data["cluster"].dropna().unique())
):

    subset = data[data["cluster"] == cluster]

    ax.scatter(
        subset["goals_per90"],
        subset["assists_per90"],
        subset["shots_on_target_per90"],
        color=color,
        label=f"Cluster {cluster}",
        s=40,
        alpha=0.75,
        depthshade=True
    )

ax.set_xlabel("Buts / 90 min", labelpad=10)
ax.set_ylabel("Passes décisives / 90 min", labelpad=10)
ax.set_zlabel("Tirs cadrés / 90 min", labelpad=10)

ax.set_title(
    "Clusters de joueurs — efficacité offensive",
    fontsize=15,
    fontweight="bold",
    pad=20
)

ax.legend(title="Clusters", loc="upper left")

plt.tight_layout()
plt.show()
# ============================================================
# DIAGRAMME RADAR : PROFIL DES CLUSTERS
# ============================================================

from math import pi

features = [
    "goals_per90",
    "assists_per90",
    "shots_on_target_per90"
]

labels = [
    "Buts / 90 min",
    "Passes décisives / 90 min",
    "Tirs cadrés / 90 min"
]

# Moyennes de chaque cluster
cluster_profiles = data.groupby("cluster")[features].mean()

# Standardisation des moyennes pour comparer les indicateurs
means = cluster_profiles.mean()
stds = cluster_profiles.std(ddof=0).replace(0, 1)

profiles_scaled = (cluster_profiles - means) / stds

# Coordonnées angulaires
N = len(labels)
angles = [i * 2 * pi / N for i in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(
    figsize=(9, 9),
    subplot_kw={"polar": True}
)

for cluster in profiles_scaled.index:
    values = profiles_scaled.loc[cluster].tolist()
    values += values[:1]

    ax.plot(
        angles,
        values,
        linewidth=2,
        label=f"Cluster {cluster}"
    )
    ax.fill(angles, values, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_title(
    "Comparaison des profils d'efficacité offensive",
    pad=25
)
ax.axhline(0, color="gray", linewidth=0.8)
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))

plt.tight_layout()
plt.show()
# ============================================================
# MEILLEURS JOUEURS DE CHAQUE CLUSTER
# ============================================================

# Seuil minimal de temps de jeu
MIN_MINUTES = 90

# Poids du score offensif (à ajuster selon ton objectif)
weights = {
    "goals_per90": 0.40,
    "assists_per90": 0.30,
    "shots_on_target_per90": 0.30
}

# Copie pour ne pas modifier les données originales
players = data.copy()

# Conversion en valeurs numériques et gestion des valeurs manquantes
stats = list(weights.keys()) + ["minutes", "player", "team", "cluster"]

for col in weights:
    players[col] = pd.to_numeric(players[col], errors="coerce")

players["minutes"] = pd.to_numeric(
    players["minutes"], errors="coerce"
)

# Exclure les joueurs sans temps de jeu suffisant
players = players[players["minutes"] >= MIN_MINUTES].copy()

# Un score synthétique fondé sur les trois indicateurs
players["performance_score"] = sum(
    players[feature].fillna(0) * weight
    for feature, weight in weights.items()
)
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Afficher les 5 meilleurs joueurs de chaque cluster
display_columns = [
    "player",
    "team",
    "cluster",
    "minutes",
    "goals_per90",
    "assists_per90",
    "shots_on_target_per90",
    "performance_score"
]

for cluster in sorted(players["cluster"].dropna().unique()):
    best_players = (
        players[players["cluster"] == cluster]
        .sort_values("performance_score", ascending=False)
        .head(15)
    )

    print(f"\n{'=' * 65}")
    print(f"CLUSTER {cluster} — TOP JOUEURS")
    print("=" * 65)

    print(
        best_players[display_columns]
        .round(3)
        .to_string(index=False)
    )
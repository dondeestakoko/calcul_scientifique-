import matplotlib.pyplot as plt
import pandas as pd

step = 0.001  # Pas de temps pour la simulation

# Fonction pour calculer l'erreur quadratique moyenne (MSE)
def mse(real_data, predicted_data):
    # Colonnes requises pour effectuer les calculs
    required_columns = ['lapin', 'renard']
    
    # Vérifier si les colonnes requises existent dans les deux datasets
    for col in required_columns:
        if col not in real_data.columns or col not in predicted_data.columns:
            raise ValueError(f"Colonne '{col}' non trouvée dans les deux jeux de données.")
    
    # Initialiser les sommes des erreurs quadratiques
    sum_squared_errors = {'lapin': 0, 'renard': 0}
    valid_count = {'lapin': 0, 'renard': 0}  # Compter les lignes valides pour chaque colonne
    
    # Parcourir les lignes des datasets
    for index in range(len(real_data)):
        for col in required_columns:
            try:
                # Convertir les valeurs en float et calculer l'erreur
                real_value = float(real_data[col].iloc[index])
                predicted_value = float(predicted_data[col].iloc[index])
                error = real_value - predicted_value
                sum_squared_errors[col] += error ** 2
                valid_count[col] += 1
            except (ValueError, TypeError):
                print(f"Valeur non numérique ignorée dans la colonne '{col}' à l'index {index}: "
                      f"{real_data[col].iloc[index]}, {predicted_data[col].iloc[index]}")
    
    # Calculer l'erreur quadratique moyenne (MSE) pour chaque colonne
    mse_values = {
        col: sum_squared_errors[col] / valid_count[col]
        for col in required_columns if valid_count[col] > 0
    }
    
    # Retourner la MSE moyenne sur toutes les colonnes
    return sum(mse_values.values()) / len(mse_values)

# Fonction de simulation des populations de lapins et de renards
def optimization(alpha, beta, delta, gamma):
    # Initialisation des listes pour le temps, les populations de lapins et de renards
    time = [0]
    lapin = [1]
    renard = [2]
    
    # Simulation sur 100 000 pas de temps   
    for _ in range(1, 100_000):
        # Calcul du prochain instant de temps
        time_update = time[-1] + step
        
        # Mise à jour de la population de lapins selon le modèle
        lapin_update = (lapin[-1] * (alpha - beta * renard[-1])) * step + lapin[-1]
        
        # Mise à jour de la population de renards selon le modèle
        renard_update = (renard[-1] * (delta * lapin[-1] - gamma)) * step + renard[-1]
        
        # Ajouter les nouvelles valeurs aux listes
        lapin.append(lapin_update)
        renard.append(renard_update)
        time.append(time_update)
    
    # Créer un DataFrame contenant les résultats
    df = pd.DataFrame({'lapin': lapin, 'renard': renard})
    return df

# Initialisation des paramètres pour trouver le meilleur ajustement
best_score = float('inf')  # Initialiser le meilleur score avec une valeur infinie
best_params = None  # Initialiser les meilleurs paramètres

# Charger les données réelles
real_data = pd.read_csv('math\calcul_scientifique-\populations_lapins_renards.csv')
real_data[['lapin', 'renard']] = real_data[['lapin', 'renard']] / 1000  # Normalisation

# Définir les plages de paramètres pour la recherche
alpha_range = [1/3, 2/3, 1, 4/3]
beta_range = [1/3, 2/3, 1, 4/3]
delta_range = [1/3, 2/3, 1, 4/3]
gamma_range = [1/3, 2/3, 1, 4/3]

# Parcourir toutes les combinaisons de paramètres
for alpha in alpha_range:
    for beta in beta_range:
        for delta in delta_range:
            for gamma in gamma_range:
                predicted = optimization(alpha, beta, delta, gamma)  # Générer les données prédites
                
                # Adapter les données prédites à la taille des données réelles
                sample_rate = len(predicted) // len(real_data)  # Ratio d'échantillonnage
                predicted_sampled = predicted.iloc[::sample_rate].reset_index(drop=True)  # Sous-échantillonnage
                
                score = mse(real_data, predicted_sampled)  # Calculer le MSE
                print(score, alpha, beta, delta, gamma)  # Afficher le score et les paramètres
                
                # Mettre à jour les meilleurs paramètres si un meilleur score est trouvé
                if score < best_score:
                    best_score = score
                    best_params = (alpha, beta, delta, gamma)

# Afficher le meilleur score et les paramètres correspondants
print(f"Meilleur score: {best_score} avec paramètres: {best_params}")

# Générer les données prédites en utilisant les meilleurs paramètres trouvés
alpha, beta, delta, gamma = best_params
predicted_data = optimization(alpha, beta, delta, gamma)

# Sous-échantillonner pour aligner avec les données réelles
sample_rate = len(predicted_data) // len(real_data)
predicted_data = predicted_data.iloc[::sample_rate].reset_index(drop=True)

# Visualisation des résultats
fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

# Graphique pour les lapins
axes[0].plot(real_data['lapin'], label='Lapins réels', color='blue', linestyle='--', marker='o')
axes[0].plot(predicted_data['lapin'], label='Lapins prédits', color='red', linestyle='-', marker='x')
axes[0].set_title('Lapins (Réel vs Prédit)')
axes[0].set_xlabel('Pas de temps')
axes[0].set_ylabel('Population')
axes[0].legend()

# Graphique pour les renards
axes[1].plot(real_data['renard'], label='Renards réels', color='blue', linestyle='--', marker='o')
axes[1].plot(predicted_data['renard'], label='Renards prédits', color='red', linestyle='-', marker='x')
axes[1].set_title('Renards (Réel vs Prédit)')
axes[1].set_xlabel('Pas de temps')
axes[1].legend()

# Afficher les graphiques
plt.tight_layout()
plt.show()

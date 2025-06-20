import numpy as np

def monte_carlo_option_price(S, K, T, r, sigma, option_type="call", num_simulations=100000):
    """
    Pricing d'une option européenne par simulation de Monte Carlo
    """
    # Génération des variables aléatoires normales
    Z = np.random.standard_normal(num_simulations)

    # Simulation des prix à maturité
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)

    # Calcul des payoffs
    if option_type == "call":
        payoffs = np.maximum(ST - K, 0)
    elif option_type == "put":
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'.")

    # Espérance et actualisation
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price

# Exemple d'utilisation
if __name__ == "__main__":
    S = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    option_type = "call"

    price = monte_carlo_option_price(S, K, T, r, sigma, option_type)
    print(f"Prix {option_type} Monte Carlo : {price:.2f} €")

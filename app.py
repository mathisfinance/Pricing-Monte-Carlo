import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def monte_carlo_option_price(S, K, T, r, sigma, option_type, num_simulations):
    Z = np.random.standard_normal(num_simulations)
    ST = S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    if option_type == "call":
        payoff = np.maximum(ST - K, 0)
    else:
        payoff = np.maximum(K - ST, 0)
    price = np.exp(-r * T) * np.mean(payoff)
    return price, ST

def black_scholes_price(S, K, T, r, sigma, option_type):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

st.title("💸 Monte Carlo Option Pricing")

S = st.number_input("Prix actuel de l'actif (S)", value=100.0)
K = st.number_input("Prix d'exercice (K)", value=100.0)
T = st.number_input("Maturité (T)", value=1.0)
r = st.number_input("Taux sans risque (r)", value=0.05)
sigma = st.number_input("Volatilité (σ)", value=0.2)
n = st.slider("Nombre de simulations Monte Carlo", 1000, 100000, 10000, step=1000)
option_type = st.selectbox("Type d'option", ["call", "put"])

if st.button("Lancer la simulation"):
    price_mc, ST = monte_carlo_option_price(S, K, T, r, sigma, option_type, n)
    price_bs = black_scholes_price(S, K, T, r, sigma, option_type)

    st.metric("Prix estimé (Monte Carlo)", f"{price_mc:.2f} €")
    st.metric("Prix théorique (Black-Scholes)", f"{price_bs:.2f} €")

    st.subheader("📊 Répartition des prix simulés à l'échéance")
    fig, ax = plt.subplots()
ax.hist(ST, bins=50, color='skyblue', edgecolor='black')
ax.set_title("Distribution des prix simulés à l’échéance")
ax.set_xlabel("Prix simulé")
ax.set_ylabel("Fréquence")
st.pyplot(fig)



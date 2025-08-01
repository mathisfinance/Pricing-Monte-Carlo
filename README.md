# Monte Carlo Option Pricing
# 🎲 Pricing d'options européennes par Monte Carlo

Ce projet implémente une méthode de **simulation de Monte Carlo** pour estimer la valeur d’une **option européenne (call ou put)**.  
C’est une approche utilisée quand les formules analytiques ne sont plus adaptées, ou pour des produits complexes.

---

## 🧠 Intuition du modèle

L'idée : générer **un grand nombre de scénarios possibles** d'évolution du prix du sous-jacent jusqu'à l'échéance, calculer le **payoff** dans chaque scénario, puis prendre la **moyenne actualisée**.

---

## 🧮 Formule utilisée

On simule les prix futurs selon une loi log-normale :

\[
S_T = S_0 \cdot \exp\left[\left(r - \frac{σ^2}{2}\right) T + σ \sqrt{T} Z\right]
\]

où :
- `Z ~ N(0,1)` : variable aléatoire normale standard
- `S_T` : prix simulé à l’échéance
- `payoff` : `max(S_T - K, 0)` pour un call, `max(K - S_T, 0)` pour un put

Puis on calcule :

\[
C = e^{-rT} \cdot \mathbb{E}[\text{payoff}]
\]

---

## 📄 Exemple d'exécution

```bash
python monte_carlo.py

"""
Application de tarification auto — prime pure d'un contrat automobile.

Cinq modèles entraînés dans `projet_tarification_auto.ipynb` sont rechargés ici :
GLM Poisson × Gamma, GLM Tweedie, XGBoost Tweedie, LightGBM Tweedie et un réseau de
neurones en deux parties. L'application affiche la prime du modèle retenu et celle des
autres modèles, et produit une grille tarifaire.

L'exposition n'est pas saisie : elle est déduite de la période de couverture, avec la
même convention qu'à l'entraînement (intervalle semi-ouvert [début, fin[).

Lancement :
    pip install -r requirements.txt
    python -m streamlit run app_tarification.py
"""

from datetime import date

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import xgboost as xgb
import lightgbm as lgb

st.set_page_config(page_title="Tarification auto — Prime pure", page_icon="🚗", layout="wide")

NUM = ["Age_assure", "Anciennete_permis", "Age_vehicule", "Power", "Value_vehicle",
       "Cylinder_capacity", "Weight", "Length", "N_doors", "Area", "Second_driver",
       "Distribution_channel", "Payment", "Seniority"]

LIB_RISK = {1: "Moto", 2: "Camionnette", 3: "Voiture", 4: "Agricole"}
LIB_FUEL = {"D": "Diesel", "P": "Essence", "Inconnu": "Inconnu"}

def _charger_xgb():
    modele = xgb.XGBRegressor()
    modele.load_model("xgb_model.json")
    return modele

@st.cache_resource
def charger():
    return {
        "glm2": joblib.load("glm2_model.pkl"),
        "glmtw": joblib.load("glmtw_model.pkl"),
        "xgb": _charger_xgb(),
        "lgb": lgb.Booster(model_file="lgb_model.txt"),
        "nn": joblib.load("nn_model.pkl"),
        "smearing": joblib.load("nn_smearing.pkl"),
        "k": joblib.load("facteurs_equilibrage.pkl"),
        "metrics": joblib.load("metrics.pkl"),
        "x_cols": joblib.load("x_cols.pkl"),
    }


def exposition_depuis_dates(date_debut, date_fin) -> float:
    """Durée sous risque en fraction d'année.

    Convention identique à l'entraînement : intervalle semi-ouvert [début, fin[ — la date
    de fin est le premier jour non couvert — d'où une simple différence de jours, sans le
    +1 du décompte inclusif. Bornée à [1 jour, 1 an] : le plancher évite un log(0) dans
    l'offset du GLM, le plafond correspond au produit annuel sur lequel les modèles sont
    calibrés.
    """
    nb_jours = (pd.Timestamp(date_fin) - pd.Timestamp(date_debut)).days
    if nb_jours <= 0:
        raise ValueError("La date de fin doit être postérieure à la date de début.")
    return float(np.clip(nb_jours / 365.25, 1 / 365.25, 1.0))


def vecteur_client(client: dict, x_cols: list) -> pd.DataFrame:
    """Encode un client comme à l'entraînement (dummies, modalités de référence omises)."""
    row = {c: 0.0 for c in x_cols}
    for var in NUM:
        row[var] = float(client[var])
    if client["Type_risk"] != 1:                      # moto = référence
        col = f"Type_risk_{int(client['Type_risk'])}"
        if col in row:
            row[col] = 1.0
    if client["Type_fuel"] != "D":                    # diesel = référence
        col = f"Type_fuel_{client['Type_fuel']}"
        if col in row:
            row[col] = 1.0
    return pd.DataFrame([row])[x_cols]


def primes_tous_modeles(client: dict, exposition: float, M: dict) -> dict:
    """Prime pure prédite par chacun des cinq modèles, en euros sur la période."""
    X = vecteur_client(client, M["x_cols"])
    Xc = sm.add_constant(X, has_constant="add")
    k = M["k"]
    log_e = np.log(exposition)

    freq = M["glm2"]["freq"].predict(Xc, offset=log_e).iloc[0]
    cout = M["glm2"]["sev"].predict(Xc).iloc[0]
    p_nn = M["nn"]["clf"].predict_proba(X)[0, 1]
    cout_nn = float(np.exp(M["nn"]["reg"].predict(X)[0])) * M["smearing"]

    return {
        "GLM Poisson x Gamma": freq * cout * k["GLM Poisson x Gamma"],
        "GLM Tweedie": M["glmtw"].predict(Xc, offset=log_e).iloc[0] * k["GLM Tweedie"],
        "XGBoost Tweedie": float(M["xgb"].predict(X)[0]) * exposition * k["XGBoost Tweedie"],
        "LightGBM Tweedie": float(M["lgb"].predict(X)[0]) * exposition * k["LightGBM Tweedie"],
        "Reseau de neurones": p_nn * cout_nn * exposition * k["Reseau de neurones"],
    }, {"frequence": freq, "cout_moyen": cout}


def formulaire_risque(prefixe: str, defauts: dict) -> dict:
    """Saisie des caractéristiques du risque, réutilisée par plusieurs onglets."""
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Âge de l'assuré", 18, 95, defauts["Age_assure"], key=f"{prefixe}_age")
        permis = st.number_input("Ancienneté du permis (années)", 0, 75,
                                 defauts["Anciennete_permis"], key=f"{prefixe}_permis")
        anc = st.number_input("Ancienneté client (années)", 0, 60, defauts["Seniority"],
                              key=f"{prefixe}_anc")
        zone = st.radio("Zone", [0, 1], index=defauts["Area"], horizontal=True,
                        format_func=lambda x: "Rural" if x == 0 else "Urbain", key=f"{prefixe}_zone")
        second = st.radio("Second conducteur", [0, 1], index=defauts["Second_driver"], horizontal=True,
                          format_func=lambda x: "Non" if x == 0 else "Oui", key=f"{prefixe}_second")
        canal = st.radio("Canal de distribution", [0, 1], index=defauts["Distribution_channel"],
                         horizontal=True, format_func=lambda x: "Agent" if x == 0 else "Courtier",
                         key=f"{prefixe}_canal")
        paiement = st.radio("Mode de paiement", [0, 1], index=defauts["Payment"], horizontal=True,
                            format_func=lambda x: "Annuel" if x == 0 else "Semestriel",
                            key=f"{prefixe}_paiement")
    with c2:
        risque = st.selectbox("Type de véhicule", [1, 2, 3, 4],
                              index=[1, 2, 3, 4].index(defauts["Type_risk"]),
                              format_func=lambda x: LIB_RISK[x], key=f"{prefixe}_risque")
        carburant = st.selectbox("Carburant", ["D", "P", "Inconnu"],
                                 index=["D", "P", "Inconnu"].index(defauts["Type_fuel"]),
                                 format_func=lambda x: LIB_FUEL[x], key=f"{prefixe}_fuel")
        age_veh = st.number_input("Âge du véhicule (années)", 0, 40, defauts["Age_vehicule"],
                                  key=f"{prefixe}_ageveh")
        puissance = st.number_input("Puissance (ch)", 0, 500, defauts["Power"], key=f"{prefixe}_pow")
        valeur = st.number_input("Valeur du véhicule (€)", 0, 300000, defauts["Value_vehicle"],
                                 step=500, key=f"{prefixe}_val")
        cylindree = st.number_input("Cylindrée (cm³)", 0, 8000, defauts["Cylinder_capacity"],
                                    step=50, key=f"{prefixe}_cyl")
        poids = st.number_input("Poids (kg)", 0, 5000, defauts["Weight"], step=10, key=f"{prefixe}_poids")
        longueur = st.number_input("Longueur (m)", 0.0, 10.0, defauts["Length"], step=0.1,
                                   key=f"{prefixe}_len")
        portes = st.number_input("Nombre de portes", 0, 6, defauts["N_doors"], key=f"{prefixe}_portes")

    return {"Age_assure": age, "Anciennete_permis": permis, "Age_vehicule": age_veh,
            "Power": puissance, "Value_vehicle": valeur, "Cylinder_capacity": cylindree,
            "Weight": poids, "Length": longueur, "N_doors": portes, "Type_risk": risque,
            "Area": zone, "Second_driver": second, "Distribution_channel": canal,
            "Payment": paiement, "Type_fuel": carburant, "Seniority": anc}


# Profil de référence : médianes et modalités dominantes du portefeuille
REFERENCE = {"Age_assure": 47, "Anciennete_permis": 24, "Age_vehicule": 12, "Power": 90,
             "Value_vehicle": 14000, "Cylinder_capacity": 1600, "Weight": 1200, "Length": 4.2,
             "N_doors": 5, "Type_risk": 3, "Area": 1, "Second_driver": 0,
             "Distribution_channel": 0, "Payment": 0, "Type_fuel": "D", "Seniority": 5}

st.title("🚗 Tarification automobile — prime pure")

try:
    M = charger()
except FileNotFoundError as e:
    st.error(f"Modèle introuvable ({e.filename}). Les fichiers .pkl doivent se trouver "
             "dans le même dossier que ce script ; ils sont générés par le notebook.")
    st.stop()

MEILLEUR = M["metrics"]["meilleur"]
COMPARAISON = pd.DataFrame(M["metrics"]["comparaison"])
st.caption(f"Modèle retenu : **{MEILLEUR}** — sélectionné sur la déviance Tweedie hors échantillon.")

onglet_tarif, onglet_grille, onglet_modeles = st.tabs(
    ["Tarification d'un contrat", "Grille tarifaire", "Comparaison des modèles"])

# ---------------------------------------------------------------- onglet 1
with onglet_tarif:
    with st.form("form_tarif"):
        st.subheader("Période de couverture")
        d1, d2 = st.columns(2)
        debut = d1.date_input("Date d'effet", date(2026, 1, 1), format="DD/MM/YYYY")
        fin = d2.date_input("Date d'échéance", date(2027, 1, 1), format="DD/MM/YYYY")
        st.caption("L'exposition est déduite de ces deux dates : la date d'échéance est le "
                   "premier jour non couvert, et l'exposition est plafonnée à un an.")

        st.subheader("Caractéristiques du risque")
        client = formulaire_risque("t", REFERENCE)
        calcul = st.form_submit_button("Calculer la prime pure", type="primary")

    if calcul:
        try:
            expo = exposition_depuis_dates(debut, fin)
        except ValueError as err:
            st.error(str(err))
            st.stop()

        nb_jours = (pd.Timestamp(fin) - pd.Timestamp(debut)).days
        if nb_jours > 366:
            st.warning(f"Période de {nb_jours} jours : les modèles sont calibrés sur des "
                       "annuités, l'exposition est donc plafonnée à un an.")

        toutes, detail = primes_tous_modeles(client, expo, M)

        st.info(f"Couverture du {debut:%d/%m/%Y} au {fin:%d/%m/%Y} — {nb_jours} jours, "
                f"soit une exposition de **{expo:.4f}** année.")

        c1, c2, c3 = st.columns([2, 1, 1])
        c1.metric(f"Prime pure — {MEILLEUR}", f"{toutes[MEILLEUR]:.2f} €")
        c2.metric("Fréquence (GLM)", f"{detail['frequence']:.4f}")
        c3.metric("Coût moyen (GLM)", f"{detail['cout_moyen']:.0f} €")

        st.subheader("Prime selon chaque modèle")
        tab = pd.DataFrame({"Modèle": list(toutes), "Prime pure (€)": [round(v, 2) for v in toutes.values()]})
        tab["Écart au modèle retenu"] = (tab["Prime pure (€)"] / toutes[MEILLEUR] - 1).map("{:+.1%}".format)
        tab["Retenu"] = np.where(tab["Modèle"] == MEILLEUR, "✅", "")
        st.dataframe(tab, hide_index=True)
        st.bar_chart(tab.set_index("Modèle")["Prime pure (€)"])

        st.caption("Prime pure = espérance de charge de sinistres sur la période. Hors "
                   "chargements (frais de gestion, commissions, marge, taxes) : ce n'est pas "
                   "un tarif commercial. La dispersion entre modèles mesure le risque de "
                   "modèle sur ce profil.")

# ---------------------------------------------------------------- onglet 2
with onglet_grille:
    st.subheader("Profil de référence")
    st.caption("La grille exprime, pour chaque modalité, le rapport entre la prime du profil "
               "modifié et celle du profil de référence ci-dessous (relativité). Une "
               "relativité de 1,20 signifie « +20 % par rapport au tarif de base ».")
    with st.expander("Modifier le profil de référence"):
        reference = formulaire_risque("g", REFERENCE)

    expo_ref = 1.0
    base, _ = primes_tous_modeles(reference, expo_ref, M)
    st.metric(f"Tarif de base — {MEILLEUR}, contrat annuel", f"{base[MEILLEUR]:.2f} €")
    st.caption(" · ".join(f"{m} : {v:.2f} €" for m, v in base.items() if m != MEILLEUR))

    VARIATIONS = {
        "Age_assure": [20, 25, 30, 40, 50, 60, 70, 80],
        "Anciennete_permis": [1, 3, 5, 10, 20, 30, 40],
        "Age_vehicule": [0, 2, 5, 10, 15, 20, 25],
        "Power": [50, 70, 90, 110, 150, 200],
        "Value_vehicle": [5000, 10000, 14000, 20000, 30000, 50000],
        "Seniority": [0, 1, 3, 5, 10, 20],
        "Type_risk": [1, 2, 3, 4],
        "Type_fuel": ["D", "P", "Inconnu"],
        "Area": [0, 1],
        "Second_driver": [0, 1],
        "Payment": [0, 1],
        "Distribution_channel": [0, 1],
    }
    ETIQ = {"Area": {0: "Rural", 1: "Urbain"}, "Second_driver": {0: "Non", 1: "Oui"},
            "Payment": {0: "Annuel", 1: "Semestriel"},
            "Distribution_channel": {0: "Agent", 1: "Courtier"},
            "Type_risk": LIB_RISK, "Type_fuel": LIB_FUEL}

    if st.button("Générer la grille tarifaire", type="primary"):
        lignes = []
        barre = st.progress(0.0)
        total = sum(len(v) for v in VARIATIONS.values())
        fait = 0
        for var, valeurs in VARIATIONS.items():
            for val in valeurs:
                profil = dict(reference)
                profil[var] = val
                pr, _ = primes_tous_modeles(profil, expo_ref, M)
                lignes.append({
                    "Variable": var,
                    "Modalité": str(ETIQ.get(var, {}).get(val, val)),
                    f"Prime {MEILLEUR} (€)": round(pr[MEILLEUR], 2),
                    "Relativité": round(pr[MEILLEUR] / base[MEILLEUR], 3),
                    **{f"{m} (€)": round(v, 2) for m, v in pr.items() if m != MEILLEUR},
                })
                fait += 1
                barre.progress(fait / total)
        barre.empty()
        grille = pd.DataFrame(lignes)
        st.session_state["grille"] = grille

    if "grille" in st.session_state:
        grille = st.session_state["grille"]
        st.dataframe(grille, hide_index=True, height=420)
        st.download_button("Télécharger la grille (CSV)",
                           grille.to_csv(index=False).encode("utf-8"),
                           "grille_tarifaire.csv", "text/csv")
        st.caption("Chaque ligne fait varier une seule variable, toutes les autres restant au "
                   "profil de référence. Les relativités d'un modèle non linéaire ne sont donc "
                   "valables qu'au voisinage de ce profil : contrairement à un GLM log-linéaire, "
                   "elles ne se multiplient pas entre elles.")

# ---------------------------------------------------------------- onglet 3
with onglet_modeles:
    st.subheader("Performances hors échantillon")
    aff = COMPARAISON.rename(columns={"modele": "Modèle", "deviance_tweedie": "Déviance Tweedie",
                                      "gini": "Gini", "prime_moyenne": "Prime moyenne (€)",
                                      "ratio_calibration": "Ratio de calibration"})
    aff["Retenu"] = np.where(aff["Modèle"] == MEILLEUR, "✅", "")
    st.dataframe(aff, hide_index=True)

    st.markdown(f"""
**Critère de sélection.** La déviance Tweedie (`p = {M['metrics']['p_tweedie']}`) mesure l'écart
entre charge prédite et charge observée sur l'échantillon test — plus elle est faible, mieux
c'est. Le Gini mesure une autre qualité : la capacité à **ordonner** les risques, indépendamment
du niveau. Le découpage train/test est fait **par contrat**, de sorte qu'aucune annuité d'un même
contrat ne se retrouve des deux côtés.

**Les cinq modèles.**

- *GLM Poisson × Gamma* — approche actuarielle classique en deux temps : fréquence des sinistres
  (Poisson, avec `log(exposition)` en offset) puis coût moyen par sinistre (Gamma). Le seul modèle
  dont chaque coefficient s'interprète comme un effet multiplicatif.
- *GLM Tweedie* — un seul modèle sur la charge annuelle, la loi de Tweedie combinant une masse en
  zéro et une queue continue.
- *XGBoost / LightGBM Tweedie* — gradient boosting avec la même perte, capables de capter les
  interactions et les effets non linéaires que le GLM impose de spécifier à la main.
- *Réseau de neurones en deux parties* — un perceptron classe la survenance d'un sinistre, un
  second estime son coût ; le produit donne la prime pure. Le retour à l'échelle est corrigé par
  un facteur de *smearing* de Duan.

**Équilibrage.** La prédiction de chaque modèle est multipliée par un facteur estimé sur
l'apprentissage, qui aligne la charge totale prédite sur la charge observée. Sans lui, la
comparaison mélangerait deux défauts distincts : un mauvais niveau global et un mauvais classement
des risques.
""")

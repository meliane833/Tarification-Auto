"""
Application de tarification auto — prime pure d'un contrat automobile.

Cinq modèles entraînés dans `projet_tarification_auto.ipynb` sont rechargés ici :
GLM Poisson × Gamma, GLM Tweedie, XGBoost Tweedie, LightGBM Tweedie et un réseau de
neurones en deux parties. L'application affiche la prime du modèle retenu et celle des
quatre autres.

L'exposition n'est pas saisie : elle est déduite de la période de couverture, avec la
même convention qu'à l'entraînement (intervalle semi-ouvert [début, fin[).

Lancement :
    pip install -r requirements.txt
    python -m streamlit run app_tarification.py
"""

from datetime import date

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import streamlit as st
import xgboost as xgb

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
        raise ValueError("La date d'échéance doit être postérieure à la date d'effet.")
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


def primes_tous_modeles(client: dict, exposition: float, M: dict):
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


# Valeurs par défaut : profil médian du portefeuille
DEFAUTS = {"Age_assure": 47, "Anciennete_permis": 24, "Age_vehicule": 12, "Power": 90,
           "Value_vehicle": 18000, "Cylinder_capacity": 1600, "Weight": 1200, "Length": 4.2,
           "N_doors": 5, "Type_risk": 3, "Area": 0, "Second_driver": 0,
           "Distribution_channel": 0, "Payment": 0, "Type_fuel": "D", "Seniority": 5}

st.title("🚗 Tarification automobile — prime pure")

try:
    M = charger()
except FileNotFoundError as e:
    st.error(f"Modèle introuvable ({e.filename}). Les fichiers de modèles doivent se trouver "
             "dans le même dossier que ce script ; ils sont générés par le notebook.")
    st.stop()

MEILLEUR = M["metrics"]["meilleur"]
st.caption(f"Modèle retenu : **{MEILLEUR}** — sélectionné sur la déviance Tweedie hors échantillon.")

# ---------------------------------------------------------------- barre latérale
with st.sidebar:
    st.subheader("Période de couverture")
    debut = st.date_input("Date d'effet", date(2026, 1, 1), format="DD/MM/YYYY")
    fin = st.date_input("Date d'échéance", date(2027, 1, 1), format="DD/MM/YYYY")
    try:
        expo = exposition_depuis_dates(debut, fin)
        nb_jours = (pd.Timestamp(fin) - pd.Timestamp(debut)).days
        st.caption(f"{nb_jours} jours → exposition **{expo:.4f}** année")
        if nb_jours > 366:
            st.caption("Période supérieure à un an : exposition plafonnée à 1.")
    except ValueError as err:
        st.error(str(err))
        st.stop()

# ---------------------------------------------------------------- formulaire
with st.form("form_tarif"):
    st.subheader("Caractéristiques du risque")
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("Âge de l'assuré", 18, 95, DEFAUTS["Age_assure"])
        permis = st.number_input("Ancienneté du permis (années)", 0, 75, DEFAUTS["Anciennete_permis"])
        anc = st.number_input("Ancienneté client (années)", 0, 60, DEFAUTS["Seniority"])
        zone = st.radio("Zone", [0, 1], index=DEFAUTS["Area"], horizontal=True,
                        format_func=lambda x: "Rural" if x == 0 else "Urbain")
        second = st.radio("Second conducteur", [0, 1], index=DEFAUTS["Second_driver"],
                          horizontal=True, format_func=lambda x: "Non" if x == 0 else "Oui")
        canal = st.radio("Canal de distribution", [0, 1], index=DEFAUTS["Distribution_channel"],
                         horizontal=True, format_func=lambda x: "Agent" if x == 0 else "Courtier")
        paiement = st.radio("Mode de paiement", [0, 1], index=DEFAUTS["Payment"], horizontal=True,
                            format_func=lambda x: "Annuel" if x == 0 else "Semestriel")
    with c2:
        risque = st.selectbox("Type de véhicule", [1, 2, 3, 4], index=2,
                              format_func=lambda x: LIB_RISK[x])
        carburant = st.selectbox("Carburant", ["D", "P", "Inconnu"], index=0,
                                 format_func=lambda x: LIB_FUEL[x])
        age_veh = st.number_input("Âge du véhicule (années)", 0, 40, DEFAUTS["Age_vehicule"])
        puissance = st.number_input("Puissance (ch)", 0, 500, DEFAUTS["Power"])
        valeur = st.number_input("Valeur du véhicule (€)", 0, 300000, DEFAUTS["Value_vehicle"], step=500)
        cylindree = st.number_input("Cylindrée (cm³)", 0, 8000, DEFAUTS["Cylinder_capacity"], step=50)
        poids = st.number_input("Poids (kg)", 0, 5000, DEFAUTS["Weight"], step=10)
        longueur = st.number_input("Longueur (m)", 0.0, 10.0, DEFAUTS["Length"], step=0.1)
        portes = st.number_input("Nombre de portes", 0, 6, DEFAUTS["N_doors"])

    calcul = st.form_submit_button("Calculer la prime pure", type="primary")

# ---------------------------------------------------------------- résultats
if calcul:
    client = {"Age_assure": age, "Anciennete_permis": permis, "Age_vehicule": age_veh,
              "Power": puissance, "Value_vehicle": valeur, "Cylinder_capacity": cylindree,
              "Weight": poids, "Length": longueur, "N_doors": portes, "Type_risk": risque,
              "Area": zone, "Second_driver": second, "Distribution_channel": canal,
              "Payment": paiement, "Type_fuel": carburant, "Seniority": anc}

    toutes, detail = primes_tous_modeles(client, expo, M)

    st.info(f"Couverture du {debut:%d/%m/%Y} au {fin:%d/%m/%Y} — {nb_jours} jours, "
            f"soit une exposition de **{expo:.4f}** année.")

    r1, r2, r3 = st.columns([2, 1, 1])
    r1.metric(f"Prime pure — {MEILLEUR}", f"{toutes[MEILLEUR]:.2f} €")
    r2.metric("Fréquence (GLM)", f"{detail['frequence']:.4f}")
    r3.metric("Coût moyen (GLM)", f"{detail['cout_moyen']:.0f} €")

    st.subheader("Prime selon chaque modèle")
    tab = pd.DataFrame({"Modèle": list(toutes),
                        "Prime pure (€)": [round(v, 2) for v in toutes.values()]})
    tab["Écart au modèle retenu"] = (tab["Prime pure (€)"] / toutes[MEILLEUR] - 1).map("{:+.1%}".format)
    tab["Retenu"] = np.where(tab["Modèle"] == MEILLEUR, "✅", "")
    st.dataframe(tab, hide_index=True)

    st.caption("Prime pure = espérance de charge de sinistres sur la période. Hors chargements "
               "(frais de gestion, commissions, marge, taxes) : ce n'est pas un tarif commercial. "
               "La dispersion entre modèles mesure le risque de modèle sur ce profil.")

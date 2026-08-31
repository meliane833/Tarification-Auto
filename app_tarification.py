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
from fpdf import FPDF
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


def fiche_client(client: dict, debut, fin, exposition: float, prime: float) -> bytes:
    """Fiche d'une page destinée au client, au format PDF."""
    oui_non = {0: "Non", 1: "Oui"}
    nb_jours = (pd.Timestamp(fin) - pd.Timestamp(debut)).days

    periode = [
        ("Date d'effet", f"{debut:%d/%m/%Y}"),
        ("Date d'échéance", f"{fin:%d/%m/%Y}"),
        ("Durée", f"{nb_jours} jours ({exposition:.3f} année)"),
    ]
    risque = [
        ("Âge de l'assuré", f"{client['Age_assure']} ans"),
        ("Ancienneté du permis", f"{client['Anciennete_permis']} ans"),
        ("Ancienneté client", f"{client['Seniority']} ans"),
        ("Zone de circulation", "Urbain" if client["Area"] == 1 else "Rural"),
        ("Second conducteur", oui_non[client["Second_driver"]]),
        ("Canal de souscription", "Courtier" if client["Distribution_channel"] == 1 else "Agent"),
        ("Mode de paiement", "Semestriel" if client["Payment"] == 1 else "Annuel"),
        ("Type de véhicule", LIB_RISK[client["Type_risk"]]),
        ("Carburant", LIB_FUEL[client["Type_fuel"]]),
        ("Âge du véhicule", f"{client['Age_vehicule']} ans"),
        ("Puissance", f"{client['Power']} ch"),
        ("Cylindrée", f"{client['Cylinder_capacity']} cm3"),
        ("Valeur du véhicule", f"{client['Value_vehicle']:,.0f} EUR".replace(",", " ")),
        ("Poids", f"{client['Weight']} kg"),
        ("Longueur", f"{client['Length']} m"),
        ("Nombre de portes", str(client["N_doors"])),
    ]

    pdf = FPDF(format="A4", unit="mm")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(20, 18, 20)
    largeur = pdf.w - 40

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(largeur, 9, "Proposition tarifaire - assurance automobile", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(110, 118, 130)
    pdf.cell(largeur, 6, f"Éditée le {date.today():%d/%m/%Y}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    def section(titre, lignes):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(110, 118, 130)
        pdf.cell(largeur, 6, titre.upper(), new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(210, 215, 225)
        pdf.line(20, pdf.get_y(), 20 + largeur, pdf.get_y())
        pdf.ln(2)
        pdf.set_text_color(0, 0, 0)
        for intitule, valeur in lignes:
            pdf.set_font("Helvetica", "", 10)
            pdf.cell(largeur * 0.6, 7, intitule)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(largeur * 0.4, 7, str(valeur), align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    section("Période de couverture", periode)

    pdf.set_draw_color(30, 39, 97)
    pdf.set_line_width(0.6)
    y0 = pdf.get_y()
    pdf.rect(20, y0, largeur, 18)
    pdf.set_xy(24, y0 + 5)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 39, 97)
    pdf.cell(largeur * 0.55, 8, "Prime pure pour la période")
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(largeur * 0.4, 8, f"{prime:,.2f} EUR".replace(",", " "), align="R")
    pdf.set_text_color(0, 0, 0)
    pdf.set_line_width(0.2)
    pdf.set_xy(20, y0 + 24)

    section("Caractéristiques du risque", risque)

    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_text_color(110, 118, 130)
    pdf.multi_cell(largeur, 4,
                   "La prime indiquée est une prime pure : l'espérance de la charge de sinistres "
                   "sur la période de couverture. Elle n'intègre ni frais de gestion, ni "
                   "commissions, ni marge, ni taxes, et ne constitue donc pas un tarif commercial. "
                   "Estimation produite par un modèle statistique calibré sur un portefeuille "
                   "historique ; elle ne vaut pas engagement de garantie.")

    return bytes(pdf.output())


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
    tab["Écart au modèle retenu"] = np.where(
        tab["Modèle"] == MEILLEUR, "—",
        (tab["Prime pure (€)"] / toutes[MEILLEUR] - 1).map("{:+.1%}".format))

    # deviance hors echantillon, mesuree une fois pour toutes a l'entrainement
    perf = pd.DataFrame(M["metrics"]["comparaison"])[["modele", "deviance_tweedie"]]
    perf.columns = ["Modèle", "Déviance (test)"]
    tab = tab.merge(perf, on="Modèle", how="left")
    tab["Retenu"] = np.where(tab["Modèle"] == MEILLEUR, "✅", "")
    st.dataframe(tab, hide_index=True)

    st.caption("La **déviance** mesure l'écart entre charge prédite et charge réellement observée, "
               "sur les 21 170 annuités de test — des contrats absents de l'apprentissage. Plus "
               "elle est faible, meilleur est le modèle ; c'est ce critère qui a désigné le modèle "
               "retenu. Elle porte sur l'ensemble du portefeuille de test, pas sur ce client-ci : "
               "pour un contrat isolé, on ne connaîtra son coût réel qu'à la fin de la période.")

    st.caption("Prime pure = espérance de charge de sinistres sur la période. Hors chargements "
               "(frais de gestion, commissions, marge, taxes) : ce n'est pas un tarif commercial. "
               "La dispersion entre modèles mesure le risque de modèle sur ce profil.")

    # ------------------------------------------------------------ export client
    st.divider()
    st.download_button(
        "📄 Exporter la fiche client (PDF)",
        data=fiche_client(client, debut, fin, expo, toutes[MEILLEUR]),
        file_name=f"proposition_tarifaire_{debut:%Y%m%d}.pdf",
        mime="application/pdf",
    )
    st.caption("Fiche d'une page reprenant les caractéristiques saisies et la prime retenue.")

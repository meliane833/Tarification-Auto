# Tarification auto — prime pure par GLM, boosting et réseau de neurones

Projet de Data Science, M1 Actuariat. Estimation de la prime pure d'un contrat automobile à partir
d'un portefeuille réel, comparaison de cinq modèles, et mise à disposition du tarif via une
application Streamlit.

**Application en ligne :** _(à compléter avec l'URL Streamlit après déploiement)_

## Données

*Dataset of an actual motor vehicle insurance portfolio* (Mendeley — Lledó & Pavía, Universitat de
València). 105 555 lignes et 30 variables, correspondant à **53 502 contrats** suivis par annuités :
chaque ligne est une période annuelle de couverture, délimitée par `Date_last_renewal` et
`Date_next_renewal`. Les périodes observées débutent entre le 02/11/2015 et le 30/11/2018 et
s'achèvent au plus tard le 30/11/2019. Le fichier de données n'est pas versionné ici.

## Les cinq modèles

Tous estiment la même quantité — l'espérance de charge annuelle — avec le même traitement de
l'exposition et le même équilibrage.

| Modèle | Principe |
|---|---|
| GLM Poisson × Gamma | Fréquence (offset `log(exposition)`) × coût moyen conditionnel |
| GLM Tweedie | Charge annuelle directe, `p = 1.5` |
| XGBoost Tweedie | Même perte, non-linéarités et interactions apprises |
| LightGBM Tweedie | Idem, croissance des arbres par feuilles |
| Réseau de neurones | Deux parties : survenance (MLPClassifier) × coût (MLPRegressor), correction de smearing |

Le modèle retenu par l'application est celui de plus faible déviance Tweedie hors échantillon. Les
quatre autres restent affichés : l'écart entre leurs primes sur un même profil mesure le risque de
modèle.

## Points méthodologiques

**Calcul de l'exposition.** `Date_lapse` est un attribut du contrat, recopié sur toutes ses
annuités : on ne peut donc pas se fier à `Lapse`, qui compte les résiliations du *client* dans
l'année, toutes polices confondues. Trois cas selon la position de `Date_lapse` par rapport à la
période de la ligne :

- postérieure → contrat actif toute la période, exposition = 1 an
- à l'intérieur → résiliation en cours d'année, exposition tronquée (3 523 lignes)
- antérieure ou égale au début → échéance générée mais contrat non reconduit : la période n'a
  jamais été couverte (14 836 lignes, 14 % de la base, dont 99,6 % sans sinistre)

Ce dernier groupe reçoit une exposition plancher d'un jour, `log(0)` étant impossible dans l'offset
du GLM. Leur attribuer une année pleine ferait passer la fréquence du portefeuille de 0,468 à 0,402
sinistre par année-police et sous-tariferait la prime d'autant. Test de sensibilité : supprimer ces
lignes donne 0,4677, les conserver au plancher 0,4683 — le traitement n'oriente pas les résultats.

L'intervalle `[Date_last_renewal, Date_next_renewal[` est semi-ouvert (la date de fin est le premier
jour de l'annuité suivante), d'où une durée calculée par simple différence, sans le `+1` du décompte
inclusif. L'application applique la même convention à la période saisie par l'utilisateur.

**Découpage train/test par contrat.** Les annuités d'un même contrat ne sont pas indépendantes. Un
découpage aléatoire par ligne placerait l'annuité 2016 en apprentissage et l'annuité 2017 en test
pour le même contrat — une fuite d'information. Le découpage est fait par `ID`
(`GroupShuffleSplit`), et les écarts-types du GLM de fréquence sont clusterisés sur la même
variable.

**Équilibrage.** Les prédictions de chaque modèle sont multipliées par le rapport entre charge
observée et charge prédite sur l'apprentissage. Sans cette correction, la comparaison mélangerait un
mauvais niveau global et un mauvais classement des risques.

**Nettoyage.** `Distribution_channel` contient `00/01/1900` sur 3 416 lignes — le jour 0 du
calendrier Excel, c'est-à-dire la valeur `0` affichée au format date ; ces valeurs sont ramenées à 0.
`Length` est absente pour 100 % des motos : elle est reconstituée à partir de la cylindrée via les
classes réglementaires deux-roues, les autres catégories étant imputées par la médiane de leur
`Type_risk`. `Type_fuel` manquant devient une modalité `"Inconnu"`.

## Limites connues

- 404 lignes portent une date de résiliation strictement antérieure à leur période, et 75 sinistres
  sont déclarés sur des périodes postérieures à la résiliation — incohérences de saisie non
  expliquées.
- `Value_vehicle` est évaluée au 31/12/2019, date postérieure à toutes les périodes observées.
- `N_claims_history` et `R_Claims_history` sont constantes au sein d'un contrat et intègrent les
  sinistres des années futures : elles sont volontairement écartées (fuite d'information).
- Le paramètre `p` du Tweedie est fixé à 1,5 sans recherche par grille.
- Le coefficient de `Type_risk_4` (véhicules agricoles, ~700 observations en apprentissage) est
  aberrant et appelle un regroupement de modalité.
- Les relativités d'un modèle à base d'arbres ne valent qu'au voisinage du profil de référence :
  contrairement à un GLM log-linéaire, elles ne se multiplient pas entre elles.

## Contenu du dépôt

```
app_tarification.py        application Streamlit (3 onglets)
requirements.txt           dépendances, versions figées pour la compatibilité des modèles sérialisés
glm2_model.pkl             GLM Poisson (fréquence) + GLM Gamma (sévérité)
glmtw_model.pkl            GLM Tweedie
xgb_model.pkl              XGBoost Tweedie
lgb_model.pkl              LightGBM Tweedie
nn_model.pkl               réseau de neurones en deux parties
nn_smearing.pkl            facteur de correction de Duan
facteurs_equilibrage.pkl   coefficients d'équilibrage par modèle
x_cols.pkl                 ordre des colonnes attendu par les modèles
metrics.pkl                comparaison des modèles et modèle retenu
```

Le notebook d'analyse et d'entraînement (`projet_tarification_auto.ipynb`) est fourni séparément ;
c'est lui qui produit tous les `.pkl` ci-dessus, dans sa dernière section.

## L'application

Trois onglets :

1. **Tarification d'un contrat** — l'utilisateur saisit une *période de couverture* et les
   caractéristiques du risque ; l'exposition est calculée par l'application. La prime du modèle
   retenu est affichée aux côtés de celles des quatre autres.
2. **Grille tarifaire** — tarif de base d'un profil de référence modifiable, puis relativité de
   chaque modalité, exportable en CSV.
3. **Comparaison des modèles** — déviances, Gini, ratios de calibration et méthodologie.

En local :

```bash
pip install -r requirements.txt
python -m streamlit run app_tarification.py
```

Les fichiers `.pkl` doivent se trouver dans le même dossier que le script.

> **Compatibilité des versions.** Les `.pkl` sont sérialisés avec les versions de bibliothèques de
> l'environnement d'entraînement. Si le déploiement échoue au chargement, aligner
> `requirements.txt` sur les versions affichées par le notebook après exécution.

## Avertissement

Les primes affichées sont des **primes pures** — l'espérance de charge de sinistres sur la période.
Elles n'incluent aucun chargement (frais de gestion, commissions, marge, taxes) et ne constituent
donc pas un tarif commercial : la prime commerciale moyenne de ce portefeuille, 315 €, en est
environ le double. Les modèles sont calibrés sur un portefeuille espagnol de 2015-2019 et n'ont pas
vocation à être transposés tels quels à un autre marché.

## Auteur

WOGNIN Méliane — M1 Actuariat

# Module IA

Module responsable de la prédiction de nouveaux rapports de propagation de pandémies.

## Installation

Il est conseillé d'activer un environnement virtuel (c'est également nécessaire sur certaines plateformes). Pour plus d'informations, Consultez la [documentation](https://docs.python.org/3/library/venv.html)

```bash
pip install -r requirements.txt
```

Il faut ensuite configurer le fichier `.env` à la racine du projet. Un exemple de fichier `.env.sample` est fourni.

## Entrainement

Pour entrainer le modèle, il faut lancer la commande suivante :

```bash
python . train
```

À la fin de l'entrainement, le modèle sera sauvegardé dans le dossier `models/`. Le score de l'entrainement et le MSE de chaque résultat seront affichés dans la console.

## Lancement de l'API

Pour lancer l'API, il faut lancer la commande suivante :

```bash
python . serve
```

## Endpoints de l'API

Une documentation OpenAPI est disponible à l'adresse `/docs` après le lancement de l'API.

### GET `/`

Headers:
- Authorization: `Bearer <token>`

Indique le statut de l'API.

### POST `/predict`

Headers:
- Authorization: `Bearer <token>`

Body:
```json
{
  "pandemic_name": "name",
  "pandemic_pathogen": "pathogen",
  "country": "country",
  "continent": "continent",
  "reports": [
    {
      "date": "YYYY-MM-DD",
      "new_cases": 100,
      "new_deaths": 10
    },
    ... * 100 (max)
  ],
  "target": {
    "date": "YYYY-MM-DD",
  }
}
```

Response:
```json
{
  "new_cases": 150,
  "new_deaths": 15,
}
```

Prédit un nouveau rapport de propagation de pandémie basée sur les données d'entrainement et les données fournies.

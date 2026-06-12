# WidowX ACT Learning - Guide francais

Ce projet contient des outils locaux pour controler un bras Trossen Robotics WidowX AI, enregistrer des demonstrations, entrainer un petit modele ACT-style et tester un checkpoint avec des garde-fous.

Documentation anglaise: [readme_en.md](readme_en.md)

Liens de recherche et papiers telecharges: [research/README.md](research/README.md)

## Structure

```text
widowx_ai/
├── apps/       # Interface web locale
├── config/     # Configuration locale du robot
├── policies/   # Replay securise de politiques
├── robot/      # Outils de connexion et mouvement
├── tools/      # Verification des datasets
└── training/   # Entrainement ACT et dashboard
```

Les donnees lourdes restent hors Git:

- `widowx_ai/recordings*/`
- `widowx_ai/models/`
- `.venv*/`
- `*.pdf`

## Installation

Depuis la racine du projet:

```bash
.venv-trossen-ui/bin/python -m pip install -r requirements.txt
```

Verifier que le driver Trossen est disponible:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.widowx_demo --check-import
```

## Connexion au bras

Configuration reseau typique du controleur:

- IP du bras: `192.168.1.2`
- IP du PC: `192.168.1.1`
- Masque: `255.255.255.0`

Verifier la connexion:

```bash
ping 192.168.1.2
```

Lire l'etat du bras sans mouvement:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.widowx_demo --ip 192.168.1.2 --variant base
```

Tester un petit mouvement uniquement si la zone est libre:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.widowx_demo --ip 192.168.1.2 --variant base --move
```

## Interface web

Mode simulation, sans mouvement reel:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.apps.interface --port 7862
```

Mode reel:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.apps.interface --real --ip 192.168.1.2 --variant base --port 7862
```

Ouvrir ensuite:

```text
http://127.0.0.1:7862
```

Dans l'interface, cocher `enable motion` avant chaque commande qui bouge le bras. Utiliser `Emergency stop` si le bras doit etre stoppe immediatement.

## Compensation de gravite

Pour guider le bras a la main:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.gravity_compensation --ip 192.168.1.2 --variant base
```

Avec une RealSense D405 montee sur la pince, le profil `d405-follower` peut mieux compenser le poids:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.robot.gravity_compensation --ip 192.168.1.2 --variant base --payload-profile d405-follower --camera-wrist-effort 0.10
```

## Enregistrement et datasets

L'interface web permet:

- d'enregistrer un mouvement guide a la main;
- de rejouer ce mouvement;
- de capturer les images camera pendant le replay;
- de produire des fichiers `motor_samples.jsonl`, `samples.jsonl` et `metadata.json`.

Verifier la synchronisation du dernier dataset:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.tools.check_dataset_sync
```

Verifier un dataset precis:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.tools.check_dataset_sync widowx_ai/recordings/NOM_DU_DATASET
```

Objectifs pratiques:

- camera autour de 30 Hz;
- moteur autour de 100 Hz;
- decalage image/moteur sous 50 ms;
- aucune image manquante.

## Entrainement ACT

Entrainer un modele local:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.training.train_act --epochs 10 --batch-size 16 --chunk-size 8 --view wrist_rgb
```

Les checkpoints sont sauvegardes dans:

```text
widowx_ai/models/
```

Dashboard de suivi:

```bash
.venv-trossen-ui/bin/python -m widowx_ai.training.act_monitor --port 7865
```

Puis ouvrir:

```text
http://127.0.0.1:7865
```

## Securite

Avant tout mouvement reel:

- fixer le bras sur une surface stable;
- liberer la zone de travail;
- commencer sans charge dans la pince;
- tester d'abord sans `--move` ou sans `--real`;
- garder l'arret d'urgence accessible.

# WidowX AI - controle local

Ce dossier ajoute une installation minimale pour tester un bras Trossen Robotics WidowX AI depuis ce projet.

## Demarrage rapide pour controler le bras

Place-toi d'abord a la racine du projet:

```bash
cd /home/mvautl/Documents/Stage
```

Verifie que le bras repond sur le reseau:

```bash
ping 192.168.1.2
```

Lance ensuite l'interface web en mode reel:

```bash
.venv-widowx/bin/python widowx_ai/interface.py --real --ip 192.168.1.2 --variant base --port 7862
```

Ouvre la page de controle:

```text
http://127.0.0.1:7862
```

Dans la page:

1. Verifie que le statut indique `REAL`.
2. Coche `enable motion` avant chaque commande qui bouge le bras.
3. Utilise d'abord `Gravity comp` pour guider le bras a la main.
4. Utilise les boutons `Home`, `Rest`, `Demo`, les sliders de joints ou les boutons gripper seulement quand la zone est degagee.
5. Utilise `Arret urgence` si le bras doit etre stoppe immediatement.

## Materiel supporte

- Trossen Robotics WidowX AI, modele `wxai_v0`
- Variantes end-effector: `base`, `leader`, `follower` <-- i have the base variantes
- Controle via Ethernet avec le driver Python officiel `trossen-arm`

Sources verifiees le 2026-04-21:

- Documentation Trossen Arm v1.9: https://docs.trossenrobotics.com/trossen_arm/v1.9/
- Produit WidowX AI: https://www.trossenrobotics.com/widowx-ai
- Repo driver: https://github.com/TrossenRobotics/trossen_arm

## Installation faite ici

Un environnement Python dedie a ete cree a la racine du projet:

```bash
/home/mvautl/.pyenv/versions/3.10.12/bin/python3.10 -m venv .venv-widowx
.venv-widowx/bin/python -m pip install --upgrade pip setuptools wheel
.venv-widowx/bin/python -m pip install -r widowx_ai/requirements.txt
```

Version installee:

```bash
.venv-widowx/bin/python -m pip show trossen-arm
```

La version installee localement est `trossen-arm 1.9.1`.

## Reseau du bras

Le controleur Trossen Arm communique par Ethernet. D'apres la documentation Trossen, le reglage usine est:

- IP controleur bras: `192.168.1.2`
- Masque: `255.255.255.0`
- Passerelle: `192.168.1.1`

Configure l'interface Ethernet du PC en IPv4 manuel, par exemple:

- IP PC: `192.168.1.1`
- Masque: `255.255.255.0`

Puis verifie:

```bash
ping 192.168.1.2
```

Le sandbox Codex ne permet pas de lire/configurer l'interface reseau ici, donc cette partie doit etre faite sur l'OS.

## Tests

Verifier seulement que le driver Python est installe:

```bash
.venv-widowx/bin/python widowx_ai/widowx_demo.py --check-import
```

Tester la connexion au bras et lire l'etat, sans mouvement:

```bash
.venv-widowx/bin/python widowx_ai/widowx_demo.py --ip 192.168.1.2 --variant base
```

Lancer une demo avec petit mouvement en joint-space:

```bash
.venv-widowx/bin/python widowx_ai/widowx_demo.py --ip 192.168.1.2 --variant base --move
```

Variantes possibles:

```bash
--variant base
--variant leader
--variant follower
```

La camera USB Intel RealSense D405 ne change pas la variante mecanique du bras. Si ton bras est la
variante `base`, garde `--variant base` pour les tests generaux. Pour la compensation de gravite,
l'outil utilise par defaut le profil de masse `follower`, car ce profil ajoute environ 105 g sur la
paume et deplace le centre de masse, ce qui correspond mieux a une D405 montee sur la pince.

Activer la compensation de gravite pour guider le bras a la main:

```bash
.venv-widowx/bin/python widowx_ai/gravity_compensation.py --ip 192.168.1.2 --variant base
```

Si la pince penche encore avec la camera, garde le profil `d405-follower` et ajuste tres doucement
le biais du poignet:

```bash
.venv-widowx/bin/python widowx_ai/gravity_compensation.py --ip 192.168.1.2 --variant base --payload-profile d405-follower --camera-wrist-effort 0.10
```

Essaie aussi le signe oppose si l'inclinaison empire. Reste dans de petites valeurs, par exemple
`0.04`, `0.08`, `0.12`.

## Interface web locale

Une interface web simple est disponible dans `widowx_ai/interface.py`.

Lance toujours cette commande depuis la racine du projet:

```bash
cd /home/mvautl/Documents/Stage
```

### Mode simulation sans mouvement reel

Par defaut l'interface demarre en dry-run, donc aucun mouvement reel n'est envoye au bras:

```bash
.venv-widowx/bin/python widowx_ai/interface.py --port 7862
```

Ouvre ensuite:

```text
http://127.0.0.1:7862
```

Ce mode est utile pour verifier que la page web demarre, que les boutons repondent et que la camera
est visible, sans connecter le robot.

### Mode reel pour controler le bras

Pour autoriser le controle reel du bras, il faut passer explicitement `--real`. Pour ton bras en
variante `base` et l'IP usine `192.168.1.2`, utilise:

```bash
.venv-widowx/bin/python widowx_ai/interface.py --real --ip 192.168.1.2 --variant base --port 7862
```

Le terminal doit afficher une ligne du type:

```text
WidowX AI interface running at http://127.0.0.1:7862 (REAL ARM ENABLED)
```

Si l'interface ne se connecte pas au bras, verifie d'abord:

- le cable Ethernet entre le PC et le controleur;
- l'IP manuelle du PC, par exemple `192.168.1.1`;
- le ping vers `192.168.1.2`;
- l'alimentation du bras;
- qu'aucune autre application n'utilise deja le controleur.

Si le lancement affiche une erreur du type `Address already in use`, le serveur est deja lance sur
le port `7862`. Dans ce cas, ouvre directement:

```text
http://127.0.0.1:7862
```

Tu peux verifier si le port est deja utilise avec:

```bash
ss -ltnp | grep 7862
```

Si tu veux lancer une deuxieme interface sans fermer la premiere, change simplement le port:

```bash
.venv-widowx/bin/python widowx_ai/interface.py --real --ip 192.168.1.2 --variant base --port 7863
```

Puis ouvre:

```text
http://127.0.0.1:7863
```

### Commandes dans l'interface

L'interface demande de cocher `enable motion` avant chaque commande de mouvement. Sans cette case,
les boutons et sliders ne peuvent pas envoyer de mouvement.

Commandes principales:

- `Gravity comp`: active la compensation de gravite pour guider le bras a la main.
- `Hold`: maintient la position actuelle.
- `Hold off`: remet le bras en compensation de gravite.
- `Home`: envoie le bras vers la position de depart.
- `Rest`: envoie le bras vers la position de repos.
- `Demo`: lance un petit mouvement de test.
- `Open gripper` / `Close gripper`: ouvre ou ferme la pince avec un effort limite.
- `Arret urgence`: stoppe le replay et met le bras en `idle` immediatement.

La vitesse automatique par defaut est `0.30 rad/s`. Le curseur `Max speed` de l'interface controle
la limite utilisee par les mouvements automatiques: `Home`, `Rest`, `Demo`, `Return to start`,
deplacement vers le debut d'un replay et replay. Si un mouvement semble trop rapide, baisse ce
curseur avant de cliquer sur le bouton.

Dans l'interface, le bouton `Gravity comp` utilise aussi par defaut `D405 / profil follower`.
Le curseur `Wrist comp` permet de corriger ton montage exact si le profil follower ne suffit pas.
Le bouton `Hold off` remet le bras en compensation de gravite, pas en `idle`, car le mode `idle`
du driver Trossen garde les joints freines.

### Camera et enregistrement

Le bloc `Camera Hub` regroupe maintenant toute la gestion camera dans un seul module:

- `Intel RealSense D405 RGB`
- `Intel RealSense D405 Depth`
- toutes les webcams USB standards detectees

La webcam integree du PC portable est exclue de cette liste.

Choisis une source unique dans la liste, puis lance `Start live`. Le flux affiche maintenant un
stream video continu dans la page de controle. Si une camera est deja ouverte
par une autre application ou un autre onglet, le driver peut renvoyer `Device or resource busy`.

La page `Teaching` propose deux modes:

- l'outil officiel Trossen pour la collecte dataset;
- un mode local `Save & Replay` pour sauvegarder rapidement un mouvement puis le rejouer doucement.

Le mode local peut enregistrer `Mouvement seul` sans camera, ou ajouter la D405 avec `Avec camera D405`.
C'est le mode a utiliser si tu veux simplement guider le bras a la main puis rejouer exactement ce
mouvement sans passer par une dataset LeRobot complete.

Pour un premier dataset ACT, utilise le flux `Teaching` ainsi:

1. `Step 1 - Record source motion`: enregistre le mouvement source sans video.
2. `Step 2 - Replay and capture`: rejoue ce mouvement et capture les cameras.
3. La D405 enregistre maintenant automatiquement `RGB` et `Depth` en meme temps.
4. Le `Replay speed` est a `0.75x` par defaut; baisse-le si le mouvement semble trop rapide.
5. Dans le bloc separe `Camera capture`, active `Crop video flux` si tu veux enregistrer un cadrage plus serre.

Options disponibles dans le bloc `Camera capture`:

- `Top camera`: selectionne la camera de dessus;
- `Start preview`: lance l'aperçu live;
- clique sur une preview (`Top camera`, `D405 RGB` ou `D405 depth`) pour afficher `Crop and output`;
- `Apply crop to`: applique le crop a la top cam, a la D405, ou aux deux;
- `Ratio`: garde le ratio source ou force `1:1`, `4:3`, `16:9`, `3:2`, `9:16`;
- `Zoom`: augmente le cadrage sans changer la resolution du flux source;
- `Offset X` / `Offset Y`: deplace la fenetre de crop dans l'image.

Le crop est applique avant l'ecriture JPEG. Les images sauvegardees dans le dataset sont donc deja
cropees, et la configuration est conservee dans `metadata.json` via `capture_sources`.
Pour verifier le sens du crop, clique `Start preview`, clique sur la preview a regler, puis ajuste
`Zoom`, `Offset X` et `Offset Y`. Les previews se mettent a jour automatiquement avant de lancer
`Replay movement + record camera`.

Le mode dataset enregistre:

- les positions moteur dans `motor_samples.jsonl` a 100 Hz;
- les images camera dans `samples.jsonl` a 30 Hz;
- un timestamp camera par ligne;
- le `motor_timestamp` le plus proche pour chaque image;
- `sync_delta_seconds`, le decalage image/moteur a verifier avant entrainement;
- `frame_timestamps`, les timestamps par camera quand plusieurs vues sont capturees.

Apres chaque capture, verifie la qualite temporelle avant d'utiliser le dataset pour ACT:

```bash
.venv-widowx/bin/python widowx_ai/check_dataset_sync.py
```

Le script inspecte le dernier dataset capture. Pour verifier un dossier precis:

```bash
.venv-widowx/bin/python widowx_ai/check_dataset_sync.py widowx_ai/recordings/NOM_DU_DATASET
```

Objectifs pratiques pour commencer:

- camera entre 20 Hz et 50 Hz, idealement autour de 30 Hz pour `top_view`, `wrist_rgb` et `wrist_depth`;
- moteur autour de 100 Hz;
- `Max image/motor sync delta` sous 50 ms;
- aucun fichier image manquant;
- si deux cameras sont capturees, `Max inter-camera spread` sous 100 ms.

L'outil officiel Trossen se lance avec:

```bash
.venv-trossen-ui/bin/trossen_ai_data_collection_ui
```

L'UI officielle Trossen gere les taches, les cameras, le dry-run, les sessions d'enregistrement,
le re-record d'un episode et les configurations robot/camera via YAML. Ferme la preview camera de
l'interface web avant de lancer l'UI Trossen, car une RealSense ne peut generalement pas etre ouverte
par deux applications en meme temps.

La configuration persistante de l'UI officielle est ici:

```text
/home/mvautl/.trossen/trossen_ai_data_collection/configs/
```

Points a verifier dans l'UI officielle avant une collecte avec le bras:

- `trossen_ai_solo` utilise par defaut un montage leader/follower Trossen. Les IP doivent etre
  ajustees dans `Robot Configuration` si tu veux utiliser une autre architecture.
- La D405 est deja configuree localement avec le serial `218622273543` en 640x480 a 30 FPS.
- Le mode `camera_preview_d405` permet de tester la D405 seule sans connecter le bras.

## Securite avant `--move`

- Fixer le bras sur une surface stable.
- Enlever les objets dans le rayon du bras.
- Garder la main proche de l'alimentation ou de l'arret d'urgence si disponible.
- Commencer sans charge dans le gripper.
- Utiliser d'abord le mode sans `--move`.

## Suite IA

Pour teleoperation, collecte de donnees, imitation learning ou OpenPI, Trossen recommande maintenant l'integration plugin LeRobot:

```bash
git clone https://github.com/TrossenRobotics/lerobot_trossen.git ~/lerobot_trossen
cd ~/lerobot_trossen
uv sync
```

Ce n'est pas installe ici, car c'est une pile plus lourde que le controle basique du bras.

### Premier entrainement ACT local

Un petit entraineur ACT-style est disponible pour verifier que les donnees locales peuvent etre
chargees et apprises:

```bash
.venv-trossen-ui/bin/python widowx_ai/train_act.py --epochs 10 --batch-size 16 --chunk-size 8 --view wrist_rgb
```

Les checkpoints sont sauvegardes dans:

```text
widowx_ai/models/
```

Le script ecrit `history.json` et `status.json` apres chaque epoch pour permettre le suivi en direct.

### Interface de suivi d'apprentissage

Lance le dashboard local:

```bash
.venv-trossen-ui/bin/python widowx_ai/act_monitor.py --port 7865
```

Ouvre ensuite:

```text
http://127.0.0.1:7865
```

Le dashboard affiche:

- le run selectionne;
- l'epoch courante;
- `train_l1`, `val_l1` et le meilleur score;
- la courbe de perte;
- la configuration et le chemin du meilleur checkpoint.

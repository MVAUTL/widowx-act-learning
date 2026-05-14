# Tutoriel HAMSTER/VILA sur NVIDIA DGX Spark

Ce guide explique comment installer et lancer la pipeline HAMSTER avec le modele VILA-1.5-13B sur la DGX Spark du lab. Il reprend aussi les bugs rencontres pendant l'installation de Matteo et les corrections appliquees.

La regle principale est simple : toute commande qui charge le GPU doit passer par SLURM avec `sbatch`. Ne lance pas `python server.py` directement sur le login node.

## 1. Connexion a la DGX

Depuis un terminal sur le reseau FTLab :

```bash
ssh -4 guest@dgx-spark.local
```

Mot de passe du compte invite :

```text
ftlab2025
```

Si le nom local ne repond pas, utilise l'IP Ethernet :

```bash
ssh -4 guest@192.168.100.36
```

## 2. Dossier de travail

Chaque personne doit travailler dans son propre dossier :

```bash
cd ~
mkdir -p intern_matteo_vulliez
cd intern_matteo_vulliez
```

Adapte le nom si tu n'es pas Matteo :

```bash
mkdir -p intern_<prenom_nom>
```

## 3. Verifier que le GPU est libre

Avant de lancer quoi que ce soit :

```bash
squeue
nvidia-smi
```

Interpretation rapide :

- `squeue` vide : aucun job SLURM en cours.
- `ST=R` : un job tourne deja.
- `ST=PD` : un job attend la file.
- `nvidia-smi` avec un process Python et beaucoup de VRAM : le GPU est occupe.

Ne tue jamais le job de quelqu'un d'autre.

## 4. Structure attendue

L'installation propre utilise cette structure :

```text
~/intern_matteo_vulliez/
|-- HAMSTER_beta/
|   |-- server.py
|   |-- gradio_server_matteo.py
|   |-- gradio_server_example.py
|   |-- VILA/
|   `-- deepspeed/
|-- Hamster_dev/
|   `-- VILA1.5-13b-.../
|-- HAMSTER_dev -> Hamster_dev/VILA1.5-13b-...
|-- envs/
|   `-- hamster-vila/
`-- slurm/
    |-- check_hamster_env.slurm
    |-- hamster_server.slurm
    `-- hamster_full_pipeline.slurm
```

Important : `HAMSTER_dev` est un lien symbolique court vers le dossier des poids reels. Ce lien evite un bug de nom de modele dans l'API.

## 5. Creer l'environnement Python

Sur DGX Spark, l'architecture est ARM64/aarch64. Beaucoup de paquets Python n'ont pas de wheel ARM64. Il faut donc preferer conda/uv et verifier CUDA explicitement.

Exemple :

```bash
cd ~/intern_matteo_vulliez
mkdir -p envs
conda create -y -p ~/intern_matteo_vulliez/envs/hamster-vila python=3.10
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/hamster-vila
```

Verifier Python :

```bash
python --version
which python
```

Installer les dependances utiles :

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install fastapi uvicorn gradio==3.50.2 openai==1.8.0 opencv-python matplotlib numpy pillow requests pydantic
python -m pip install transformers==4.37.2 datasets==2.16.1 peft tyro webdataset nltk==3.3 scikit-learn==1.2.2 protobuf
python -m pip install 's2wrapper@git+https://github.com/bfshi/scaling_on_scales'
```

Verifier CUDA :

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

Si `cuda: False`, n'essaie pas de lancer HAMSTER. Il faut corriger PyTorch/CUDA avant.

## 6. Recuperer le code HAMSTER et VILA

Le setup de Matteo utilise :

- `HAMSTER_beta/` pour le serveur HAMSTER.
- `HAMSTER_beta/VILA/` pour le code VILA.
- le commit VILA recommande par HAMSTER.

Si tu repars de zero :

```bash
cd ~/intern_matteo_vulliez
git clone https://github.com/NVlabs/VILA.git HAMSTER_beta/VILA
cd HAMSTER_beta/VILA
git checkout a5a380d6d09762d6f3fd0443aac6b475fba84f7e
```

Ensuite, installe VILA en editable depuis l'environnement :

```bash
cd ~/intern_matteo_vulliez/HAMSTER_beta/VILA
python -m pip install -e .
```

Si le repo HAMSTER n'est pas encore present, copie ou clone le dossier contenant au minimum :

```text
server.py
gradio_server_example.py
setup_server.sh
```

## 7. Telecharger les poids VILA-1.5-13B HAMSTER

Installer Git LFS si necessaire :

```bash
git lfs install
```

Telecharger le checkpoint Hugging Face :

```bash
cd ~/intern_matteo_vulliez
git clone https://huggingface.co/yili18/Hamster_dev
```

Le dossier final attendu contient un sous-dossier du type :

```text
Hamster_dev/VILA1.5-13b-robopoint_1432k+rlbench_all_tasks_256_1000_eps_sketch_v5_alpha+droid_train99_sketch_v5_alpha_fix+bridge_data_v2_train90_10k_sketch_v5_alpha-e1-LR1e-5
```

Verifier les gros fichiers :

```bash
find ~/intern_matteo_vulliez/Hamster_dev -name '*.safetensors' -lh
```

Tu dois voir plusieurs shards LLM d'environ 5 GB, plus `mm_projector` et `vision_tower`.

Creer le lien court obligatoire :

```bash
cd ~/intern_matteo_vulliez
MODEL_REAL=$(find "$HOME/intern_matteo_vulliez/Hamster_dev" -maxdepth 1 -type d -name 'VILA1.5-13b-*' | head -1)
ln -sfn "$MODEL_REAL" "$HOME/intern_matteo_vulliez/HAMSTER_dev"
ls -l "$HOME/intern_matteo_vulliez/HAMSTER_dev"
```

## 8. Corrections de bugs necessaires

### Bug 1 : `set -u` casse conda CUDA

Erreur observee :

```text
CUDAARCHS_BACKUP: unbound variable
```

Cause : certains scripts `conda deactivate.d` NVIDIA lisent des variables non definies. Avec `set -u`, le script SLURM plante.

Correction : dans les scripts `.slurm`, utiliser :

```bash
set -eo pipefail
```

et non :

```bash
set -euo pipefail
```

### Bug 2 : `ModuleNotFoundError: No module named 'deepspeed'`

Sur DGX Spark ARM64, installer DeepSpeed complet peut etre fragile. Pour l'inference single-GPU HAMSTER, un shim minimal suffit si le code importe seulement `deepspeed.comm`.

Creer :

```bash
mkdir -p ~/intern_matteo_vulliez/HAMSTER_beta/deepspeed
```

Fichier `~/intern_matteo_vulliez/HAMSTER_beta/deepspeed/__init__.py` :

```python
"""Minimal DeepSpeed compatibility shim for single-GPU HAMSTER inference."""
```

Fichier `~/intern_matteo_vulliez/HAMSTER_beta/deepspeed/comm.py` :

```python
import torch.distributed as _dist

is_initialized = _dist.is_initialized
get_rank = _dist.get_rank
get_world_size = _dist.get_world_size
new_group = _dist.new_group

def init_distributed(dist_backend="nccl", dist_init_required=True, *args, **kwargs):
    if dist_init_required and not _dist.is_initialized():
        _dist.init_process_group(backend=dist_backend, init_method="env://")
```

### Bug 3 : modules Python manquants

Erreurs possibles :

```text
ModuleNotFoundError: No module named 's2wrapper'
ModuleNotFoundError: No module named 'datasets'
ModuleNotFoundError: No module named 'protobuf'
```

Corrections :

```bash
python -m pip install 's2wrapper@git+https://github.com/bfshi/scaling_on_scales'
python -m pip install datasets==2.16.1 peft tyro webdataset openai==1.8.0 nltk==3.3 scikit-learn==1.2.2
python -m pip install protobuf
```

### Bug 4 : Gradio envoie le mauvais nom de modele

Erreur observee :

```text
422 Unprocessable Entity
unexpected value; permitted: ... 'HAMSTER_dev'
given: 'Hamster_dev'
```

Dans `gradio_server_example.py`, la ligne originale peut etre :

```python
MODEL = "Hamster_dev"
```

Correction :

```python
MODEL = "HAMSTER_dev"
```

### Bug 5 : le backend compare au nom du dossier checkpoint

Erreur observee :

```text
The endpoint is configured to use the model VILA1.5-13b-..., but the request model is HAMSTER_dev
```

Cause : `server.py` calcule `model_name = get_model_name_from_path(model_path)`. Si `--model-path` pointe vers le long dossier `VILA1.5-13b-...`, l'API refuse `HAMSTER_dev`.

Correction : lancer le backend avec :

```bash
--model-path "$HOME/intern_matteo_vulliez/HAMSTER_dev"
```

et non avec le long chemin reel.

## 9. Script SLURM complet backend + interface web

Creer `~/intern_matteo_vulliez/slurm/hamster_full_pipeline.slurm` :

```bash
#!/bin/bash
#SBATCH --job-name=matteo-hamster-full
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/slurm-logs/%u_%x_%j.out
#SBATCH --error=/home/slurm-logs/%u_%x_%j.err

set -eo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ~/intern_matteo_vulliez/envs/hamster-vila

export PYTHONUNBUFFERED=1
export HF_HOME="$HOME/intern_matteo_vulliez/.hf-cache"

MODEL_DIR="$HOME/intern_matteo_vulliez/HAMSTER_dev"
if [ ! -e "$MODEL_DIR" ]; then
  echo "ERROR: model path not found: $MODEL_DIR"
  exit 2
fi

cd ~/intern_matteo_vulliez/HAMSTER_beta
printf '127.0.0.1\n' > ip_eth0.txt

cleanup() {
  if [ -n "${BACKEND_PID:-}" ]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "=== Backend start at $(date) ==="
python -u -W ignore server.py \
  --host 0.0.0.0 \
  --port 8000 \
  --model-path "$MODEL_DIR" \
  --conv-mode vicuna_v1 &
BACKEND_PID=$!

echo "Waiting for HAMSTER backend on http://127.0.0.1:8000/docs"
for i in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:8000/docs >/dev/null 2>&1; then
    echo "Backend ready after ${i} checks"
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "ERROR: backend exited before becoming ready"
    wait "$BACKEND_PID"
    exit 1
  fi
  sleep 5
done

if ! curl -fsS http://127.0.0.1:8000/docs >/dev/null 2>&1; then
  echo "ERROR: backend did not become ready in time"
  exit 1
fi

echo "=== Web UI start at $(date) ==="
echo "Open: http://192.168.100.36:7860"
python -u gradio_server_matteo.py
```

## 10. Adapter Gradio pour le reseau local

Copier l'exemple :

```bash
cd ~/intern_matteo_vulliez/HAMSTER_beta
cp gradio_server_example.py gradio_server_matteo.py
```

Verifier/corriger le nom du modele :

```bash
grep -n 'MODEL =' gradio_server_matteo.py
```

La ligne doit etre :

```python
MODEL = "HAMSTER_dev"
```

Modifier le lancement Gradio pour ecouter sur le reseau local :

```python
demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
```

Ne pas utiliser `share=True` sauf besoin explicite : le serveur est deja accessible sur le reseau FTLab.

## 11. Lancer la pipeline

Depuis la DGX :

```bash
cd ~/intern_matteo_vulliez
sbatch slurm/hamster_full_pipeline.slurm
```

Exemple de sortie :

```text
Submitted batch job 366
```

Verifier :

```bash
squeue -u $USER
```

Tu dois voir :

```text
JOBID PARTITION NAME      USER  ST TIME
366   main      matteo-h  guest R  ...
```

## 12. Ouvrir l'interface

Depuis un navigateur sur le reseau FTLab :

```text
http://192.168.100.36:7860
```

Backend API/debug :

```text
http://192.168.100.36:8000/docs
```

## 13. Tester que tout fonctionne

Depuis la DGX :

```bash
curl -s -o /dev/null -w 'backend:%{http_code}\n' http://127.0.0.1:8000/docs
curl -s -o /dev/null -w 'web:%{http_code}\n' http://127.0.0.1:7860
```

Resultat attendu :

```text
backend:200
web:200
```

Test API avec l'image exemple :

```bash
cd ~/intern_matteo_vulliez/HAMSTER_beta
python - <<'PY'
import base64, cv2
from openai import OpenAI

img = cv2.imread("examples/ocr_reasoning.jpg")
_, buf = cv2.imencode(".jpg", img)
b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

client = OpenAI(base_url="http://127.0.0.1:8000", api_key="fake-key")
resp = client.chat.completions.create(
    model="HAMSTER_dev",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "In the image, move the S to the plate the arrow is pointing at. Return <ans> points."},
        ],
    }],
    max_tokens=128,
    extra_body={"num_beams": 1, "use_cache": False, "temperature": 0.0, "top_p": 0.95},
)

print(resp.choices[0].message.content)
PY
```

Une reponse correcte ressemble a :

```text
[{'type': 'text', 'text': '<ans>[(0.58, 0.61), <action>Close Gripper</action>, (0.62, 0.43), <action>Open Gripper</action>]</ans>'}]
```

## 14. Lire les logs

Les logs SLURM sont dans `/home/slurm-logs/`.

Pour le job `366` :

```bash
tail -f /home/slurm-logs/guest_matteo-hamster-full_366.out
tail -f /home/slurm-logs/guest_matteo-hamster-full_366.err
```

Remplace `366` par ton job id.

## 15. Arreter la pipeline

```bash
scancel <jobid>
```

Exemple :

```bash
scancel 366
```

## 16. Relancer proprement

Si le serveur ne repond plus :

```bash
squeue -u $USER
scancel <ancien_jobid>
cd ~/intern_matteo_vulliez
sbatch slurm/hamster_full_pipeline.slurm
```

Attends ensuite que les logs affichent :

```text
Model HAMSTER_dev loaded successfully. Context length: 2048
Backend ready
Running on local URL: http://0.0.0.0:7860
```

## 17. Regles importantes DGX Spark

- Utiliser `sbatch` pour tout ce qui touche au GPU.
- Ne jamais lancer le backend VILA directement sur le login node.
- Mettre ton nom dans le job SLURM.
- Garder les fichiers dans `~/intern_<prenom_nom>/`.
- Utiliser un `--time` realiste. Par defaut : `12:00:00`. Maximum lab : `72h`.
- Sauvegarder les checkpoints si tu fais de l'entrainement.
- Ne pas supprimer les logs ou fichiers des autres.
- Ne pas utiliser `sudo`.

## 18. Etat connu pour Matteo

Installation actuelle :

```text
Workdir: /home/guest/intern_matteo_vulliez
Env:     /home/guest/intern_matteo_vulliez/envs/hamster-vila
Model:   /home/guest/intern_matteo_vulliez/HAMSTER_dev
Script:  /home/guest/intern_matteo_vulliez/slurm/hamster_full_pipeline.slurm
Web:     http://192.168.100.36:7860
API:     http://192.168.100.36:8000/docs
```

Dernier test valide :

```text
model="HAMSTER_dev"
backend HTTP 200
web HTTP 200
generation <ans> OK
```

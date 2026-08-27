"""
Script de visualisation des courbes d'apprentissage — LLaMA 3-8B v4-r32
========================================================================
Génère une figure à 3 panneaux (identique au format v4 ALLaM/Mistral) :
  - Panneau gauche  : Train loss vs Validation loss par step
  - Panneau centre  : Score BLEU de validation par step
  - Panneau droit   : Progression par époque (val loss vs BLEU — double axe)

Usage
-----
  python plot_courbes_llama3_r32.py

Le script lit le fichier trainer_state.json sauvegardé par HuggingFace
dans le dossier OUTPUT_DIR (produit automatiquement par le Trainer).

Structure attendue de trainer_state.json (extrait) :
  {
    "log_history": [
      {"step": 100, "loss": 0.72, "eval_loss": 0.88, "eval_bleu": 71.4, "epoch": 0.98},
      {"step": 200, "loss": 0.55, "eval_loss": 0.84, "eval_bleu": 73.1, "epoch": 1.98},
      ...
    ],
    "best_metric": 73.2,
    "best_model_checkpoint": "checkpoint-100"
  }

Si le fichier n'existe pas encore (avant l'entraînement), le script
génère une figure exemple avec des données fictives pour valider le
format visuel.

pip install matplotlib numpy
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURATION — ADAPTER SELON VOTRE MACHINE
# ══════════════════════════════════════════════════════════════
OUTPUT_DIR   = r"C:\Users\infocom\tradiction_dialect\llama3_8b_tun_msa_r32_output-3000"
FIGURE_PATH  = os.path.join(OUTPUT_DIR, "courbe_apprentissage_llama3_r32.png")

# ── Recherche intelligente de trainer_state.json ──────────────────────────────
# HuggingFace sauvegarde trainer_state.json dans chaque checkpoint-XXX
# ET parfois à la racine. On cherche partout et on prend le fichier
# dont le checkpoint a le step le plus élevé (historique le plus complet).
def find_trainer_state(output_dir):
    candidates = []

    # 1. Racine du dossier output
    root_f = os.path.join(output_dir, "trainer_state.json")
    if os.path.isfile(root_f):
        candidates.append((root_f, -1))          # step=-1 → priorité basse

    # 2. Tous les sous-dossiers checkpoint-XXX
    if os.path.isdir(output_dir):
        for entry in os.listdir(output_dir):
            if entry.startswith("checkpoint-"):
                step_str = entry.split("-")[-1]
                step = int(step_str) if step_str.isdigit() else 0
                ckpt_f = os.path.join(output_dir, entry, "trainer_state.json")
                if os.path.isfile(ckpt_f):
                    candidates.append((ckpt_f, step))

    if not candidates:
        return None

    # Trier par step décroissant → le checkpoint le plus avancé en premier
    candidates.sort(key=lambda x: x[1], reverse=True)
    chosen, step = candidates[0]
    loc = f"checkpoint-{step}" if step >= 0 else "racine"
    print(f"  trainer_state.json trouvé : {chosen}  [{loc}]")
    if len(candidates) > 1:
        print(f"  Autres candidats ignorés   : {[c for c,_ in candidates[1:]]}")
    return chosen

STATE_FILE = find_trainer_state(OUTPUT_DIR)

# ── Paramètres de l'expérience (pour l'annotation) ────────────
LORA_R       = 32
DROPOUT      = 0.1
WEIGHT_DECAY = 0.1
MAX_EPOCHS   = 2

# ── Palette (identique aux courbes v4 ALLaM/Mistral) ──────────
COLOR_TRAIN  = "#4C72B0"   # bleu foncé
COLOR_VAL    = "#DD8452"   # orange
COLOR_BLEU   = "#55A868"   # vert teal
COLOR_CKPT   = "#C44E52"   # rouge pointillé early stop
COLOR_FILL   = "#55A868"   # fill area BLEU
BG_COLOR     = "#F8F9FA"

# ══════════════════════════════════════════════════════════════
# 2. LECTURE DU trainer_state.json
# ══════════════════════════════════════════════════════════════

def load_trainer_state(path):
    """
    Lit trainer_state.json et extrait :
      - steps_train  : steps où la train loss est loggée
      - train_losses : train loss correspondantes
      - steps_eval   : steps d'évaluation
      - val_losses   : validation loss aux étapes d'éval
      - bleu_scores  : score BLEU aux étapes d'éval
      - epochs_eval  : époque à chaque step d'éval
      - best_step    : step du meilleur checkpoint
      - best_bleu    : meilleur score BLEU
    """
    with open(path, "r", encoding="utf-8") as f:
        state = json.load(f)

    log = state.get("log_history", [])

    steps_train, train_losses = [], []
    steps_eval, val_losses, bleu_scores, epochs_eval = [], [], [], []

    for entry in log:
        # entrée train (contient "loss" mais pas "eval_loss")
        if "loss" in entry and "eval_loss" not in entry:
            steps_train.append(entry["step"])
            train_losses.append(entry["loss"])
        # entrée eval
        if "eval_loss" in entry:
            steps_eval.append(entry["step"])
            val_losses.append(entry["eval_loss"])
            bleu_scores.append(entry.get("eval_bleu", 0.0))
            epochs_eval.append(entry.get("epoch", 0))

    # Meilleur checkpoint
    best_step = None
    best_bleu = None
    ckpt = state.get("best_model_checkpoint", "")
    if ckpt:
        # "checkpoint-400" → 400
        parts = ckpt.replace("\\", "/").split("/")[-1].split("-")
        if len(parts) >= 2 and parts[-1].isdigit():
            best_step = int(parts[-1])
    if best_step is None and steps_eval:
        # fallback : step avec le BLEU max
        idx = int(np.argmax(bleu_scores))
        best_step = steps_eval[idx]
    if bleu_scores:
        best_bleu = max(bleu_scores)

    return {
        "steps_train"  : steps_train,
        "train_losses" : train_losses,
        "steps_eval"   : steps_eval,
        "val_losses"   : val_losses,
        "bleu_scores"  : bleu_scores,
        "epochs_eval"  : epochs_eval,
        "best_step"    : best_step,
        "best_bleu"    : best_bleu,
    }


def make_dummy_data():
    """Données fictives pour valider le rendu visuel avant l'entraînement."""
    steps_train  = list(range(20, 201, 20))
    train_losses = [0.85, 0.76, 0.68, 0.61, 0.55, 0.50, 0.46, 0.43, 0.40, 0.38]
    steps_eval   = [100, 200]
    val_losses   = [0.92, 0.87]
    bleu_scores  = [71.0, 74.5]
    epochs_eval  = [0.98, 1.98]
    return {
        "steps_train"  : steps_train,
        "train_losses" : train_losses,
        "steps_eval"   : steps_eval,
        "val_losses"   : val_losses,
        "bleu_scores"  : bleu_scores,
        "epochs_eval"  : epochs_eval,
        "best_step"    : 200,
        "best_bleu"    : 74.5,
    }


# ══════════════════════════════════════════════════════════════
# 3. CHARGEMENT
# ══════════════════════════════════════════════════════════════
dummy_mode = False
if STATE_FILE is not None:
    print(f"✓ Lecture {STATE_FILE}")
    data = load_trainer_state(STATE_FILE)
    if not data["steps_eval"]:
        print("  ⚠ Aucune entrée eval trouvée — mode démonstration")
        data = make_dummy_data()
        dummy_mode = True
else:
    print(f"⚠ trainer_state.json introuvable dans {OUTPUT_DIR} ni dans ses checkpoints")
    if os.path.isdir(OUTPUT_DIR):
        ckpts = [e for e in os.listdir(OUTPUT_DIR) if e.startswith("checkpoint-")]
        print(f"  Dossiers checkpoint détectés : {ckpts if ckpts else 'aucun'}")
    data = make_dummy_data()
    dummy_mode = True
    os.makedirs(OUTPUT_DIR, exist_ok=True)

steps_train  = data["steps_train"]
train_losses = data["train_losses"]
steps_eval   = data["steps_eval"]
val_losses   = data["val_losses"]
bleu_scores  = data["bleu_scores"]
epochs_eval  = data["epochs_eval"]
best_step    = data["best_step"]
best_bleu    = data["best_bleu"]

print(f"  Steps éval   : {steps_eval}")
print(f"  Val losses   : {val_losses}")
print(f"  BLEU scores  : {bleu_scores}")
print(f"  Best step    : {best_step}  (BLEU = {best_bleu:.1f})")

# ══════════════════════════════════════════════════════════════
# 4. CONSTRUCTION DE LA FIGURE — 3 PANNEAUX
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
fig.patch.set_facecolor(BG_COLOR)

# ── Titre principal ────────────────────────────────────────────
suffix = " [DÉMO]" if dummy_mode else ""
fig.suptitle(
    f"Courbe d'apprentissage \u2014 LLaMA 3-8B (Tunisien \u2192 MSA) v4-r32{suffix}",
    fontsize=14, fontweight="bold", y=1.01
)

# ─────────────────────────────────────────────────────────────
# PANNEAU GAUCHE : Train loss vs Val loss
# ─────────────────────────────────────────────────────────────
ax1 = axes[0]
ax1.set_facecolor(BG_COLOR)
ax1.set_title("Loss (train vs validation)", fontsize=11, pad=8)

# Courbe train (tous les steps)
if steps_train:
    ax1.plot(steps_train, train_losses,
             color=COLOR_TRAIN, marker="o", markersize=4,
             linewidth=2.0, label="Train loss")
    for s, v in zip(steps_train, train_losses):
        ax1.annotate(f"{v:.3f}", (s, v),
                     textcoords="offset points", xytext=(0, 6),
                     fontsize=7.5, color=COLOR_TRAIN, ha="center")

# Courbe val (steps éval seulement)
if steps_eval:
    ax1.plot(steps_eval, val_losses,
             color=COLOR_VAL, marker="s", markersize=6,
             linewidth=2.0, linestyle="--", label="Validation loss")
    for s, v in zip(steps_eval, val_losses):
        ax1.annotate(f"{v:.3f}", (s, v),
                     textcoords="offset points", xytext=(0, -14),
                     fontsize=8.5, color=COLOR_VAL, ha="center", fontweight="bold")

# Early stop line
if best_step:
    ax1.axvline(x=best_step, color=COLOR_CKPT, linestyle=":",
                linewidth=1.8, label=f"Best ckpt (step {best_step})")

ax1.set_xlabel("Step", fontsize=10)
ax1.set_ylabel("Loss", fontsize=10)
ax1.legend(fontsize=8.5, loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.4)
ax1.tick_params(labelsize=9)

# ─────────────────────────────────────────────────────────────
# PANNEAU CENTRE : Score BLEU de validation
# ─────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.set_facecolor(BG_COLOR)
ax2.set_title("Score BLEU \u2014 eval_bleu (validation)", fontsize=11, pad=8)

if steps_eval and bleu_scores:
    ax2.fill_between(steps_eval, bleu_scores, alpha=0.18, color=COLOR_FILL)
    ax2.plot(steps_eval, bleu_scores,
             color=COLOR_BLEU, marker="D", markersize=7,
             linewidth=2.5, label="BLEU (val)")
    for s, b in zip(steps_eval, bleu_scores):
        ax2.annotate(f"{b:.1f}", (s, b),
                     textcoords="offset points", xytext=(0, 8),
                     fontsize=9, color=COLOR_BLEU, ha="center", fontweight="bold")

    # Point max BLEU
    idx_max = int(np.argmax(bleu_scores))
    ax2.scatter([steps_eval[idx_max]], [bleu_scores[idx_max]],
                color=COLOR_BLEU, s=120, zorder=5,
                label=f"Max BLEU {best_bleu:.1f} (step {best_step})")
    # Early stop line
    if best_step:
        ax2.axvline(x=best_step, color=COLOR_CKPT, linestyle=":",
                    linewidth=1.8, label=f"Early stop ckpt (step {best_step})")

ax2.set_xlabel("Step", fontsize=10)
ax2.set_ylabel("BLEU Score", fontsize=10)
ax2.legend(fontsize=8.5, loc="lower right")
ax2.grid(True, linestyle="--", alpha=0.4)
ax2.tick_params(labelsize=9)
if bleu_scores:
    margin = max(1.0, (max(bleu_scores) - min(bleu_scores)) * 0.4)
    ax2.set_ylim(min(bleu_scores) - margin, max(bleu_scores) + margin * 1.5)

# ─────────────────────────────────────────────────────────────
# PANNEAU DROIT : Progression par époque (val loss vs BLEU)
# ─────────────────────────────────────────────────────────────
ax3 = axes[2]
ax3_bleu = ax3.twinx()
ax3.set_facecolor(BG_COLOR)
ax3_bleu.set_facecolor(BG_COLOR)
ax3.set_title("Progression par époque (val loss vs BLEU)", fontsize=11, pad=8)

# Grouper par époque entière
epoch_data = {}
for step, ep, vl, bl in zip(steps_eval, epochs_eval, val_losses, bleu_scores):
    ep_int = int(round(ep))
    if ep_int not in epoch_data or step > epoch_data[ep_int]["step"]:
        epoch_data[ep_int] = {"step": step, "val_loss": vl, "bleu": bl}

ep_keys   = sorted(epoch_data.keys())
ep_steps  = [epoch_data[e]["step"]     for e in ep_keys]
ep_valloss= [epoch_data[e]["val_loss"] for e in ep_keys]
ep_bleu   = [epoch_data[e]["bleu"]     for e in ep_keys]

if ep_steps:
    ax3.plot(ep_steps, ep_valloss,
             color=COLOR_VAL, marker="s", linewidth=2.2,
             linestyle="--", label="Val loss")
    ax3_bleu.plot(ep_steps, ep_bleu,
                  color=COLOR_BLEU, marker="D", linewidth=2.2,
                  label="BLEU")
    for s, vl, bl in zip(ep_steps, ep_valloss, ep_bleu):
        ax3.annotate(f"{vl:.3f}", (s, vl),
                     textcoords="offset points", xytext=(-10, 6),
                     fontsize=8, color=COLOR_VAL)
        ax3_bleu.annotate(f"{bl:.1f}", (s, bl),
                          textcoords="offset points", xytext=(5, -12),
                          fontsize=8, color=COLOR_BLEU, fontweight="bold")

    # Annoter les époques
    for ep_n, s in zip(ep_keys, ep_steps):
        ax3.axvline(x=s, color="gray", linestyle=":", alpha=0.3, linewidth=1)
        ax3.annotate(f"Époque {ep_n}", (s, max(ep_valloss)),
                     textcoords="offset points", xytext=(3, -20),
                     fontsize=7.5, color="gray")

ax3.set_xlabel("Step", fontsize=10)
ax3.set_ylabel("Validation loss", fontsize=10, color=COLOR_VAL)
ax3_bleu.set_ylabel("BLEU Score", fontsize=10, color=COLOR_BLEU)
ax3.tick_params(axis="y", labelcolor=COLOR_VAL, labelsize=9)
ax3_bleu.tick_params(axis="y", labelcolor=COLOR_BLEU, labelsize=9)
ax3.tick_params(axis="x", labelsize=9)
ax3.grid(True, linestyle="--", alpha=0.3)

# Légende combinée
legend_handles = [
    Line2D([0],[0], color=COLOR_VAL,  marker="s", lw=2, linestyle="--", label="Val loss"),
    Line2D([0],[0], color=COLOR_BLEU, marker="D", lw=2, label="BLEU"),
]
ax3.legend(handles=legend_handles, fontsize=8.5, loc="upper left")

# ─────────────────────────────────────────────────────────────
# ANNOTATION BAS DE PAGE (hyperparamètres)
# ─────────────────────────────────────────────────────────────
hyperparam_str = (
    f"LoRA r={LORA_R}  |  dropout={DROPOUT}  |  weight_decay={WEIGHT_DECAY}  |  "
    f"epochs_max={MAX_EPOCHS}  |  Early Stop patience=1 (metric: eval_bleu \u2191)  |  "
    f"Best ckpt: step {best_step}"
)
fig.text(0.5, -0.02, hyperparam_str,
         ha="center", fontsize=8.5, color="#444444",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8EFF6",
                   edgecolor="#AABBCC", linewidth=0.8))

# ══════════════════════════════════════════════════════════════
# 5. SAUVEGARDE
# ══════════════════════════════════════════════════════════════
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight",
            facecolor=BG_COLOR, edgecolor="none")
plt.close()

print(f"\n✅ Figure sauvegardée → {FIGURE_PATH}")
if dummy_mode:
    print("   ⚠ Mode démonstration — relancer après l'entraînement pour les vraies courbes.")

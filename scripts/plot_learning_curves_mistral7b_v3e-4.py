"""
plot_learning_curves_mistral7b_v4.py
-------------------------------------
Visualisation des courbes d'apprentissage du modèle Mistral-7B-Instruct.
Compatible avec finetune_mistral7b_tun_msa_v4000.py.

Paramètres du fine-tuning (v4) :
  - Model ID                : mistralai/Mistral-7B-Instruct-v0.3
  - Format prompt           : <s>[INST] instruction\n\ninput [/INST] output</s>
  - num_train_epochs        : 3  (Early Stopping patience=1, arrêt recommandé époque 2)
  - metric_for_best_model   : eval_bleu  (greater_is_better=True)
  - learning_rate           : 3e-4  (cosine scheduler, warmup_steps=100)
  - logging_steps           : 10
  - eval_steps              : 100
  - save_steps              : 100  (aligné sur eval_steps)
  - weight_decay            : 0.10
  - LoRA                    : r=16, lora_alpha=32, lora_dropout=0.10
  - Fix dtype               : bfloat16 forcé après training (avant inférence)
  - Méthode                 : QLoRA 4-bit NF4 + LoRA PEFT

Prérequis :
    pip install matplotlib
"""

import os
import glob
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = r"C:\Users\infocom\tradiction_dialect\mistral7b_tun_msa_output-4000-3e-4"
SAVE_PATH  = os.path.join(OUTPUT_DIR, "courbe_apprentissage_mistral7b_v4.png")
SEP        = "=" * 60

# Paramètres v4 (pour annotations)
MODEL_LABEL    = "Mistral-7B-Instruct v0.3"
LORA_R         = 16
LORA_DROPOUT   = 0.10
WEIGHT_DECAY   = 0.10
LEARNING_RATE  = "3e-4"
MAX_EPOCHS     = 3
EARLY_PATIENCE = 1

# Palette couleurs Mistral
C_TRAIN    = "#6C63FF"   # violet Mistral — train loss
C_VAL      = "#F07B5B"   # orange — val loss
C_BLEU     = "#3BB8A0"   # vert-teal — BLEU
C_EARLYSTOP = "#E63B6F"  # rose — early stop / best ckpt


# ── Recherche du trainer_state.json ───────────────────────────────────────────

def find_trainer_state(output_dir: str) -> str | None:
    """
    Cherche trainer_state.json dans cet ordre :
      1. Directement dans output_dir
      2. Dans les sous-dossiers checkpoint-*, en prenant le plus récent
    """
    root_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(root_path):
        return root_path

    checkpoints = glob.glob(
        os.path.join(output_dir, "checkpoint-*", "trainer_state.json")
    )
    if checkpoints:
        checkpoints.sort(
            key=lambda p: int(os.path.basename(os.path.dirname(p)).split("-")[-1])
        )
        return checkpoints[-1]

    return None


# ── Extraction du meilleur checkpoint ─────────────────────────────────────────

def find_best_step(state: dict) -> int | None:
    """
    Récupère le step du meilleur modèle depuis best_model_checkpoint.
    """
    best_ckpt = state.get("best_model_checkpoint")
    if best_ckpt:
        basename = os.path.basename(best_ckpt.rstrip("/\\"))
        if basename.startswith("checkpoint-"):
            try:
                return int(basename.split("-")[-1])
            except ValueError:
                pass
    return None


# ── Extraction de l'historique ─────────────────────────────────────────────────

def extract_history(log_history: list[dict]) -> dict:
    """
    Reconstruit les métriques depuis trainer.state.log_history.

    - Entrées avec "loss"      → steps d'entraînement (logging_steps=10)
    - Entrées avec "eval_loss" → points d'évaluation  (eval_steps=100)

    v4 : eval_bleu toujours présent (compute_metrics activé,
         metric_for_best_model="eval_bleu").
    """
    history = {
        "step":       [],
        "epoch":      [],
        "train_loss": [],
        "val_loss":   [],
        "bleu":       [],
    }
    train_loss_buffer = None

    for entry in log_history:
        if "loss" in entry:
            train_loss_buffer = entry["loss"]
        if "eval_loss" in entry:
            history["step"].append(int(entry.get("step", 0)))
            history["epoch"].append(entry.get("epoch", len(history["epoch"]) + 1))
            history["train_loss"].append(train_loss_buffer)
            history["val_loss"].append(entry.get("eval_loss"))
            history["bleu"].append(entry.get("eval_bleu"))

    return history


# ── Style global ───────────────────────────────────────────────────────────────

def set_plot_style() -> None:
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linestyle":    "--",
    })


# ── Graphe 1 : Loss ───────────────────────────────────────────────────────────

def plot_loss(ax, steps, train_loss, val_loss, best_step: int | None) -> None:
    ax.plot(steps, train_loss, marker="o", linewidth=2,
            color=C_TRAIN, label="Train loss")
    ax.plot(steps, val_loss,   marker="s", linewidth=2, linestyle="--",
            color=C_VAL,   label="Validation loss")

    for x, y in zip(steps, train_loss):
        if y is not None:
            ax.annotate(f"{y:.3f}", (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7.5, color=C_TRAIN)
    for x, y in zip(steps, val_loss):
        if y is not None:
            ax.annotate(f"{y:.3f}", (x, y),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=7.5, color=C_VAL)

    if best_step is not None:
        ax.axvline(best_step, color=C_EARLYSTOP, linestyle=":",
                   linewidth=1.8, label=f"Best ckpt (step {best_step})")

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Loss (train vs validation)", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(framealpha=0.6)


# ── Graphe 2 : BLEU ───────────────────────────────────────────────────────────

def plot_bleu(ax, steps, bleu_scores, best_step: int | None) -> None:
    valid = [(s, b) for s, b in zip(steps, bleu_scores) if b is not None]
    if not valid:
        ax.text(
            0.5, 0.5,
            "Score BLEU non disponible\n(vérifier compute_metrics)",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=10, color="gray",
        )
        ax.set_title("Score BLEU — eval_bleu (validation)", fontsize=12)
        return

    st, bl = zip(*valid)
    ax.plot(st, bl, marker="D", linewidth=2, color=C_BLEU, label="BLEU (val)")
    ax.fill_between(st, bl, alpha=0.12, color=C_BLEU)

    for x, y in zip(st, bl):
        ax.annotate(f"{y:.1f}", (x, y),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.5, color="#1d7a68")

    best_bleu_val  = max(bl)
    best_bleu_step = st[bl.index(best_bleu_val)]
    ax.scatter([best_bleu_step], [best_bleu_val],
               s=90, zorder=5, color="#1d7a68",
               label=f"Max BLEU {best_bleu_val:.1f} (step {best_bleu_step})")

    if best_step is not None:
        ax.axvline(best_step, color=C_EARLYSTOP, linestyle=":",
                   linewidth=1.8, label=f"Early stop ckpt (step {best_step})")

    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("BLEU Score", fontsize=11)
    ax.set_title("Score BLEU — eval_bleu (validation)", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(framealpha=0.6)


# ── Graphe 3 : Progression par époque ─────────────────────────────────────────

def plot_epoch_progress(ax, steps, epochs, val_loss, bleu) -> None:
    """
    Axe double : val_loss (gauche) vs BLEU (droite) par step,
    avec bandes verticales colorées par époque.
    Permet de visualiser à quelle époque l'Early Stopping s'est déclenché.
    """
    colors_epoch = ["#EAF0FB", "#FEF9E7", "#EAFAF1", "#FDEDEC"]

    epoch_step_ranges: dict[int, list[int]] = {}
    for s, e in zip(steps, epochs):
        ep = int(e)
        epoch_step_ranges.setdefault(ep, []).append(s)

    for ep in sorted(epoch_step_ranges):
        ep_steps = epoch_step_ranges[ep]
        color = colors_epoch[(ep - 1) % len(colors_epoch)]
        ax.axvspan(min(ep_steps), max(ep_steps), alpha=0.30, color=color,
                   label=f"Époque {ep}")

    ax.plot(steps, val_loss, marker="s", linewidth=2, linestyle="--",
            color=C_VAL, label="Val loss")
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("Validation loss", fontsize=11, color=C_VAL)
    ax.tick_params(axis="y", labelcolor=C_VAL)

    ax2 = ax.twinx()
    valid_bleu = [(s, b) for s, b in zip(steps, bleu) if b is not None]
    if valid_bleu:
        sb, bb = zip(*valid_bleu)
        ax2.plot(sb, bb, marker="D", linewidth=2, color=C_BLEU, label="BLEU")
        ax2.set_ylabel("BLEU Score", fontsize=11, color="#1d7a68")
        ax2.tick_params(axis="y", labelcolor="#1d7a68")
        ax2.spines["right"].set_visible(True)

    ax.set_title("Progression par époque (val loss vs BLEU)", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    handles_l, labels_l = ax.get_legend_handles_labels()
    if valid_bleu:
        handles_r, labels_r = ax2.get_legend_handles_labels()
        handles_l += handles_r
        labels_l  += labels_r
    ax.legend(handles_l, labels_l, framealpha=0.6, fontsize=8, loc="upper right")


# ── Pipeline principal ─────────────────────────────────────────────────────────

def plot_learning_curves(
    log_history: list[dict],
    state: dict,
    save_path: str = SAVE_PATH,
    title: str = f"Courbe d'apprentissage — Mistral-7B-Instruct (Tunisien → MSA) v4",
) -> str:
    history   = extract_history(log_history)
    best_step = find_best_step(state)

    if not history["step"]:
        print("⚠ Aucune donnée d'évaluation trouvée dans log_history.")
        return ""

    set_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plot_loss(
        axes[0],
        history["step"], history["train_loss"], history["val_loss"],
        best_step,
    )
    plot_bleu(
        axes[1],
        history["step"], history["bleu"],
        best_step,
    )
    plot_epoch_progress(
        axes[2],
        history["step"], history["epoch"],
        history["val_loss"], history["bleu"],
    )

    # Bandeau paramètres v4
    param_text = (
        f"lr={LEARNING_RATE}  |  LoRA r={LORA_R}  |  dropout={LORA_DROPOUT}  |  "
        f"weight_decay={WEIGHT_DECAY}  |  epochs_max={MAX_EPOCHS}  |  "
        f"Early Stop patience={EARLY_PATIENCE} (metric: eval_bleu ↑)"
        + (f"  |  Best ckpt: step {best_step}" if best_step else "")
    )
    fig.text(
        0.5, -0.03, param_text,
        ha="center", va="center", fontsize=9, color="#555555",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F5F5F5",
                  edgecolor="#CCCCCC", alpha=0.8),
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Courbes sauvegardées : {save_path}")
    return save_path


# ── Point d'entrée ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP)
    print(f"  {MODEL_LABEL} v4 — Courbes d'apprentissage")
    print(SEP)

    state_path = find_trainer_state(OUTPUT_DIR)

    if state_path is None:
        print(f"❌ trainer_state.json introuvable dans :")
        print(f"   {OUTPUT_DIR}")
        print(f"   ni dans ses sous-dossiers checkpoint-*")
        print("   Lancez d'abord finetune_mistral7b_tun_msa_v4000.py.")
    else:
        print(f"  ✓ Fichier trouvé : {state_path}\n")
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        log_history = state.get("log_history", [])
        if not log_history:
            print("❌ log_history vide dans trainer_state.json.")
        else:
            print(f"  ✓ {len(log_history)} entrées dans log_history.\n")

            best_step = find_best_step(state)
            if best_step:
                print(f"  ✓ Meilleur checkpoint détecté : step {best_step}")
                best_bleu = state.get("best_metric")
                if best_bleu is not None:
                    print(f"    eval_bleu au best ckpt      : {best_bleu:.2f}")
            else:
                print("  ⚠ best_model_checkpoint absent "
                      "(early stopping non déclenché ou entraînement incomplet).")

            print()
            plot_learning_curves(log_history, state)

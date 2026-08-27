"""
plot_learning_curves.py
------------------------
Visualisation des courbes d'apprentissage du modèle AraBART.
Compatible avec finetune_arabart_tun_msa.py.

À exécuter après l'entraînement, quand `trainer` est disponible en mémoire,
ou en passant directement log_history en argument.

Les métriques extraites sont :
  - train_loss   (loggée tous les logging_steps=20)
  - eval_loss    (loggée tous les eval_steps=100)
  - (eval_bleu si compute_metrics est configuré)

Prérequis :
    pip install matplotlib
"""

import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ── Configuration ──────────────────────────────────────────────────────────────

OUTPUT_DIR = r"C:\Users\infocom\tradiction_dialect\arabart_tun_msa_output-3000"
SAVE_PATH  = os.path.join(OUTPUT_DIR, "courbe_apprentissage_arabart.png")
SEP        = "=" * 60


# ── Extraction de l'historique ─────────────────────────────────────────────────

def extract_history(log_history: list[dict]) -> dict:
    """
    Reconstruit les métriques par epoch depuis trainer.state.log_history.

    Stratégie :
      - Les entrées avec "loss" correspondent aux steps d'entraînement.
      - Les entrées avec "eval_loss" correspondent aux évaluations (eval_steps=100).
      - On regroupe par epoch entière pour lisibilité.

    Args:
        log_history : trainer.state.log_history

    Returns:
        dict avec epoch, train_loss, val_loss, bleu (None si absent)
    """
    history = {"epoch": [], "train_loss": [], "val_loss": [], "bleu": []}
    train_loss_buffer = None

    for entry in log_history:
        # Buffer du dernier train loss connu
        if "loss" in entry:
            train_loss_buffer = entry["loss"]

        # À chaque évaluation, on enregistre un point
        if "eval_loss" in entry:
            history["epoch"].append(int(entry.get("epoch", len(history["epoch"]) + 1)))
            history["train_loss"].append(train_loss_buffer)
            history["val_loss"].append(entry.get("eval_loss"))
            # eval_bleu présent seulement si compute_metrics est défini dans le trainer
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

def plot_loss(ax, epochs: list, train_loss: list, val_loss: list) -> None:
    """
    Trace train loss vs validation loss.
    Paramètres cohérents avec le fine-tuning :
      - 10 epochs max, early stopping patience=3
      - eval_steps=100, logging_steps=20
    """
    ax.plot(epochs, train_loss, marker="o", linewidth=2,
            color="#5B8AF0", label="Train loss")
    ax.plot(epochs, val_loss, marker="s", linewidth=2,
            linestyle="--", color="#F07B5B", label="Validation loss")

    for x, y in zip(epochs, train_loss):
        if y is not None:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color="#5B8AF0")

    for x, y in zip(epochs, val_loss):
        if y is not None:
            ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color="#F07B5B")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("Loss (train vs validation)", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(framealpha=0.6)


# ── Graphe 2 : BLEU ───────────────────────────────────────────────────────────

def plot_bleu(ax, epochs: list, bleu_scores: list) -> None:
    """
    Trace le score BLEU de validation.
    Note : disponible seulement si compute_metrics est configuré dans le trainer.
    Sinon, affiche un message informatif.
    """
    valid = [(e, b) for e, b in zip(epochs, bleu_scores) if b is not None]

    if not valid:
        ax.text(0.5, 0.5,
                "Score BLEU non disponible\n(compute_metrics non configuré)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, color="gray")
        ax.set_title("Score BLEU (validation)", fontsize=12)
        return

    ep, bl = zip(*valid)
    ax.plot(ep, bl, marker="D", linewidth=2, color="#52C299", label="BLEU (val)")
    ax.fill_between(ep, bl, alpha=0.12, color="#52C299")

    for x, y in zip(ep, bl):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color="#2e8c68")

    best_epoch = ep[bl.index(max(bl))]
    ax.axvline(best_epoch, color="#2e8c68", linestyle=":",
               linewidth=1.5, label=f"Meilleure epoch ({best_epoch})")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("BLEU Score", fontsize=11)
    ax.set_title("Score BLEU (validation)", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(framealpha=0.6)


# ── Pipeline principal ─────────────────────────────────────────────────────────

def plot_learning_curves(
    log_history: list[dict],
    save_path: str = SAVE_PATH,
    title: str = "Courbe d'apprentissage — AraBART (Tunisien → MSA)",
) -> str:
    """
    Génère et sauvegarde les courbes d'apprentissage.

    Args:
        log_history : trainer.state.log_history
        save_path   : chemin de sauvegarde (PNG)
        title       : titre de la figure

    Returns:
        Chemin de l'image sauvegardée
    """
    history = extract_history(log_history)

    if not history["epoch"]:
        print("⚠ Aucune donnée d'évaluation trouvée dans log_history.")
        return ""

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plot_loss(axes[0], history["epoch"], history["train_loss"], history["val_loss"])
    plot_bleu(axes[1], history["epoch"], history["bleu"])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Courbes sauvegardées : {save_path}")
    return save_path


# ── Point d'entrée ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import glob

    def find_trainer_state(output_dir: str) -> str | None:
        """
        Cherche trainer_state.json dans cet ordre :
          1. Directement dans output_dir
          2. Dans les sous-dossiers checkpoint-*, en prenant le plus récent (numéro le plus élevé)
        """
        # 1. Racine du dossier
        root_path = os.path.join(output_dir, "trainer_state.json")
        if os.path.exists(root_path):
            return root_path

        # 2. Sous-dossiers checkpoint-XXXX
        checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*", "trainer_state.json"))
        if checkpoints:
            # Trier par numéro de step (le plus grand = le plus récent)
            checkpoints.sort(key=lambda p: int(os.path.basename(os.path.dirname(p)).split("-")[-1]))
            return checkpoints[-1]   # checkpoint le plus avancé

        return None

    print(SEP)
    print("  AraBART — Courbes d'apprentissage")
    print(SEP)

    state_path = find_trainer_state(OUTPUT_DIR)

    if state_path is None:
        print(f"❌ trainer_state.json introuvable dans :")
        print(f"   {OUTPUT_DIR}")
        print(f"   ni dans ses sous-dossiers checkpoint-*")
        print("   Lancez d'abord finetune_arabart_tun_msa.py.")
    else:
        print(f"  ✓ Fichier trouvé : {state_path}\n")

        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        log_history = state.get("log_history", [])
        if not log_history:
            print("❌ log_history vide dans trainer_state.json.")
        else:
            print(f"  ✓ {len(log_history)} entrées dans log_history.\n")
            plot_learning_curves(log_history)

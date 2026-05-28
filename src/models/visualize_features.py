import joblib
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path

# ── IMPORTS REQUIRED FOR PICKLE UNPACKING ────────────────────────────
# This tells Python exactly how to reconstruct the saved artifact 
# without throwing AttributeError shortcuts.
try:
    from src.models.train import ModelArtifact, TrainingConfig
except ImportError:
    # Fallback dummy placeholders if the above names differ slightly
    class ModelArtifact: pass
    class TrainingConfig: pass

def main():
    print("======================================================================")
    print("📊 Generating LightGBM Feature Importance Visualizations")
    print("======================================================================")

    # 1. Map out directory paths
    model_path = Path("outputs/models/SE3_lgbm_imbalance.joblib")
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"❌ Error: Could not find model artifact at {model_path}")
        print("Please ensure you ran 'python -m src.models.train' successfully first.")
        return

    # 2. Safely unpickle the artifact with its known parent blueprints
    print(f"Loading model serialization file: {model_path}...")
    artifact = joblib.load(model_path)

    # 3. Extract the raw LightGBM model instance out of the artifact structure
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
    elif hasattr(artifact, "model"):
        model = artifact.model
    else:
        model = artifact  # Fallback directly to the object

    # 4. Initialize the Matplotlib figure canvas
    fig, ax = plt.subplots(figsize=(12, 8))

    # 5. Leverage LightGBM's built-in vector plotting tool
    # 'gain' tells us the total reduction of loss contributed by each split rule
    lgb.plot_importance(
        model, 
        ax=ax, 
        max_num_features=20, 
        importance_type="gain", 
        precision=1,
        color="skyblue",
        edgecolor="black"
    )

    plt.title("LightGBM Feature Importance (Gain Metric) — Zone SE3 (Stockholm)", fontsize=14, fontweight="bold")
    plt.xlabel("Total Information Gain Contributed to Splitting Criteria", fontsize=11)
    plt.ylabel("Engineered Pipeline Feature Columns", fontsize=11)
    plt.tight_layout()

    # 6. Save the physical chart image straight down to disk
    chart_file = output_dir / "feature_importance_se3.png"
    plt.savefig(chart_file, dpi=300)
    plt.close()

    print("\n======================================================================")
    print(f"🎉 SUCCESS: Visualization rendered perfectly.")
    print(f"Chart saved down to: {chart_file.absolute()}")
    print("======================================================================")

if __name__ == "__main__":
    main()
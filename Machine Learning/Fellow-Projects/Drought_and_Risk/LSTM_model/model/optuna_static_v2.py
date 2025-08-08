import os
import yaml
import subprocess
import optuna
import pandas as pd
import time
from datetime import datetime
from neuralhydrology.evaluation import evaluate

# Paths and constants
BASE_CONFIG = "basins_static.yml"
RESULTS_CSV = "optuna_lstm_results.csv"
METRICS_CSV = "optuna_lstm_metrics.csv"

RUNS_DIR = "/home/jovyan/watersoft/optuna_results_v3"
FINAL_RUN_DIR = os.path.join(RUNS_DIR, "best_final_run")
FINAL_CONFIG_PATH = os.path.join(FINAL_RUN_DIR, "final_config.yml")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(FINAL_RUN_DIR, exist_ok=True)

if os.path.exists(METRICS_CSV):
    os.remove(METRICS_CSV)

def objective(trial):
    start_time = time.time()

    # Define hyperparameter space
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    output_dropout = trial.suggest_float("output_dropout", 0.0, 0.6)
    hidden_size = trial.suggest_int("hidden_size", 64, 256, step=32)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])

    # Load and update configuration
    with open(BASE_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    config["model"] = "ealstm"
    config["head"] = "regression"
    config["learning_rate"] = {0: float(learning_rate)}
    config["output_dropout"] = float(output_dropout)
    config["hidden_size"] = int(hidden_size)
    config["batch_size"] = int(batch_size)
    config["seq_length"] = 30

    # Create run directory
    run_name = f"lr{learning_rate:.1e}_drop{output_dropout:.2f}_hs{hidden_size}_bs{batch_size}"
    run_dir = os.path.join(RUNS_DIR, run_name)
    config["run_dir"] = run_dir
    config["experiment_name"] = f"optuna_trial_{trial.number}"
    os.makedirs(run_dir, exist_ok=True)

    # Write updated config
    config_path = os.path.join(run_dir, "config.yml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    print(f"[{datetime.now().isoformat()}] Starting Trial {trial.number}: {run_name}")

    try:
        subprocess.run([
            "python", "-m", "neuralhydrology.nh_run", "train", "--config-file", config_path
        ], check=True)

        val_metrics = evaluate(run_dir, data_set="val")
        nse = val_metrics.get("NSE", -9999)

        val_metrics["trial"] = trial.number
        val_metrics["run_name"] = run_name
        pd.DataFrame([val_metrics]).to_csv(
            METRICS_CSV, mode='a', index=False, header=not os.path.exists(METRICS_CSV)
        )

        print(f"[{datetime.now().isoformat()}] Trial {trial.number} completed | NSE: {nse:.4f} | Duration: {time.time() - start_time:.1f} seconds")

        trial.report(-nse, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return -nse

    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Trial {trial.number} failed | Error: {e}")
        return 9999

if __name__ == "__main__":
    print("Starting hyperparameter tuning...")
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=2))
    study.optimize(objective, n_trials=10)

    df_trials = study.trials_dataframe()
    df_trials.to_csv(RESULTS_CSV, index=False)

    print("\nBest Hyperparameters Found:")
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print(f"Best NSE: {-study.best_value:.4f}")

    # Retrain the best model
    with open(BASE_CONFIG, "r") as f:
        final_config = yaml.safe_load(f)

    final_config.update({
        "model": "ealstm",
        "head": "regression",
        "learning_rate": {0: float(study.best_params["learning_rate"])},
        "output_dropout": float(study.best_params["output_dropout"]),
        "hidden_size": int(study.best_params["hidden_size"]),
        "batch_size": int(study.best_params["batch_size"]),
        "seq_length": 30,
        "epochs": 50,
        "run_dir": FINAL_RUN_DIR,
        "experiment_name": "best_final_run"
    })

    with open(FINAL_CONFIG_PATH, "w") as f:
        yaml.dump(final_config, f)

    print(f"\nRetraining best model in: {FINAL_RUN_DIR}")
    subprocess.run([
        "python", "-m", "neuralhydrology.nh_run",
        "train",
        "--config-file", FINAL_CONFIG_PATH
    ], check=True)

    #print(f"\nEvaluating best model on test set...")
    #subprocess.run([
     #  "python", "-m", "neuralhydrology.nh_run",
      #  "evaluate",
       # "--config-file", FINAL_CONFIG_PATH,
        #"--period", "test"
   # ], check=True)

from typing import Dict

import wandb


def init_run(project_name: str, run_name: str, config: Dict) -> None:
    wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        reinit=True,
    )


def log_metrics(metrics: Dict[str, float]) -> None:
    wandb.log(metrics)


def update_summary(values: Dict[str, float]) -> None:
    summary = wandb.run.summary
    for key, value in values.items():
        summary[key] = value


def finish_run() -> None:
    wandb.finish()

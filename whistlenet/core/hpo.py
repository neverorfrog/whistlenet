import lightning as L
import optuna

from config import WhistlenetConfig
from whistlenet.models import WhistleNet


def optuna_callback(study: optuna.study.Study, trial: optuna.Trial) -> None:
    print(f"Trial {trial.number} finished with value {trial.value}")


def hpo(config: WhistlenetConfig, data: L.LightningDataModule) -> None:
    """
    Perform hyperparameter optimization for the Baseline MLP model
    using Optuna.
    Args:
        config (Config): Configuration object containing hyperparameter
        settings and other configurations.
        data (L.LightningDataModule): LightningDataModule object containing
        the dataset.
    Returns:
        None
    This function initializes an Optuna study to optimize hyperparameters
    for the Whistlenet model. It defines an objective function
    that suggests hyperparameters, trains the model, and evaluates its
    performance. The best hyperparameters found during the optimization
    are then updated in the provided configuration object.
    The hyperparameters optimized include:
        - Learning rate
        - Kernel hidden channels
        - Kernel size
    In the end, the optimal parameters are written into the config object.
    """

    print("Starting a new hyperparameter optimization study...")

    def objective(
        trial: optuna.Trial,
        model: L.LightningModule,
        data: L.LightningDataModule,
    ) -> float:

        print("Starting a new trial...")
        print(f"Trial number: {trial.number}")

        # Hyperparameters to optimize
        config.optimizer.lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
        config.hidden_channels = trial.suggest_categorical(
            "hidden_channels", choices=[16, 24, 32]
        )
        config.kernel.hidden_channels = trial.suggest_categorical(
            "kernel_hidden_channels", choices=[16, 24, 32]
        )
        config.kernel.size = trial.suggest_categorical(
            "kernel_size", choices=[15, 27, 35, 51]
        )
        config.kernel.activation = trial.suggest_categorical(
            "activation", choices=["Sine", "LeakyReLU", "ReLU"]
        )

        print(f"Learning rate: {config.optimizer.lr}")
        print(f"Hidden Channels: {config.hidden_channels}")
        print(f"Kernel Hidden Channels: {config.kernel.hidden_channels}")
        print(f"Kernel Size: {config.kernel.size}")
        print(f"Kernel Activation: {config.kernel.activation}")

        trainer = L.Trainer(max_epochs=3)

        trainer.fit(model, data)
        optimized_value: float = trainer.logged_metrics["val/loss"]
        return optimized_value

    model = WhistleNet(1, 1, config)
    sampler = optuna.samplers.TPESampler(seed=32)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="whistlenet_hpo",
    )

    study.optimize(
        lambda trial: objective(trial, model, data),
        n_trials=5,
        callbacks=[optuna_callback],
    )

    print("Hyperparameter optimization completed.")
    print(
        f"Best hyperparameters found:\n\
          Learning rate: {study.best_params['lr']}\n\
          Hidden Channels: {study.best_params['hidden_channels']}\n\
          Kernel Hidden CHannels: {study.best_params['kernel_hidden_channels']}\n\
          Kernel Size: {study.best_params['kernel_size']}\n\
          Kernel Activation: {study.best_params['activation']}"
    )

    config.optimizer.lr = study.best_params["lr"]
    config.hidden_channels = study.best_params["hidden_channels"]
    config.kernel.hidden_channels = study.best_params["kernel_hidden_channels"]
    config.kernel.size = study.best_params["kernel_size"]
    config.kernel.activation = study.best_params["activation"]

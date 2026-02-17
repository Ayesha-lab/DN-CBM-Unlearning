from Models.mlp import MLP
import torch
from data_splitter import BinaryMNISTLoader
from utils import train_load_model
from Unlearning.certified_unlearning import CertifiedUnlearning
import itertools
from mlflow_setup import setup_mlflow
import mlflow
from collections import  namedtuple
import time
import logging
logging.getLogger("mlflow").setLevel(logging.WARNING)
import copy


"""" 
Create and load initial model: model_I and save for reuse
create dataset and split it into train and test set
further split train set into retain and forget set
train model_I on train set -> model_T
train model_I on retain set -> model_R
unlearn forget set on model_T -> model_U
"""

def main():
    ui_url = setup_mlflow()

    model_i = MLP(input_dim = 784, output_dim = 2)
    initial_state = copy.deepcopy(model_i.state_dict())
    torch.save(model_i.state_dict(), 'checkpoints/model_i.pth')

    params = {'load_model': [False],
              'lr': [0.05,0.01],
              'epochs': [20, 30],
              'batch_size': [128],
              'epsilon': [10, 1.0, 0.1],
              'delta': [1e-5],
              'ft_epochs': [6,3,1],
              'num_iterations': [6,1],
              'unlearning_lr': [0.0001],
              'lambda_reg': [10.0, 750]
              }

    Config = namedtuple('Config', params.keys())
    valid_combos = [Config(*c) for c in itertools.product(*params.values())]
    print("grid size: ", len(valid_combos))

    for cfg in valid_combos:
        with mlflow.start_run(run_name=f"lr_{cfg.lr}_epochs_{cfg.epochs}_bs_{cfg.batch_size}"):
            mlflow.log_params({
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "epsilon": cfg.epsilon,
                "delta": cfg.delta,
                "finetune_epochs": cfg.ft_epochs,
                "num_iterations": cfg.num_iterations,
                "unlearning_lr": cfg.unlearning_lr,
                "lambda_reg": cfg.lambda_reg
            })

            loader = BinaryMNISTLoader(data_dir='./Datasets', batch_size=cfg.batch_size)
            train_loader, test_loader, retain_loader, forget_loader = loader.load_data()

            print("Beginning initial trainings...", flush=True)
            model_t, c_0, c_1 = train_load_model('train', cfg,
                                                 model_i, train_loader)
            model_i.load_state_dict(initial_state)
            model_r, *_ = train_load_model('retain', cfg,
                                           model_i, retain_loader)

            unlearner = CertifiedUnlearning(model_t)

            model_u = unlearner.unlearn(
                retain_loader=retain_loader,
                epsilon=cfg.epsilon,
                delta=cfg.delta,
                num_iterations=cfg.num_iterations,
                learning_rate=cfg.unlearning_lr,
                lambda_reg= cfg.lambda_reg,
                clip_norm_0=c_0,
                clip_norm_1=c_1,
                verbose=True
            )
            torch.save(model_u.state_dict(), f'checkpoints/model_u_{cfg.lr}_{cfg.epochs}_{cfg.batch_size}.pth')

            model_ft  = unlearner.post_unlearn_finetune(
                retain_loader=retain_loader,
                num_epochs= cfg.ft_epochs,
                learning_rate=cfg.lr,
                weight_decay=0.0005,
                verbose=True
            )
            mlflow.log_params({'C0': c_0, 'C1':c_1})

            torch.save(model_ft.state_dict(), f'checkpoints/model_ft_{cfg.lr}_{cfg.epochs}_{cfg.batch_size}.pth')

if __name__ == "__main__":
    main()




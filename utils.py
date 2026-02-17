from train import train_model
import torch

def train_load_model(train_retain: str, cfg: tuple, model_i, data_loader):
    if cfg.load_model:
        checkpoint = torch.load(f'checkpoints/model_{train_retain[0]}_{cfg.lr}_{cfg.epochs}_{cfg.batch_size}.pth')
        model_i.load_state_dict(checkpoint['model_state_dict'])
        model = model_i
        c_0 = checkpoint['c_0']
        c_1 = checkpoint['c_1']
        print(f"{train_retain} model loaded from checkpoint")
    else:
        model, c_0, c_1 = train_model(model_i.state_dict(), data_loader, cfg.lr, cfg.epochs)
        print(f"Training model on {train_retain} dataset")
        torch.save({
            'model_state_dict': model.state_dict(),
            'c_0': c_0,
            'c_1': c_1
        }, f'checkpoints/model_{train_retain[0]}_{cfg.lr}_{cfg.epochs}_{cfg.batch_size}.pth')

    return model, c_0, c_1

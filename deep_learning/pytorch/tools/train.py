import _init_paths
import argparse
import torch
from torch.utils.data import DataLoader
from config.ConfigParams import ConfigParams
from data.Preprocessing import Preprocessing
from data.StdFileSystemDataset import StdFileSystemDataset
from model.ModelsFactory import ModelsFactory


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="PyTorch training script")
    parser.add_argument("--dataset_train_dir", required=True, type=str, help="Dataset train directory")
    parser.add_argument("--dataset_val_dir", required=True, type=str, help="Dataset validation directory")
    parser.add_argument("--config_file", required=True, type=str, help="Config file path")
    parser.add_argument("--model_output_path", required=False, type=str, default="./export/model.pth",
                        help="Filepath where to save the PyTorch model")
    args = parser.parse_args()
    return args


def main():

    # Load params
    args = do_parsing()
    print(args)

    # Load config file with model, hyperparameters and preprocessing
    config = ConfigParams(args.config_file)

    # Prepare preprocessing transform pipeline
    preprocessing_transforms = Preprocessing(config)
    preprocessing_transforms_train = preprocessing_transforms.get_transforms_train()
    preprocessing_transforms_val = preprocessing_transforms.get_transforms_val()

    # Read Dataset
    dataset_train = StdFileSystemDataset(args.dataset_train_dir, preprocessing_transforms_train)
    print("Train - Classes: {0}, Samples: {1}".format(str(len(dataset_train.get_classes())), str(len(dataset_train))))
    dataset_val = StdFileSystemDataset(args.dataset_val_dir, preprocessing_transforms_val)
    print("Validation - Classes: {0}, Samples: {1}".
          format(str(len(dataset_train.get_classes())), str(len(dataset_train))))

    # Load model and apply .train() and .cuda()
    model = ModelsFactory.create(config, len(dataset_train.get_classes()))
    print(model)
    model.cuda()
    model.train()

    # Create a PyTorch DataLoader from CatDogDataset (two of them: train + val)
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=8)

    # TODO: Set Optimizer and Loss

    # TODO: Do the training with validation evaluation after each epoch

    # Save model
    torch.save(model.state_dict(), args.model_output_path)

    print("End")


if __name__ == "__main__":
    main()

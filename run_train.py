"""entrypoint"""
import logging
from pathlib import Path
import sys
logger = logging.getLogger(__name__)

import bdb2025
import datasets
import modeling
import train

def parse_config(): 

    config = {
        "folder": Path("/Users/henrykraessig/code/bdb2025/data/"),
        "minimal": True,
        "weeks": 1,
        "interp": 40,
        "mask": .15,
        "train": .70,
        "val": .15,
        "test": .15,
        "batch_size": 32,
        "hidden_size": 64,
        "num_layers": 2,
        "num_heads": 2,
        "ffn_size": 64 * 4,
        "epochs": 100,
        "lr": 0.001,
        "patience": 10,
        "device": "cpu",
        "force": False
    }

    return config


def data_pipeline(config):
    # load
    if config["minimal"]:
        tracking = bdb2025.load_csv_data([Path("minimal_tracking.csv")])[0]
    else:
        tracking = bdb2025.load_tracking_data(folder=config["folder"])
    players = bdb2025.load_player_data(folder=config["folder"])
    plays = bdb2025.load_play_data(folder=config["folder"])
    # clean
    players = bdb2025.clean_player_data(players)
    tracking = bdb2025.clean_tracking_data(tracking)
    tracking = bdb2025.mirror_tracking_plays(tracking)
    static = bdb2025.prepare_static_data(tracking, plays, players)
    # prepare
    dynamic = datasets.interpolate_movement(tracking, config["interp"])

    return dynamic, static


def main(config: dict):
    dynamic_path = Path("./data/dynamic.npy")
    static_path = Path("./data/static.parquet")

    if dynamic_path.exists() and static_path.exists() and not config["force"]:
        dynamic = datasets.read_dynamic(dynamic_path)
        static = bdb2025.read_static(static_path)
    else:
        dynamic, static = data_pipeline(config)
        bdb2025.write_static(static)
        datasets.write_dynamic(dynamic)
    input, target = datasets.prepare_data_arrays(static, dynamic) # yeah, idc
    input, mask = datasets.mask_input(input, dynamic, config["mask"])
    n = input.shape[0]
    train_idx, val_idx, test_idx = datasets.train_val_test_split(n, config["train"], config["val"], config["test"])
    train_dataset = datasets.NflDataset(input, target, mask, train_idx)
    val_dataset = datasets.NflDataset(input, target, mask, val_idx)
    # debug
    # model
    model = modeling.NflBERT(
        input.shape[-1], # in 
        target.shape[-1], # out
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"], 
        num_heads=config["num_heads"], 
        ffn_size=config["ffn_size"],
    ).to(config["device"])
    # train
    train.train(config, model, train_dataset, val_dataset)
    modeling.save_model(model, Path("./models/model.pt"))
    
    return


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger("choreographer").setLevel(logging.WARNING)
    logging.getLogger("kaleido").setLevel(logging.WARNING)
    
    logger.info("starting")
    config = parse_config()

    main(config)
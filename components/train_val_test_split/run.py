"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import pandas as pd
import wandb
import os
import tempfile
from sklearn.model_selection import train_test_split
from wandb_utils.log_artifact import log_artifact

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="data_split")
    run.config.update(args)

    logging.info(f"Downloading artifact {args.input}")
    artifact_local_path = run.use_artifact(args.input).file()
    
    df = pd.read_csv(artifact_local_path)
    
    logger.info("Splitting the dataset")
    train_val, test = train_test_split(df, test_size=args.test_size, random_state=args.random_seed,
                                       stratify=df[args.stratify_by] if args.stratify_by != "none" else None)
    
    for df, split in zip([train_val, test], ["trainval", "test"]):
        logging.info(f"Uploading the {split}_data.csv dataset")

        with tempfile.NamedTemporaryFile("w",suffix="csv",delete=False) as fp:   

            df.to_csv(fp.name, index=False)

            artifact = wandb.Artifact(
                name=f"{split}_data.csv",
                type=f"{split}_data",
                description=f"{split}_split_of_dataset",
            )
            
            artifact.add_file(fp.name)
            
            logger.info(f"Logging artifact {split}_data.csv dataset")
            run.log_artifact(artifact)
            
            artifact.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")

    parser.add_argument("input", type=str, help="Input artifact to split")

    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )

    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )

    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )

    args = parser.parse_args()

    go(args)
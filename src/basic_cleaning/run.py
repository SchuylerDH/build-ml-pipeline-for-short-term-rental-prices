#!/usr/bin/env python
"""
download from wandb and apply some basic cleaning, exporting the result to a new artifact.
"""
import argparse
import logging
import wandb
import pandas as pd
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.info(f"DF dhape BEFORE dropping outliers: {df.shape}")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    logger.info(f"DF dhape AFTER dropping outliers: {df.shape}")

    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info(f"DF dhape BEFORE dropping rows outside of NYC: {df.shape}")
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    logger.info(f"DF dhape AFTER dropping rows outside of NYC: {df.shape}")

    logger.info("Data is cleaned.")

    # Save the cleaned dataset
    logger.info("Saving the output artifact")
    file_name = "clean_sample.csv"
    df.to_csv(file_name, index=False)

    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(file_name)

    logger.info("Logging artifact")
    run.log_artifact(artifact)
    
    os.remove(file_name)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cleaning of data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum number for price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum number for price",
        required=True
    )


    args = parser.parse_args()

    go(args)

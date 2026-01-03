import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import shutil
from typing import Text
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from modules.components import init_components

# Set pipeline name
PIPELINE_NAME = "diabetes-pipeline"

# Files for pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/diabetes_transform.py"
TRAINER_MODULE_FILE = "modules/diabetes_trainer.py"

# Files for pipeline outputs
OUTPUT_BASE = "output"
serving_model_dir = os.path.abspath(os.path.join(OUTPUT_BASE, "serving_model"))
pipeline_root = os.path.abspath(os.path.join(OUTPUT_BASE, PIPELINE_NAME))
metadata_path = os.path.abspath(os.path.join(pipeline_root, "metadata.sqlite"))


def init_local_pipeline(components, pipeline_root: Text) -> pipeline.Pipeline:
    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing",
        # auto-detect based on on the number of CPUs available during execution time.
        "--direct_num_workers=0",
        "--no_pipeline_type_check",
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        beam_pipeline_args=beam_args,
    )


logging.set_verbosity(logging.INFO)

components = init_components(
    DATA_ROOT,
    training_module=TRAINER_MODULE_FILE,
    transform_module=TRANSFORM_MODULE_FILE,
    training_steps=5000,
    eval_steps=1000,
    serving_model_dir=serving_model_dir,
)

pipeline = init_local_pipeline(components, pipeline_root)
BeamDagRunner().run(pipeline=pipeline)

# Get the latest pushed model from the internal TFX artifacts
pusher_dir = os.path.join(pipeline_root, "Pusher", "pushed_model")
latest_run = max([d for d in os.listdir(pusher_dir) if d.isdigit()], key=int)
source_path = os.path.join(pusher_dir, latest_run)

# The destination version folder created by the Pusher
dest_versions = [d for d in os.listdir(serving_model_dir) if d.isdigit()]
if dest_versions:
    latest_version = max(dest_versions, key=int)
    dest_path = os.path.join(serving_model_dir, latest_version)

    print(f"Syncing variables from {source_path} to {dest_path}...")

    # Copy the variables folder if it's missing
    src_vars = os.path.join(source_path, "variables")
    dst_vars = os.path.join(dest_path, "variables")

    if os.path.exists(src_vars):
        if os.path.exists(dst_vars):
            shutil.rmtree(dst_vars)
        shutil.copytree(src_vars, dst_vars)
        print("Variables synced successfully!")
    else:
        print("Error: Source variables folder not found in Pusher artifacts.")
else:
    print("Error: No version folder found in serving_model_dir.")

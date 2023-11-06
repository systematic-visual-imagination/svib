# SVIB: Systematic Visual Imagination Benchmark

- Official codebase associated with the paper *"Imagine the Unseen World: A Benchmark for Systematic Generalization in Visual World Models"* published at **NeurIPS'23**.
- Provides code for training and evaluating the baselines; and generating datasets as discussed in the paper.
- **Project Page**: [Systematic Visual Imagination Benchmark](https://systematic-visual-imagination.github.io/)

**Directory Structure Overview**
- **`configs`**: Default model hyperparameters.
- **`data_creation`**: Code for dataset creation.
- **`helpers`**: General helper functions.
- **`predictors`**: Decoder modules.
- **`representers`**: Image encoder modules referred to as *representers*.
- **`ssms`**: Modules for the state-space models.
- **`tasks`**: Main python scripts for task training and evaluation.


## Training Baselines Guide

### Setting Up the Environment

**Step 1:** Navigate to the `code` directory. Ensure the `code` directory (i.e., the project root) is added to the `PYTHONPATH`.

```shell
# Append the project root to PYTHONPATH
export PYTHONPATH=/path/to/code:$PYTHONPATH
```

### Image-to-Image Baselines
This section provides steps to train and evaluate the Image-to-Image baselines.

#### Training:
**Step 2.1:** Execute the baseline training script:
```shell
python tasks/image_to_image/train.py --data_path "data/" --log_path "logs/" --representer "vit" --predictor "patch_transformer"
```
- Monitor training curves via Tensorboard.
- Access saved checkpoints in the log directory specified by `--log_path`.
- Choose a representer with options like ViT or CNN using `--representer`.

#### Evaluation:
**Step 2.2:** Once models are trained, evaluate them with:
```shell
python tasks/image_to_image/eval.py --data_path "data/" --representer "vit" --predictor "patch_transformer" --load_path "saved_model.pt.tar"
```
- Ensure `--load_path` points to the model checkpoint from Step 2.1.


### SSM-based Baselines
This section outlines the process for training the State-Space Model (SSM) based baselines.

#### **Step 3.1: Training the State-Space Model**
1. Navigate to the desired directory based on your SSM choice:
    - `ssms/dynamic_slot` for SSM-Slot
    - `ssms/dynamic_vae` for SSM-VAE
2. Execute the training command:
```shell
python train.py --data_path "data/" --log_path "logs/"
```

#### **Step 3.2: Training the Decoder**
Train a decoder that acts as a probe on top of the pretrained SSM representation by running:
```shell
python tasks/image_to_image/train.py --data_path "data/" --log_path "logs/" --representer "dynamic_slot" --representer_path "path/to/representer_model.pt" --no_joint --predictor "patch_transformer"
```
Ensure the `--representer_path` points to the pretrained SSM model's saved checkpoint.

#### **Step 3.3: Evaluation**
After completing the training steps, evaluate the model:
```shell
python tasks/image_to_image/eval.py --data_path "data/" --representer "dynamic_slot" --predictor "patch_transformer" --load_path "saved_model.pt.tar"
```
- Make sure the `--load_path` directs to the checkpoint saved during the Step 3.2 training process.

### Oracle Baseline

This section provides a step-by-step guide to train and evaluate the Oracle baseline.

#### **Step 4.1: Training the Baseline**
Run the command below to train the Oracle baseline on the CLEVRTex dataset:
```shell
python tasks/gt_to_image/train.py --data_path "data/" --log_path "logs/" --data_reader "clevrtex"
```
To train on other datasets like dSprites or CLEVR, adjust the `--data_reader` and `--data_path` arguments accordingly.

#### **Step 4.2: Evaluating the Model**
After training, evaluate the model using:
```shell
python tasks/gt_to_image/eval.py --data_path "data/" --predictor "patch_transformer" --load_path "saved_model.pt.tar" --data_reader "clevrtex"
```
Ensure the `--load_path` directs to the checkpoint saved during the Step 4.1 training process.

## Dataset Generation Guide

### dSprites Tasks
To generate a dSprites task dataset, first navigate to the directory: 
```
data_creation/dsprites
```
Then, execute the command:
```
sh generate.sh {rule} {save_path} {train_alpha} {test_alpha}
```
Key points:
- `{train_alpha}`: A sequence of real numbers from 0.0 to 1.0 (e.g., "0.0 0.2 0.4 0.6").
- `{test_alpha}`: A single value between 0.0 and 1.0.

### CLEVR and CLEVRTex Tasks
For CLEVR and CLEVRTex tasks, change to either:
```
data_creation/clevr
```
or
```
data_creation/clevrtex
```
Next, run:
```
sh create.sh {rule} {save_path} {binds_file}
```
Here, `{binds_file}` refers to a file specifying the allowed factor combinations for a given split. Based on the required alpha, select the suitable binds file from the `precomputed_binds` directory.

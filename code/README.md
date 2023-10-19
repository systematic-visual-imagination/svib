# Systematic Visual Imagination Benchmark Codebase

This is the official code base for running the baselines and creating the datasets accompanying the NeurIPS'23 Datasets and Benchmark Track submission titled "Imagine the Unseen World: A Systematic Visual Imagination Benchmark".

## Running the Baselines

**Step 1:** Add the project root to PYTHONPATH.

```shell
export PYTHONPATH=/path/to/project_root:$PYTHONPATH
```

### Image-to-Image Baselines
In this section, we describe how one may train the Image-to-Image baselines.

**Step 2.1:** Run the baseline training script:

```shell
python tasks/image_to_image/train.py --data_path "data/" --log_path "logs/" --representer "vit" --predictor "patch_transformer"
```
Note that:
- You may monitor the training curves via Tensorboard and access the saved checkpoints in the log folder set by `--log_path`.
- You can set the desired representer, e.g., ViT or CNN via the `--representer` argument.

**Step 2.2:** After your task models are trained, you can run evaluation as follows:

```shell
python tasks/image_to_image/eval.py --data_path "data/" --representer "vit" --predictor "patch_transformer" --load_path "saved_model.pt.tar"
```
Note that:
- The load path set by the argument `--load_path` should point to a model checkpoint file saved during training when Step 2.1 was executed.

### SSM-based Baselines
In this section, we describe how one may train the SSM-based baselines.

**Step 3.1:** To train an SSM, navigate to the appropriate directory as follows:
- `ssms/dynamic_slot` for SSM-Slot
- `ssms/dynamic_vae` for SSM-VAE
Then, execute the following:
```shell
python train.py --data_path "data/" --log_path "logs/" \  
```
This will perform the training the desired state-space model.

**Step 3.2:** Next, train the decoder that acts as a probe on top of pre-trained SSM representation. For this, execute the following:
```shell
python tasks/image_to_image/train.py --data_path "data/" --log_path "logs/" --representer "dynamic_slot" --representer_path "path/to/representer_model.pt" --no_joint --predictor "patch_transformer"
```
where you need to point to the saved model of the pretrained SSM via the argument `--representer_path`.

**Step 3.3:** Finally, after both these training steps are complete, you can run evaluation as follows:

```shell
python tasks/image_to_image/eval.py --data_path "data/" --representer "dynamic_slot" --predictor "patch_transformer" --load_path "saved_model.pt.tar"
```
Note that:
- The load path set by the argument `--load_path` should point to a model checkpoint file saved during training when Step 3.2 was executed.


### Oracle
In this section, we describe how one may train the Oracle baseline.

**Step 4.1** Execute the following command to train the baseline.
```shell
python tasks/gt_to_image/train.py --data_path "data/" --log_path "logs/" --data_reader "clevrtex"
```
Here, we took the example of training on CLEVRTex by setting the `--data_reader` to `clevrtex`. By appropriately setting this argument, one may execute this code for dSprites and CLEVR as well.

**Step 4.2** Execute the following command to evaluate the trained baseline.
```shell
python tasks/gt_to_image/eval.py --data_path "data/" --predictor "patch_transformer" --load_path "saved_model.pt.tar" --data_reader "clevrtex"
```
Note that:
- The load path set by the argument `--load_path` should point to a model checkpoint file saved during training when Step 4.1 was executed.


## Creating Datasets


### dSprites Tasks
To make a dSprites tasks, navigate to `data_creation/dsprites` and run the following command:

```
sh generate.sh {rule} {save path} {train alpha} {test alpha}
```
Note that:
- The train alpha is a sequence of real numbers ranging from 0.0 to 1.0. (e.g., "0.0 0.2 0.4 0.6")
- The test alpha should be in the range of 0.0 to 1.0.

### CLEVR and CLEVRTex Tasks
To make the CLEVR and CLEVRTex tasks, navigate to the appropriate directory i.e., `data_creation/clevr` or `data_creation/clevrtex`. Then run the following script:

```
sh create.sh {rule} {save path} {binds file}
```
where the argument indicated by "binds file" denotes a file listing the allowed factor combinations to expose in this split. Depending on the desired alpha, one can choose the appropriate binds file from the directory `precomputed_binds`.

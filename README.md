# Fork-and-use



Training deep learning models can be repetitive and monotonous, involving tasks such as creating environments, defining models and datasets, and writing training scripts.



To streamline this process, I created this repository for `fork-and-use` to enable faster experimentation. The design allows for minimal changes to run the experiments.



I used the technology stack as below:



## Docker



**Reproducibility** is important, so Docker ensures consistency in dependencies and configurations, and is faster than setting up an `anaconda` environment.



**For installation**:

1. Install [Docker engine](https://docs.docker.com/engine/install/).

2. To be able to use GPU-accelerated containers, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

3. Need to build the image: `docker build -t "$(DOCKER_IMG)" .`

4. Run the container.



```bash

docker  run  -d  --gpus  all  -v  ".:/code"  -v  "$(DATA_DIR):/mnt"  --env-file  "$(ENV_FILE)"  \

--shm-size=8gb -p 8080:8080 --name "$(DOCKER_CONTAINER)" "$(DOCKER_IMG)" tail -f /dev/null

```



-  `-v`: Mounts the SSD and current folder to enable code changes without rerunning the container, also providing access to the SSD.

-  `--env-file`: Defines environment variables inside the container. Currently, the only environment variable I use is the WANDB API KEY.

-  `--shm-size`: The default shared memory size in Docker is only 64MB, which is very low. For loading images, a larger RAM size is required. Define it through this command.



5. Access the container: `docker exec -it "$(DOCKER_CONTAINER)" /bin/bash`



6. Stop & Remove: `docker stop "$(DOCKER_CONTAINER)" & docker rm "$(DOCKER_CONTAINER)"`



## Training Packages



-  `Training`: Used for the training loop due to its convenient, flexible interface for multi-GPU training. Learn more about [PyTorch Lightning](https://www.pytorchlightning.ai/).

- Logging:

- Wandb: The API KEY should be defined in the `.env` file. Learn more about [Weights & Biases](https://wandb.ai/).



- Evaluation:

- Torchmetrics: For evaluation metrics. Learn more about [Torchmetrics](https://torchmetrics.readthedocs.io/en/latest/).



Refer to the `train_script.py` & `classification_learner.py` files. The configuration YAML is defined in the `configs` folder, where you can select callbacks and trainer settings.



-  `Hyperparameter search`: **Ray[tune]** - Used for hyperparameter selection. Learn more about [Ray Tune](https://docs.ray.io/en/latest/tune/index.html).



Refer to the `tune_script.py`. Since hyperparameters usually include batch size and learning rate, I define the search space within the script file. In the future, I'll load it from a config file.



Ensure that the max batch size/largest model configuration will not cause a `CUDA Out of Memory` error.



## Repo structure



-  `config.yaml`:

- A file inside of `configs` folder that specifies hyperparameters for training such as optimizer, scheduler, dataloader specifications, etc.

- In `utils.py`, config YAML is parsed and appropriate modules (like optimizer, the model, etc) are loaded.



-  `pre-commit`:

- To have clean and organized code, I use pre-commit hooks. Right before committing the changes to the repo, it checks the repo for the issues defined in the `.pre-commit-config.yaml`.



-  `pyproject.toml`:

- Settings for `pylint, black, isort` are defined. Necessary for `pre-commit`.



-  `Makefile`:

- Automates several common CLI commands. The names or paths should be configured in the file.

- After forking / cloning the repo, type `make setup` to install the `pre-commit hooks`.

- Additionally, Docker and git commands are defined.



# Toy Experiments



## ResNet classification on CIFAR-10



As an example, I did small computer vision training for CIFAR-10 classification with [Wide ResNet](https://arxiv.org/abs/1605.07146). In `models/wide_resnet.py`, I implemented the paper. The dataset class is already defined in `torchvision`, just need to pass augmentation functions (transforms) while initializing them.



Currently, both transform and dataset class are imported from `utils.py`. Ideally, Dataset must be defined `dataset` folder.



To tune write command: `python tune_script --config_path configs/wide_resnet_cifar.yaml`



To train: `python train_script --config_path configs/wide_resnet_cifar.yaml`.



# To Do



- [x] Repo description and Reproducibility

- [ ] FSDP Distributed Training for LLMs

- [ ] Model deployment with Gradio

- [ ] Multi-node Training



> Note: This repo was created for personal use. I'd be happy for your suggestions for improvement, or to fix potential errors.
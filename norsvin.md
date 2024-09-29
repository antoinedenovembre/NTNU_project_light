# How to make the Norsvin docker work

## Install CUDA/Docker tools

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Launch the docker

```bash
cd ../norsvin_model
./norsvin_docker_manager.sh
```

## Run the training

In the docker, run the following command:
```bash
python /app/training/training.py
```
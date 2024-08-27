# NTNU EfficientDet training

![NorsvinLogo](https://www.facebook.com/NorsvinSA/)

Deep learning project about animal behaviour.

---

## Features

For now, the project includes:

- EfficientDet model training and saving
- EfficientDet evaluation (including metrics and image testing)

---

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.10+

---

## Installation

Step-by-step guide on how to install the project:

```bash
git clone https://github.com/antoinedenovembre/NTNU_project.git
cd NTNU_project
python install_requirements.py
```

**/!\ Important: You also need to make sure you have the following structure, including the `data` folder**
```
NTNU_project_light
│   README.md
│   main.py
│   install_requirements.py
│
└───data
│   └───train
│   │   │   annotations
│   │   │   │   train.json
│   │   │ 
│   │   │   images
│   │   │   │   image1.jpg/png/...
│   │   │   │   ...
│   │
│   └───val
│       │   annotations
│       │   │   val.json
│       │
│       │   images
│           │   image1.jpg/png/...
│           │   ...
│
└───efficient_det
│
└───utils
│
└───scripts
```

---

## Usage

Here is the command to run the project:

```bash
python main.py
```

---

## Documentation

[Documentation](documentation/paper.pdf)

## Q&As

### What PC can be used?

- Ideally one with a sufficient GPU, like NVIDIA RTX 2080Ti

### What OS can be used?

- Any Linux distro should do the trick, but I recommend using Ubuntu 22.04+

---

## Contributing

This repository is a fork from [this repository](https://github.com/FayazRahman/barlow-effdet)

## Authors and Acknowledgment

Show your appreciation to those who have contributed to the project.

- [@FayazRahman](https://github.com/FayazRahman) - Contribution
- [@antoinedenovembre](https://github.com/antoinedenovembre) - Contribution


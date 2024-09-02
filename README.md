# NTNU EfficientDet training

<p align="center">
    <img src="documentation/logo.png" width="30%" />
</p>

## Introduction

Deep learning project about animal behaviour.

## Features

For now, the project includes:

- EfficientDet model (no backbone) training and saving
- EfficientDet model (no backbone) evaluation (including metrics and image testing)
- EfficientNet backbone training and saving
- EfficientDet with EfficientNet backbone training and saving
- EfficientDet with EfficientNet backbone evaluation (including metrics and image testing)

## Prerequisites

Before you begin, ensure you have met the following requirement:

- Python 3.10+

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
│   └───backbone
│   │   │   image1.jpg/png/...
│   │   │   ...
│   │
│   └───effdet
│       └─── train
│       │    └─── annotations
│       │    │    │   annotations.json
│       │    │    │   ...
│       │    │
│       │    └─── images
│       │         │   image1.jpg/png/...
│       │         │   ...
│       │
│       └─── test
│            └─── annotations
│            │    │   annotations.json
│            │    │   ...
│            │
│            └─── images
│                 │   image1.jpg/png/...
│                 │   ...
│
└───efficient_det
│
└───barlow
│
└───utils
│
└───scripts
│
└───documentation
│
└───output
```

The annotations shall have the following structure
```
{
  "annotations": [
    {
      "area": 87292,
      "bbox": [
        576,
        98,
        547,
        204
      ],
      "category_id": 6,
      "id": 1, <!-- Should be the number of the annotation -->
      "image_id": 2, <!-- Should be the number of the image -->
      "iscrowd": 0
    },
    ...
    ]
}
```

## Usage

Here is the command to run the project:

```bash
python main.py
```

## Documentation

Overall view of the project:
[Documentation - Overall view](documentation/paper.pdf)

Technical documentation about evaluation metrics and loss function:
[Documentation - Metrics](documentation/metrics.md)

## Q&As

### What PC can be used?

- Ideally one with a sufficient GPU, like NVIDIA RTX 2080Ti

### What OS can be used?

- Any Linux distro should do the trick, but I recommend using Ubuntu 22.04+

## Contributing

This repository is a fork from [this repository](https://github.com/FayazRahman/barlow-effdet)

## Authors and Acknowledgment

Show your appreciation to those who have contributed to the project.

- [@FayazRahman](https://github.com/FayazRahman) - Contribution
- Dev Narayan Chakkalakkal Binu - Contribution
- [@antoinedenovembre](https://github.com/antoinedenovembre) - Contribution


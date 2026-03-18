# SAIDA: Synthetic-Augmented Iterative Domain Adaptation

## Overview

This repository contains the code to reproduce CD-FSOD challenge submission results.

- data_name: [dataset1, dataset2, dataset3]
- shot: [1, 5, 10]

```plaintext
SAIDA/
├── Dockerfile
│
├── NTIRE2026/
│   ├── dataset1/
│   │   ├── annotations/
│   │   ├── train/
│   │   └── test/
│   ├── dataset2/
│   │   ├── annotations/
│   │   ├── train/
│   │   └── test/
│   └── dataset3/
│       ├── annotations/
│       ├── train/
│       └── test/
│
├── weights/
│   └── codetr-[data_name]-[shot].pth
│
├── download.sh
│
├── executable.sh
...
```

## Build

```bash
git submodule update --init --recursive
DOCKER_BUILDKIT=1 docker build -t saida .
```


## Preparation

#### 1. Challenge dataset

Place the challenge dataset ([google drive](https://drive.google.com/drive/folders/19Ylfklp2TW_HijR2ZrVt-ayMNQfo__T2)) under `./NTIRE2026/`.


#### 2. Model checkpoints

Place the fine-tuned Co-DETR model checkpoints under `./weights/`:

```bash
bash download.sh
```


## Inference

Set the `SERVER_IP` environment variable to point to the server hosting the fine-tuned ZERO model.
This enables API requests to retrieve predictions for arbitrary test images.

For security reasons, the server is only temporarily exposed, and access is restricted to the challenge organizers.
The `SERVER_IP` has been shared privately with the organizers via email.

```bash
export SERVER_IP="XXX.XXX.XXX.XXX"
```

Then, run the following command to generate prediction results in the `./submit/` directory:

```bash
bash executable.sh
```

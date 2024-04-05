<p align="center">
    <h1 align="center">anomaly detection</h1>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/pierg/anomaly_detection?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/pierg/anomaly_detection?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/pierg/anomaly_detection?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/pierg/anomaly_detection?style=default&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
</p>



##  Overview

This project develops an anomaly detection system using various deep learning models, including Transformer and LSTM architectures, to identify anomalies in event data streams. It benchmarks against traditional models like [DeepLog](https://github.com/Thijsvanede/DeepLog), using precision, recall, and F1 score metrics. The system supports Slurm job management and offers a general framework for modeling event data streams. Validated on the [Hadoop HDFS dataset](https://github.com/logpai/loghub/blob/master/HDFS/README.md), it demonstrates effectiveness in real-world scenarios.

---



##  Getting Started

**System Requirements:**

* **Python**: `version 3.11`
* **[*Poetry*](https://python-poetry.org/docs/)** for dependencies management

###  Installation

1. Clone the anomaly_detection repository:

```console
$ git clone https://github.com/pierg/anomaly_detection
```

2. Change to the project directory:
```console
$ cd anomaly_detection
```

3. Install the dependencies:
```console
$ poetry install
```

###  Usage

Run anomaly_detection using the command below:
```console
$ python main.py
```


---


##  Repository Structure

```sh
└── anomaly_detection/
    ├── README.md
    ├── anomaly_detection
    │   ├── __init__.py
    │   ├── configs
    │   ├── data
    │   ├── main.py
    │   ├── models
    │   ├── optimizers
    │   ├── series
    │   ├── trainers
    │   └── utils
    ├── data
    │   └── hdfs_deeplog
    ├── pyproject.toml
    └── slurm
        ├── cacel.sh
        ├── clean_up.sh
        ├── j_main.sh
        ├── logs.sh
        ├── queue.sh
        └── submit.sh
```

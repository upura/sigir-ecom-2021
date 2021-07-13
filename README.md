# The 3rd Place Solution of Purchase Intent Prediction in SIGIR eCOM 2021 Data Challenge

This repository contains code for the 3rd place solution of purchase intent prediction in [SIGIR eCOM 2021 Data Challenge](https://sigir-ecom.github.io/data-task.html).

## Summary

Our best result was given by the weighted averaging of the following two models.

- Gradient boosting decision trees (GBDT)
- Neural Networks (NN)

Note that the task of this competition is difficult so that most participants could not exceed a baseline just predicting all samples as negative.
One of the important points is to figure out the clues which can break the baseline.
We applied adversarial validation methodology, creating a classifier for train and test data, for validation data selection.
This evaluation approach guided us, and we decided to use prospective model for the specific test data.
More detailed information will be available on the technical paper (to appear), the [slides](https://speakerdeck.com/upura/adversarial-validation-to-select-validation-data-for-evaluating-performance-in-e-commerce-purchase-intent-prediction), and the [video](https://youtu.be/Vs24X6L88rQ).

## Implementation

The implementation of GBDT used a supporting tool for machine learning competitions named [Ayniy](https://github.com/upura/ayniy).

### Data Preparation

```bash
docker-compose up -d
docker exec -it sigir2021 bash
```
```bash
cd experiments
python prepare_df.py
python prepare_cart_df.py
python prepare_cart_Xy.py
```

### Training GBDT

```bash
docker-compose up -d
docker exec -it sigir2021 bash
```
```bash
cd experiments
python runner.py --run configs/run000.yml
python runner.py --run configs/run001.yml
python runner.py --run configs/run002.yml
python runner.py --run configs/run003.yml
python runner.py --run configs/run004.yml
python runner.py --run configs/run005.yml
```

### Training NN

GPU on Google Colab was used for running this script.

```bash
cd experiments
pip install ../requirements_colab.txt
python train_nn.py
```

### Rule-based

```bash
docker-compose up -d
docker exec -it sigir2021 bash
```
```bash
cd experiments
rule_based_submission.py
```

### Adversarial validation

```bash
docker-compose up -d
docker exec -it sigir2021 bash
```
```bash
cd experiments
python runner.py --run configs/run008.yml
python runner.py --run configs/run009.yml
python runner.py --run configs/run010.yml
python runner.py --run configs/run011.yml
python runner.py --run configs/run012.yml
python runner.py --run configs/run013.yml
python prepare_cart_val.py
```

### Weighted averaging

```bash
docker-compose up -d
docker exec -it sigir2021 bash
```
```bash
cd experiments
python weighted_averaging.py
```

# SIGIR eCOM 2021 Data Challenge

This repository contains code for [SIGIR eCOM 2021 Data Challenge](https://sigir-ecom.github.io/data-task.html).

See also:

- [official beselines repository](https://github.com/coveooss/SIGIR-ecom-data-challenge)
- [first teammate repository](https://github.com/hakubishin3/sigir-ecom-2021)
- [second teammate repository](https://github.com/koukyo1994/sigir2021)

## Procedure

```bash
docker-compose up -d
docker exec -it sigir2021 bash
```
```
cd experiments
python prepare_df.py
python prepare_cart_df.py
python prepare_cart_Xy.py
sh runner.sh
python weighted_averaging.py
```

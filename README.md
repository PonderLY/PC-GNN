# PC-GNN

This is the author implementation of "[Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection](https://dl.acm.org/doi/abs/10.1145/3442381.3449989)" (WebConf 2021).

Slides and video can be found [here](https://ponderly.github.io/).

[Yang Liu](https://ponderly.github.io/), [Xiang Ao](https://aoxaustin.github.io/), Zidi Qin, Jianfeng Chi, Jinghua Feng, Hao Yang and [Qing He](http://people.ucas.ac.cn/~heqing?language=en). 

## Requirements

```
argparse          1.1.0
networkx          1.11
numpy             1.16.4
scikit_learn      0.21rc2
scipy             1.2.1
torch             1.4.0
```

## Dataset

YelpChi and Amazon can be downloaded from [here](https://github.com/YingtongDou/CARE-GNN/tree/master/data) or [dgl.data.FraudDataset](https://docs.dgl.ai/api/python/dgl.data.html#fraud-dataset).

Put them in `/data` directory and run `unzip /data/Amazon.zip` and `unzip /data/YelpChi.zip` to unzip the datasets.

Run `python src/data_process.py` to pre-process the data.

Kindly note that there may be two versions of node features for YelpChi. The old version has a dimension of 100 and the new version is 32. In our paper, the results are reported based on the old features.

## Usage

```sh
python main.py --config ./config/pcgnn_yelpchi.yml
```

## Citation

```
@inproceedings{liu2021pick,
  title={Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection},
  author={Liu, Yang and Ao, Xiang and Qin, Zidi and Chi, Jianfeng and Feng, Jinghua and Yang, Hao and He, Qing},
  booktitle={Proceedings of the Web Conference 2021},
  pages={3168--3177},
  year={2021}
}
```

## Other implementations

Thanks for [Jack Huang](https://github.com/f26401004/PC-GNN) and [Ronald D. R. Pereira](https://github.com/ronaldpereira/pick-and-choose-gnn) for their kind implementations.
## A BiLSTM+GCN Encoder for Target-oriented Opinion Words Extraction

This repo contains the official PyTorch implementation on the best performing model (BiLSTM+GCN) for paper [An Empirical Study on Leveraging Position Embeddings for Target-oriented Opinion Words Extraction](https://arxiv.org/abs/2109.01238) accepted at EMNLP2021.

**The TOWE datasets**: Details on the TOWE Datasets can be found in [(Fan et al., 2019)](https://aclanthology.org/N19-1259.pdf).

### Requirements

- Python 3 (tested on 3.6.13)
- PyTorch (tested on 1.9.0)

### Preparation

First, download and unzip `300-dim` GloVe vectors from the Stanford website into the directory `data/glove/`
```
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d data/glove/
rm glove.840B.300d.zip
```

For each dataset (e.g., 14lap), prepare vocabulary and initial word vectors with:

```python prepare_vocab.py --dataset 14lap --wv_file glove.840B.300d.txt```

This will write `vocab.pkl` and `embedding.npy` into the dir `data/14lap`.

### Training and Evaluation

Train and evaluate a BiLSTM+GCN encoder with:

```python train.py --dataset 14lap --gcn_layers 1 --save_dir best_model_log```

This will train a BiLSTM+GCN encoder with a single GCN layer.

The argument --gcn_layers takes an integer for the encoder

Training a BiLSTM encoder has the corresponding `gcn_layers: 0`.

### Alternative Training and Evaluation

Train and evaluate all datasets for different number of layers

```python run1.py```

### Citation

We currently have the arxiv [paper](https://arxiv.org/abs/2109.01238) you can cite if this code is beneficial to you:
```bibtex
@article{mensah2021empirical,
  title={An Empirical Study on Leveraging Position Embeddings for Target-oriented Opinion Words Extraction},
  author={Mensah, Samuel and Sun, Kai and Aletras, Nikolaos},
  journal={arXiv preprint arXiv:2109.01238},
  year={2021}
}
```

# On the Unexpected Effectiveness of Reinforcement Learning for Sequential Recommendation

In recent years, Reinforcement Learning (RL) has shown great promise in session-based recommendation. Sequential models that use RL have reached state-of-the-art performance for the Next-item Prediction (NIP) task. This result is intriguing, as the NIP task only evaluates how well the system can correctly recommend the next item to the user, while the goal of RL is to find a policy that optimizes rewards in the long term– sometimes at the expense of suboptimal shortterm performance. Then, how can RL improve the system’s performance on short-term metrics? This article investigates this question by exploring proxy learning objectives, which we identify as goals RL models might be following, and thus could explain the performance boost. We found that RL– when used as an auxiliary loss– promotes the learning of embeddings that capture information about the user’s previously interacted items. Subsequently, we replaced the RL objective with a straightforward auxiliary loss designed to predict the number of items the user interacted with. This substitution results in performance gains comparable to RL. These findings pave the way to improve performance and understanding of RL methods for recommender systems.

This repository contains the code used for the paper ***On the Unexpected Effectiveness of Reinforcement Learning for Sequential Recommendation*** ([ICML 2024](https://openreview.net/pdf?id=ie3vXkMvRY))

```bibtex
@inproceedings{lab-etal-icml24,
  title={On the Unexpected Effectiveness of Reinforcement Learning for Sequential Recommendation},
  author={Silva, {\'A}lvaro Labarca and Parra, Denis and Icarte, Rodrigo Toro},
  booktitle={Proceedings of the 41st International Conference on Machine Learning (ICML)}
  year={2024}
}
```

Our methods build on top of the SQN model ([SIGIR 2020]([https://openreview.net/pdf?id=ie3vXkMvRY](https://www.researchgate.net/profile/Alexandros-Karatzoglou/publication/342093511_Self-Supervised_Reinforcement_Learning_for_Recommender_Systems/links/5ee73cf1a6fdcc73be7bbc67/Self-Supervised-Reinforcement-Learning-for-Recommender-Systems.pdf)))

```bibtex
@inproceedings{xin-etal-sigir20,
  title={Self-supervised reinforcement learning for recommender systems},
  author={Xin, Xin and Karatzoglou, Alexandros and Arapakis, Ioannis and Jose, Joemon M},
  booktitle={Proceedings of the 43rd International ACM Special Interest Group on Information Retrieval Conference (SIGIR)},
  pages={931--940},
  year={2020}
}
```

This code is meant to be a clean and usable version of our approach. Please let us know if you find any bugs or have questions about it. We'll be happy to help you!

# Installation instructions

The code has the following requirements:

* Python 3.5
* tensorflow 1.13.1
* Pandas
* dm-tree 0.1.1
* trf
* tensorflow-probability 0.6.0
* scikit-learn

# How to Run the Code

The repository is divided between the two datasets that were used in the paper, RC15 and RetailRocket. Within each of these folders, the algorithms are subdivided into three different sections:
* **baselines**: Contain the self-supervised sequential algorithms used as a baseline for the results. Running `python Caser.py --data=<path_to_data_directory>` will output the result of one run with the Caser baseline.
* **RL approaches:** This folder contains the **SQN** and **SAC** algorithms as proposed by **Xin et al. (2020)**, as well as the **EVAL** model, which modifies the loss function of the SQN model. **RL_preds.py** is a modified version of the SQN model which removes the self-supervised, cross-entropy based head and computes the loss and predictions directly from the RL head. Its performance can be found in Appendix E. of the ([paper](https://openreview.net/pdf?id=ie3vXkMvRY)).
* **Proxy Approaches:** This folder contains the files to run the **CAT**, **CAT3**, **FUT** and **HIST** models, which replace the RL head of SQN with a proxy-learning objective function.

For each algorithm in **RL approaches** and **Proxy Approaches**, the flag `--model=` can be used to select which base algorithm (GRU, Caser, NItNet or SASRec) will be used for one run of the model. Alternatively, the flag `--runs={X}` can be used to run X runs of every base model sequentially. Meaning, `python CAT.py --runs=2` will sequentially run: `GRU-CAT, Caser-CAT, NIN-CAT, SAS-CAT, GRU-CAT, Caser-CAT, NIN-CAT, SAS-CAT`. By default, the code will do one run of each model.

# Examples

```CMD
python SQN.py --data=<path_to_data_directory> --model=NItNet  #One run of NIN-SQN
python FUT.py --data=<path_to_data_directory> --runs=1        #One run of the FUT model for each base model
python FUT.py --data=<path_to_data_directory>                 #One run of the FUT model for each base model
python SASRec.py --data=<path_to_data_directory>              #One run of SASRec
```

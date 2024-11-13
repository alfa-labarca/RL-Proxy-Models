# Revision Notice: October 2024

Following the publication of this article, we extended our research by testing variations of our proxy models. During this process, we identified a bug in the proxy models that caused the proxy loss to target the incorrect feature column in both the *-HIST (GRU-HIST, Caser-HIST, NIN-HIST, SAS-HIST) and *-FUT models.
 
To address this, we reran the affected models on both datasets (RetailRocket and RC15) and included new results for the *-FUT models on the RC15 dataset, as these were absent in the original publication. The revised results, alongside the original published metrics, are presented in the tables below:

## Revised results on RetailRocket dataset
![image](https://github.com/user-attachments/assets/adc7a8bd-46b1-461a-9184-feaccbc771e8)

![image](https://github.com/user-attachments/assets/40e03e09-ffbf-4b19-b5da-6834dfbe5cdd)

## Revised results on RC15 (Yoochoose) dataset
![image](https://github.com/user-attachments/assets/f94a70ef-e1df-4942-8a32-fb5d4cff0d77)

![image](https://github.com/user-attachments/assets/80630591-e2b5-4d2c-96ba-5a8dae403966)

  
While the revised results show some differences, the overall conclusions remain unchanged: the proxy methods consistently outperform the baseline versions, with results closely matching those obtained by the RL-based *-DQN method. However, one notable change is that, in this revised version, the *-FUT proxy method performs as well as, and occasionally better than, the *-HIST method.
 
If you have any questions, do not hesitate sending us a message to:
* Alvaro Labarca-Silva: aalabarca (at) uc (dot) cl
* Denis Parra: dparras (at) uc (dot) cl
* Rodrigo Toro: rntoro (at) uc (dot) cl

# On the Unexpected Effectiveness of Reinforcement Learning for Sequential Recommendation

In recent years, Reinforcement Learning (RL) has shown great promise in session-based recommendation. Sequential models that use RL have reached state-of-the-art performance for the Next-item Prediction (NIP) task. This result is intriguing, as the NIP task only evaluates how well the system can correctly recommend the next item to the user, while the goal of RL is to find a policy that optimizes rewards in the long term– sometimes at the expense of suboptimal shortterm performance. Then, how can RL improve the system’s performance on short-term metrics? This article investigates this question by exploring proxy learning objectives, which we identify as goals RL models might be following, and thus could explain the performance boost. We found that RL– when used as an auxiliary loss– promotes the learning of embeddings that capture information about the user’s previously interacted items. Subsequently, we replaced the RL objective with a straightforward auxiliary loss designed to predict the number of items the user interacted with. This substitution results in performance gains comparable to RL. These findings pave the way to improve performance and understanding of RL methods for recommender systems.

# Description of this repository

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

To use our code, first unzip the datasets. Then, you can use the commands from [scripts.sh](https://github.com/alfa-labarca/RL-Proxy-Models/blob/main/scripts.sh) to replicate the results from our paper. For example, the following commands run a GRU model with (and without) the different auxiliary losses studied in the paper: 

```bash
python3 Baselines/GRU.py --data=Datasets/RetailRocket --lr=0.005
python3 Proxy\ Approaches/HIST.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 Proxy\ Approaches/CAT.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 Proxy\ Approaches/FUT.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 RL\ approaches/EVAL.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 RL\ approaches/SQN.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
```

The code is divided into three folders:
* **baselines**: This folder contains the self-supervised sequential algorithms (**GRU**, **Caser**, **NextItNet**, and **SASRec**).
* **RL approaches:** This folder contains the **SQN** and **SAC** algorithms as proposed by **Xin et al. (2020)**, as well as the **EVAL** model, which modifies the loss function of the SQN model. **RL_preds.py** is a modified version of the SQN model -- which removes the self-supervised head and computes the loss and predictions directly from the RL head. Its performance can be found in Appendix E. of the ([paper](https://openreview.net/pdf?id=ie3vXkMvRY)).
* **Proxy Approaches:** This folder contains the files to run the **CAT**, **CAT3**, **FUT**, and **HIST** models -- which replace the RL head of SQN with a proxy-learning objective function.

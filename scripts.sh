

# Retail Rocket experiments

python3 Baselines/GRU.py --data=Datasets/RetailRocket --lr=0.005
python3 Baselines/Caser.py --data=Datasets/RetailRocket --lr=0.005
python3 Baselines/NextItNet.py --data=Datasets/RetailRocket --lr=0.005
python3 Baselines/SASRec.py --data=Datasets/RetailRocket --lr=0.005

python3 Proxy\ Approaches/HIST.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 Proxy\ Approaches/HIST.py --data=Datasets/RetailRocket --model=Caser --lr=0.005
python3 Proxy\ Approaches/HIST.py --data=Datasets/RetailRocket --model=NItNet --lr=0.005
python3 Proxy\ Approaches/HIST.py --data=Datasets/RetailRocket --model=SASRec --lr=0.005

python3 Proxy\ Approaches/CAT.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 Proxy\ Approaches/CAT.py --data=Datasets/RetailRocket --model=Caser --lr=0.005
python3 Proxy\ Approaches/CAT.py --data=Datasets/RetailRocket --model=NItNet --lr=0.005
python3 Proxy\ Approaches/CAT.py --data=Datasets/RetailRocket --model=SASRec --lr=0.005

python3 Proxy\ Approaches/CAT3.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 Proxy\ Approaches/CAT3.py --data=Datasets/RetailRocket --model=Caser --lr=0.005
python3 Proxy\ Approaches/CAT3.py --data=Datasets/RetailRocket --model=NItNet --lr=0.005
python3 Proxy\ Approaches/CAT3.py --data=Datasets/RetailRocket --model=SASRec --lr=0.005

python3 Proxy\ Approaches/FUT.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 Proxy\ Approaches/FUT.py --data=Datasets/RetailRocket --model=Caser --lr=0.005
python3 Proxy\ Approaches/FUT.py --data=Datasets/RetailRocket --model=NItNet --lr=0.005
python3 Proxy\ Approaches/FUT.py --data=Datasets/RetailRocket --model=SASRec --lr=0.005

python3 RL\ approaches/EVAL.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 RL\ approaches/EVAL.py --data=Datasets/RetailRocket --model=Caser --lr=0.005
python3 RL\ approaches/EVAL.py --data=Datasets/RetailRocket --model=NItNet --lr=0.005
python3 RL\ approaches/EVAL.py --data=Datasets/RetailRocket --model=SASRec --lr=0.005

python3 RL\ approaches/SQN.py --data=Datasets/RetailRocket --model=GRU --lr=0.005
python3 RL\ approaches/SQN.py --data=Datasets/RetailRocket --model=Caser --lr=0.005
python3 RL\ approaches/SQN.py --data=Datasets/RetailRocket --model=NItNet --lr=0.005
python3 RL\ approaches/SQN.py --data=Datasets/RetailRocket --model=SASRec --lr=0.005


# RC experiments

python3 Baselines/GRU.py --data=Datasets/RC15 --lr=0.005
python3 Baselines/Caser.py --data=Datasets/RC15 --lr=0.01
python3 Baselines/NextItNet.py --data=Datasets/RC15 --lr=0.01
python3 Baselines/SASRec.py --data=Datasets/RC15 --lr=0.01

python3 Proxy\ Approaches/HIST.py --data=Datasets/RC15 --model=GRU --lr=0.005
python3 Proxy\ Approaches/HIST.py --data=Datasets/RC15 --model=Caser --lr=0.01
python3 Proxy\ Approaches/HIST.py --data=Datasets/RC15 --model=NItNet --lr=0.01
python3 Proxy\ Approaches/HIST.py --data=Datasets/RC15 --model=SASRec --lr=0.01

python3 Proxy\ Approaches/CAT.py --data=Datasets/RC15 --model=GRU --lr=0.005
python3 Proxy\ Approaches/CAT.py --data=Datasets/RC15 --model=Caser --lr=0.01
python3 Proxy\ Approaches/CAT.py --data=Datasets/RC15 --model=NItNet --lr=0.01
python3 Proxy\ Approaches/CAT.py --data=Datasets/RC15 --model=SASRec --lr=0.01

python3 Proxy\ Approaches/CAT3.py --data=Datasets/RC15 --model=GRU --lr=0.005
python3 Proxy\ Approaches/CAT3.py --data=Datasets/RC15 --model=Caser --lr=0.01
python3 Proxy\ Approaches/CAT3.py --data=Datasets/RC15 --model=NItNet --lr=0.01
python3 Proxy\ Approaches/CAT3.py --data=Datasets/RC15 --model=SASRec --lr=0.01

python3 Proxy\ Approaches/FUT.py --data=Datasets/RC15 --model=GRU --lr=0.005
python3 Proxy\ Approaches/FUT.py --data=Datasets/RC15 --model=Caser --lr=0.01
python3 Proxy\ Approaches/FUT.py --data=Datasets/RC15 --model=NItNet --lr=0.01
python3 Proxy\ Approaches/FUT.py --data=Datasets/RC15 --model=SASRec --lr=0.01

python3 RL\ approaches/EVAL.py --data=Datasets/RC15 --model=GRU --lr=0.005
python3 RL\ approaches/EVAL.py --data=Datasets/RC15 --model=Caser --lr=0.01
python3 RL\ approaches/EVAL.py --data=Datasets/RC15 --model=NItNet --lr=0.01
python3 RL\ approaches/EVAL.py --data=Datasets/RC15 --model=SASRec --lr=0.01

python3 RL\ approaches/SQN.py --data=Datasets/RC15 --model=GRU --lr=0.005
python3 RL\ approaches/SQN.py --data=Datasets/RC15 --model=Caser --lr=0.01
python3 RL\ approaches/SQN.py --data=Datasets/RC15 --model=NItNet --lr=0.01
python3 RL\ approaches/SQN.py --data=Datasets/RC15 --model=SASRec --lr=0.01
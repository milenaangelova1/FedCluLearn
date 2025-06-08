# FedCluLearn
This is the code repository that was developed for the paper "FedCluLearn: Federated Continual Learning using Stream Micro-Cluster Indexing Scheme".

# Requirements
The Python version is Python 3.10.12.
```bash
pip install -r requirements.txt
```

# Run control experiment
```bash
python src/fl_experiment_control.py
```
with the following parameters for the 5G data 
```bash
n_clients=3
fl_round_count=200
lookback = 10  
num_partitions = 4 
training_percentage=0.50 # 0.25 or 0.75

# uncomment the following lines in the code
train_data = pd.read_csv(f"../data/5g_data/train_client_{client_id}.csv")
test_data = pd.read_csv(f"../data/5g_data/test_client_{client_id}.csv")

label = 'rnti_count'
targets = [label]
drop_features = [label]

# select an algorithm. Uncomment one of them in the code.
# different algorithm names below. Choose one of them.
# FedAvg, FedAtt, FedProx, 
# FedCluLearn_Prox, FedCluLearn_Prox_recent, FedCluLearn_Prox_percentage
# FedCluLearn, FedCluLearn_recent, FedCluLearn_percentage
```
The script will generate a results file in the root directory.

# Run real experiment
The same as the control expriment but with different params.
```bash
python src/fl_experiment_real.py
```
with the following parameters for the 5G data 
```bash
n_clients=12
fl_round_count=250
lookback = 9
num_partitions = 5 
training_percentage=0.50 # 0.25 or 0.75

# uncomment the following lines in the code
names = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']
train_data = pd.read_csv(f"../data/air_quality/train_client_{names[i]}.csv")
test_data = pd.read_csv(f"../data/air_quality/test_client_{names[i]}.csv")

label = 'PM2.5'
targets = [label]
drop_features = [label, train_data.columns[9]]

# select an algorithm. Uncomment one of them in the code.
# different algorithm names below. Choose one of them.
# FedAvg, FedAtt, FedProx, 
# FedCluLearn_Prox, FedCluLearn_Prox_recent, FedCluLearn_Prox_percentage
# FedCluLearn, FedCluLearn_recent, FedCluLearn_percentage
```
# EDAs
Original 5G data => https://github.com/vperifan/federated-time-series-forecasting?tab=readme-ov-file \
notebooks/eda/eda_5G.ipynb => 5G data 

Original Air pollution data => https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
notebooks/eda/eda_air_pollution.ipynb => Air pollution data

# Plots
All results can be seen visually in the folder ```notebooks```.

# Cite
If you are using this code please cite this paper: \
M. Angelova, V. Boeva, S. Abghari, S. Ickin and X. Lan. FedCluLearn: Federated Continual Learning using Stream Micro-Cluster Indexing Scheme. ECML PKDD 2025 (Porto, Portugal, 15-19 September 2025)

# Email me
Milena Angelova \
milena.angelova@bth.se \
mangelova@mail.com


from sklearn.preprocessing import RobustScaler
import torch
import pandas as pd
import numpy as np
import copy
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

from utils import update_model

def weighted_avg(global_model_dict, local_models, clients_data, client_ids)-> torch.nn.Module:
    """
    Perform weighted averaging of local models to update the global model.

    Args:
        global_model_dict (dict): Dictionary containing the global model.
        local_models (dict): Dictionary containing local models from clients.
        clients_data (dict): Dictionary clients data.

    Returns:
        torch.nn.Module: The updated global model.
    """
    weights = []
    for client in client_ids:
        weights.append(clients_data[f'client{client}']['size'])

    state_dicts = [model.state_dict() for model in local_models.values()]
    total_sum_weights = sum(weights)
    normalized_weights = [weight / total_sum_weights for weight in weights]

    with torch.no_grad():
        for key in global_model_dict['global_model'].state_dict().keys():
            stacked_params = torch.stack(
                [state_dict[key] * normalized_weights[i] for i, (state_dict) in enumerate(state_dicts)], dim=0
            )

            global_model_dict['global_model'].state_dict()[key].copy_(stacked_params.sum(dim=0))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


def get_weights(g_model):
    list_weights = []
    for i in range(0, len(list(g_model.parameters())),2):
        list_weights.append([list(g_model.parameters())[i].detach().numpy(), list(g_model.parameters())[i+1].detach().numpy()])
    return list_weights

def fed_att(global_model_dict, local_models, epsilon=0.01):
    """
    Implements Attentive Federated Averaging (AttentiveFedAvg).
    
    Args:
        global_model (torch.nn.Module): The global model.
        local_models (dict): Dictionary containing local models {client_id: model}.
        epsilon (float): Step size for global model optimization.
    
    Returns:
        torch.nn.Module: The updated global model.
    """
    global_model = copy.deepcopy(global_model_dict['global_model'])
    global_weights = copy.deepcopy(global_model.state_dict())
    num_clients = len(local_models)

    # Initialize attention coefficients
    alpha = {name: torch.zeros(num_clients) for name in global_weights.keys()}

    # Compute similarity scores and attention weights
    with torch.no_grad():
        for l, layer_name in enumerate(global_weights.keys()):  # Iterate over layers
            s_k = []
            
            # Compute L2 distance between global model and local models
            for k, (client_id, local_model) in enumerate(local_models.items()):
                local_weights = local_model.state_dict()
                s_k.append(torch.norm(global_weights[layer_name] - local_weights[layer_name], p=2))
            
            # Convert similarity scores to tensor
            s_k = torch.tensor(s_k)
            
            # Apply softmax to compute attention weights
            alpha[layer_name] = torch.softmax(s_k, dim=0)

    # Aggregate model updates
    with torch.no_grad():
        for layer_name in global_weights.keys():
            weighted_sum = torch.zeros_like(global_weights[layer_name])

            for k, (client_id, local_model) in enumerate(local_models.items()):
                local_weights = local_model.state_dict()
                weighted_sum += alpha[layer_name][k] * (global_weights[layer_name] - local_weights[layer_name])

            # Update global model
            global_weights[layer_name] -= epsilon * weighted_sum

    # Load updated weights into the global model
    global_model_dict['global_model'].load_state_dict(global_weights)

def local_fedprox_update(model, global_model, data_loader, loss_fn, optimizer, mu=0.1, num_epochs=1):
    """
    Performs local training using FedProx.

    Args:
        model: Local client model.
        global_model: Global model sent from the server.
        data_loader: DataLoader for the client's data.
        loss_fn: Loss function.
        optimizer: Optimizer (e.g., Adam, SGD).
        mu: Proximal term coefficient.
        num_epochs: Number of local training epochs.

    Returns:
        Updated local model weights.
    """
    global_weights = {name: param.clone().detach() for name, param in global_model.state_dict().items()}

    model.train()
    for epoch in range(num_epochs):
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Proximal term: ||w - w_global||^2
            prox_term = sum((torch.norm(param - global_weights[name]) ** 2) for name, param in model.named_parameters())
            loss += (mu / 2) * prox_term

            loss.backward()
            optimizer.step()

    return model.state_dict()

def fed_prox(global_model_dict, clients_data):
    """
    Implements the full FedProx algorithm.

    Args:
        global_model_dict: Dictionary {global_model: global model}
        clients_data: Dictionary {client_id: local_model}.
        frac: Fraction of clients participating per round.
    """
    global_model = copy.deepcopy(global_model_dict['global_model'])
    state_dicts = [model.state_dict() for model in clients_data.values()]

    # Aggregate local updates using mean (FedAvg-style)
    new_global_weights = copy.deepcopy(global_model.state_dict())
    for name in new_global_weights.keys():
        new_global_weights[name] = torch.mean(torch.stack([weights[name] for weights in state_dicts]), dim=0)

    global_model_dict['global_model'].load_state_dict(new_global_weights)


def test_the_client_model(model, clients_data, client_id, label):
    model.eval()
    df_tmp_test = clients_data[f'client{client_id}']['test']
    # target_scaler = clients_data[f'client{client_id}']['target_scaler']
    df_tmp_test.reset_index(inplace=True, drop=True) 

    # Fit the scaler on training data
    targets = [label]
    drop_features = [label]
    
    y_test = torch.tensor(df_tmp_test[targets].values.reshape(-1,1)).float()
    X_test = torch.tensor(df_tmp_test.drop(drop_features, axis=1).values).float()

    
    # Make predictions
    with torch.no_grad():
        y_pred = model(X_test)
        # y_pred = target_scaler.inverse_transform(y_pred.detach().numpy())
        # y_test = target_scaler.inverse_transform(y_test.detach().numpy())

        y_pred = y_pred.detach().numpy() * clients_data[f'client{client_id}']['y_iqr'] + clients_data[f'client{client_id}']['y_medium']
        y_test = y_test.detach().numpy() * clients_data[f'client{client_id}']['y_iqr'] + clients_data[f'client{client_id}']['y_medium']

    mse, mae, r2 = mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)
    return mse / np.var(y_test), mae / np.mean(y_test), r2

def test_the_global_model(model, test_dict, clients_data, label):
    model.eval()

    mse_list, mae_list, r2_list = [], [], []
    for client_id, test_df in test_dict.items():
        targets = [label]
        drop_features = [label]
        # target_scaler = clients_data[f'client{client_id}']['target_scaler']
        
        # Fit the scaler on training data
        y_test = torch.tensor(test_df[targets].values.reshape(-1,1)).float()
        X_test = torch.tensor(test_df.drop(columns=drop_features, axis=1).values).float()
        
        # Make predictions
        with torch.no_grad():
            y_pred = model(X_test)
            # y_pred = target_scaler.inverse_transform(y_pred.detach().numpy())
            # y_test = target_scaler.inverse_transform(y_test.detach().numpy())
            y_pred = y_pred.detach().numpy() * clients_data[f'client{client_id}']['y_iqr'] + clients_data[f'client{client_id}']['y_medium']
            y_test = y_test.detach().numpy() * clients_data[f'client{client_id}']['y_iqr'] + clients_data[f'client{client_id}']['y_medium']

        mse, mae, r2 = mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)
        mse_list.append(mse/np.var(y_test))
        mae_list.append(mae/np.mean(y_test))
        r2_list.append(r2)

    return mse_list, mae_list, r2_list
    
def write_to_file(mse, mae, r2, best_mse, fl_round, client_id, direction, type: str, outputs:list):
    result_line = [fl_round,client_id,learning_rate,direction,mse,best_mse,mae,r2,lookback,num_epochs,hidden_size,batch_size]
    fl_results.append(result_line)
    print(f"{type} validation: Round: {fl_round}, Client: {client_id}, LR: {learning_rate}, Direction: {direction}, MSE: {mse:.4f}, Best MSE: {best_mse}, MAE: {mae}, R2: {r2}, Num. neurons: {lookback}, Num. epochs: {num_epochs}, Hidden size: {hidden_size}, Batch size: {batch_size}")
    # model_dict[str(client_id)] = copy.deepcopy(model)
    _writeToFile(";".join(map(str,result_line)), outputs)

n_clients=3 # 5G 3 # Air quality 12

# Params 5G and Air quality data
batch_size = 128
hidden_size = 128
second_size = 64
output_size = 1
learning_rate = 0.0001
num_epochs = 3
fl_round_count=200 # 5G 200 # Air quality 250
lookback = 10  # 5G 10 # Air quality 9
epsilon=0.1
mu=0.1  # 0.001
num_partitions = 4  # 5G 4 # Air quality 5
training_percentage=0.75
SI_threshold = 0.50

first_shift = 50
second_shift = 100
third_shift = 150
forth_shift = 200
fifth_shift = 250
sixth_shift = 300

time = datetime.now()
# different algorithm names below. Choose one of them.
# FedAvg, FedAtt, FedProx, 
# FedCluLearn_Prox, FedCluLearn_Prox_recent, FedCluLearn_Prox_percentage
# FedCluLearn, FedCluLearn_recent, FedCluLearn_percentage
algorithm_name = 'FedCluLearn_Prox_percentage'
expname=f"{algorithm_name}_{time}"

def _writeToFile(line, outputs: list):
    for output_name in outputs:
        with open(f'{output_name}_{expname}.txt', 'a') as file:
            #for line in lines_to_append:
            file.write(line + '\n')

class TimeSeriesDatasetNN(Dataset):
    def __init__(self, data, label):
        self.X, self.y = self.preprocess(data, label)
    
    def preprocess(self, data, label):
        targets = [label]
        drop_features = [label]
        
        train_y = torch.tensor(data[targets].values.reshape(-1, 1)).float()
        train_x = torch.tensor(data.drop(columns=drop_features, axis=1).values).float()
       
        return train_x, train_y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

class NLayerNet(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(NLayerNet, self).__init__()
        # Define the first fully connected layer
        self.fc1 = nn.Linear(lookback, hidden_size)
        # Define the second fully connected layer
        self.fc2 = nn.Linear(hidden_size, second_size)
        # Define the third hidden layer
        self.fc3 = nn.Linear(second_size, output_size)

    def forward(self, x):
        # Apply the first layer and ReLU activation
        x = F.relu(self.fc1(x))
        # Apply the second layer
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
       
        return x
        
def real_experiment(fl_round, client_id, clients_data, shift_points, n_clients, num_epochs):
    """
    Assigns training and test data based on federated learning round and client ID.

    Args:
    - fl_round: Current federated learning round.
    - client_id: ID of the client.
    - clients_data: Dictionary containing train and test data for each client.
    - shift_points: List containing shift rounds (e.g., [first_shift, second_shift, third_shift, forth_shift]).
    - n_clients: Number of clients.

    Returns:
    - df_tmp_train: Training data for the client.
    - client_test_id: ID of the test client.
    - direction: Concept name assigned to the client.
    - test_data: Dictionary with test data per client.
    - test_data_global: Global test data dictionary.
    - test_global_model_concepts: List of concepts per client.
    - epochs: Number of epochs (default: 3).
    """

    # Determine which shift we are in
    shift_index = next((i for i, shift in enumerate(shift_points) if fl_round < shift), None)
    if shift_index is None:
        return None, None, None, None, None, None, 0  # Return empty if no valid shift

    # Generate dynamic concept names
    concepts = {i: f'concept{i+1}' for i in range(n_clients)}

    # Assign training data based on client_id and shift_index
    df_tmp_train = clients_data[f'client{client_id}.{shift_index}']['train'].copy()
    client_test_id = f'{client_id}.{shift_index}'
    direction = concepts[client_id]
    epochs = num_epochs

    # Assign test data for all clients dynamically
    test_data = {f'{i}.{shift_index}': clients_data[f'client{i}.{shift_index}']['test'].copy() for i in range(n_clients)}
    test_data_global = test_data.copy()  # Copy test data to global dictionary
    test_global_model_concepts = list(concepts.values())  # List of all concepts

    return df_tmp_train, client_test_id, direction, test_data, test_data_global, test_global_model_concepts, epochs

clients_data = {}
test_data_list = []
for i in range(n_clients):
    
    # 5G
    if i == 0:
        client_id = 'PobleSec'
    if i == 1:
        client_id = 'LesCorts'
    if i == 2:
        client_id = 'ElBorn'

    base_path = os.path.dirname('__file__')
    # 5G Data original
    train_data = pd.read_csv(os.path.join(base_path, 'data', '5g_data', f'train_client_{client_id}.csv'))
    test_data = pd.read_csv(os.path.join(base_path, 'data', '5g_data', f'test_client_{client_id}.csv'))

    # Air quality data
    # names = ['Aotizhongxin', 'Changping', 'Dingling', 'Dongsi', 'Guanyuan', 'Gucheng', 'Huairou', 'Nongzhanguan', 'Shunyi', 'Tiantan', 'Wanliu', 'Wanshouxigong']
    # train_data = pd.read_csv(os.path.join(base_path, 'data', 'air_quality', f'train_client_{names[i]}.csv'))
    # test_data = pd.read_csv(os.path.join(base_path, 'data', 'air_quality', f'test_client_{names[i]}.csv'))

    # 5G data
    label = 'rnti_count'
    targets = [label]
    drop_features = [label]
    
    # Air quality
    # label = 'PM2.5'
    # targets = [label]
    # drop_features = [label, train_data.columns[9]]

    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()

    x_train = feature_scaler.fit_transform(train_data.drop(drop_features, axis=1).values)
    y_train = target_scaler.fit_transform(train_data[targets].values)

    x_test = feature_scaler.transform(test_data.drop(drop_features, axis=1).values)
    y_test = target_scaler.transform(test_data[targets].values)

    train_data = pd.DataFrame(x_train)
    train_data[label] = y_train

    test_data = pd.DataFrame(x_test)
    test_data[label] = y_test

    client_train_list = np.array_split(train_data, num_partitions)
    client_test_list = np.array_split(test_data, num_partitions)
    
    for j, (train, test) in enumerate(zip(client_train_list, client_test_list)):
        clients_data[f'client{i}.{j}'] = {
            'train': train, 
            'test': test,
            'y_iqr': target_scaler.scale_,
            'y_medium': target_scaler.center_,
            'size': train.shape[0]
        }

results=[]
global_model_dict={}

all_rounds_global_model={}
prev_direction=""
fl_results=[]
fl_rounds_statistics = {}
test_data_global = {}
test_global_model_concepts = {}

for fl_round in range(fl_round_count): 
    model_dict={}
    client_ids = []
    for client_id in range(n_clients):
        df_tmp_train, client_test_id, direction, test_data, test_data_global, test_global_model_concepts, epochs = real_experiment(fl_round, client_id, clients_data, [first_shift, second_shift, third_shift, forth_shift], n_clients, num_epochs)
        
        df_tmp_train.reset_index(inplace=True,drop=True)
        
        dataset = TimeSeriesDatasetNN(df_tmp_train, label)
        data_loader = DataLoader(dataset, batch_size=batch_size)

        criterion = nn.MSELoss()

        if(fl_round==0):
            init_model = NLayerNet(hidden_size, output_size)
            model = copy.deepcopy(init_model)
            global_model_dict['global_model'] = model
        else:
            model = copy.deepcopy(global_model_dict["global_model"])

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # weight_decay=1e-5

        # Train the model
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
    
            for x, y in data_loader:
                output = model(x)
                loss = criterion(output, y)
                optimizer.zero_grad()
                if algorithm_name in ['FedProx', 'FedCluLearn_Prox', 'FedCluLearn_Prox_recent', 'FedCluLearn_Prox_percentage']:
                    # Proximal term: ||w - w_global||^2
                    global_weights = {name: param.clone().detach() for name, param in global_model_dict['global_model'].state_dict().items()}
                    prox_term = sum((torch.norm(param - global_weights[name]) ** 2) for name, param in model.named_parameters())
                    loss += (mu / 2) * prox_term
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            epoch_loss /= len(data_loader)
            
            print(f"After local training: Round: {fl_round}, Client: {client_test_id}, LR: {optimizer.param_groups[0]['lr']}, Epoch: {epoch + 1}, MSE: {epoch_loss:.4f}")

            mse, mae, r2 = test_the_client_model(model, clients_data, client_id=client_test_id, label=label)

        model_dict[str(client_id)] = copy.deepcopy(model)
                
        write_to_file(mse, mae, r2, None, fl_round, int(float(client_test_id)), direction, type='local', outputs=['results', 'clusters'])
        client_ids.append(client_test_id)

    if algorithm_name == 'FedAvg':
        weighted_avg(global_model_dict, model_dict, clients_data, client_ids)
    elif algorithm_name == 'FedAtt':
        fed_att(global_model_dict, model_dict, epsilon)
    elif algorithm_name == 'FedProx':
        fed_prox(global_model_dict, model_dict)
    else:
        with torch.no_grad():
            update_model(model_dict, fl_rounds_statistics, global_model_dict, fl_round, expname, n_clients, algorithm_name, training_percentage, SI_threshold)

    # Test the global model - all test data from each client
    mse_list, mae_list, r2_list = test_the_global_model(global_model_dict["global_model"], test_dict=test_data_global, clients_data=clients_data, label=label)
    for mse, mae, r2, client_id, concept in zip(mse_list, mae_list, r2_list, range(n_clients), test_global_model_concepts):
        write_to_file(mse, mae, r2, None, fl_round, client_id, f'{concept}', type='global', outputs=['global_model_evaluation'])

print(f"local_{algorithm_name} = 'results_{expname}.txt'")
print(f"global_{algorithm_name} = 'global_model_evaluation_{expname}.txt'")
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
import pandas as pd
import numpy as np
import torch
import copy
import math
from constants import FL_ROUND_COUNT

def write_to_file(result_line, filenames=['results', 'clusters', 'output'], print_statement=True):
    '''
    Write results into a text file.

    :param: result_line: list
        result_line = [fl_round, client_id, learning_rate, direction, mse, mae,r2, num_neurons, hidden_size, batch_size]
    '''
    line = ";".join(map(str,result_line))
    if print_statement:
        line = result_line
    
    for filename in filenames:
        with open(f'{filename}.txt', 'a') as file:
            #for line in lines_to_append:
            file.write(line + '\n')

def get_client_models_params(client_models):
    if not bool(client_models):
        raise Exception("Client models are empty.")
    
    clients_params = {}
    for client_id, client_model in client_models.items():
        clients_params[client_id] = np.concatenate([p.detach().flatten().cpu().numpy() for p in client_model.parameters()])
    return pd.DataFrame.from_dict(clients_params, orient='index')

def kmeans_clustering(client_models, expname):
    # Perform k-means clustering on global model parameters
    clients_params_df = get_client_models_params(client_models)
    silhouette_scores = []
  
    for n_clusters in range(2, min(10, clients_params_df.shape[0])):  # Limit to maximum 10 clusters
        kmeans = KMeans(n_clusters=n_clusters).fit(clients_params_df)
        if len(set(kmeans.labels_)) > 1:
            silhouette_score_val = silhouette_score(clients_params_df, kmeans.labels_)
            silhouette_scores.append(silhouette_score_val)
    if bool(silhouette_scores):
        optimal_n_clusters = np.argmax(silhouette_scores) + 2  # Adding 2 as we started from 2 clusters
        print(f"Optimal number of clusters: {optimal_n_clusters}")
        with open(f"clusters_{expname}.txt", "a") as f:
            f.write(f"Optimal number of clusters: {optimal_n_clusters}\n")
        kmeans = KMeans(n_clusters=optimal_n_clusters).fit(clients_params_df)
        clients_params_df['cluster'] = kmeans.labels_
    else:
        clients_params_df['cluster'] = clients_params_df.shape[0] * [0]
    return clients_params_df

def chebyshev_bounderies(k, LS, SS, n_clients):
    # Calculate the mean and standard deviation of the distances
    mean = LS / n_clients
    std = np.sqrt(SS / n_clients)

    # Calculate the range within k standard deviations
    upper_boundary = mean + (k * std)
    lower_boundary = mean - (k * std)

    return upper_boundary, lower_boundary

def calculate_cluster_statistics(data, cluster_labels, expname, n_clients):
    # Calculate statistics for each cluster
    cluster_statistics = []
    for cluster_label in set(cluster_labels):
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        cluster_params = data.iloc[cluster_indices].to_numpy()

        linear_sum = np.sum(cluster_params, axis=0)
        squared_sum = np.sum((cluster_params) ** 2, axis=0)
        n_clients_in_cluster = len(cluster_indices)
        client_frequency = np.array([list(cluster_indices).count(i) for i in range(n_clients)])
        print(f'Cluster: {cluster_label}: {client_frequency}')
        with open(f"clusters_{expname}.txt", "a") as f:
            f.write(f'Cluster: {cluster_label}: {client_frequency}')

        cluster_statistics.append({
            'linear_sum': linear_sum,
            'squared_sum': squared_sum,
            'n_clients': n_clients_in_cluster,
            'client_frequency': client_frequency,
            'cluster_label': cluster_label,
            'active': True
        })

    return cluster_statistics

def load_new_params(client_model, new_params):
    # Load the new parameters into the model's state_dict
    model_state_dict = client_model.state_dict()
    model_state_dict.update(new_params)
    client_model.load_state_dict(model_state_dict)

def calculate_totals(fl_rounds_statistics: dict) -> None:
    for _, cluster in fl_rounds_statistics.items():
        rounds = list(cluster.keys())
        total_round = copy.deepcopy(cluster[rounds[0]])
        
        for index in range(1, len(rounds)):
            fl_round = rounds[index]
            if fl_round == 'total':
                continue
            if bool(cluster[fl_round]):
                total_round['n_clients'] += cluster[fl_round]['n_clients']
                total_round['linear_sum'] += cluster[fl_round]['linear_sum']
                total_round['squared_sum'] += cluster[fl_round]['squared_sum']
                total_round['client_frequency'] += cluster[fl_round]['client_frequency']
                total_round['cluster_label'] = cluster[fl_round]['cluster_label']
        cluster['total'] = total_round

def calculate_totals_for_current_cluster(cluster) -> None:
    rounds = list(cluster.keys())
    total_round = copy.deepcopy(cluster[rounds[0]])
    
    for index in range(1, len(rounds)):
        fl_round = rounds[index]
        if fl_round == 'total':
            continue
        if bool(cluster[fl_round]):
            total_round['n_clients'] += cluster[fl_round]['n_clients']
            total_round['linear_sum'] += cluster[fl_round]['linear_sum']
            total_round['squared_sum'] += cluster[fl_round]['squared_sum']
            total_round['client_frequency'] += cluster[fl_round]['client_frequency']
            total_round['cluster_label'] = cluster[fl_round]['cluster_label']
    cluster['total'] = total_round
    
def build_current_global_model_eq_3(fl_rounds_statistics):
    overall_model = 0
    total_n_clients = 0
    for _, cluster  in fl_rounds_statistics.items():
        if cluster['total']['active']:
            overall_model += cluster['total']['linear_sum']
            total_n_clients += cluster['total']['n_clients']
    return overall_model / total_n_clients

def build_current_global_model_eq_3_recent_cf(fl_rounds_statistics, fl_round):
    overall_model = 0
    total_n_clients = 0
    for _, cluster  in fl_rounds_statistics.items():
        if cluster['total']['active']:
            overall_model += cluster[fl_round]['linear_sum']
            total_n_clients += cluster[fl_round]['n_clients']
    return overall_model / total_n_clients

def build_current_global_model_eq_3_recent_cf_percentage(fl_rounds_statistics, fl_round, training_percentage=0.50):
    overall_model = 0
    total_n_clients = 0
    n_percentage_rounds = math.ceil(fl_round * training_percentage)

    for _, cluster  in fl_rounds_statistics.items():
        if cluster['total']['active']:
            rounds = sorted(list(filter(lambda n: isinstance(n, int), list(cluster.keys()))), reverse=True)[0:n_percentage_rounds+1]
            for r_round in rounds:
                overall_model += cluster[r_round]['linear_sum']
                total_n_clients += cluster[r_round]['n_clients']
    return overall_model / total_n_clients

def generate_state_dict(aggregated_model: list, state_dict_schema):
    global_state_dict = copy.deepcopy(state_dict_schema)
    latest_i = 0
    for key, item in state_dict_schema.items():
        list_number = len(item)
        data = []
        if isinstance(item.tolist()[-1], float):
            list_number_inside = len(item)
            data = aggregated_model[latest_i: latest_i + list_number_inside]
            latest_i += list_number_inside
        else:
            list_number_inside = len(item[-1])
            for i in range(latest_i, latest_i + (list_number * list_number_inside), list_number_inside):
                data.append(aggregated_model[i: i + list_number_inside])
            latest_i += (list_number * list_number_inside)

        global_state_dict[key] = torch.as_tensor(data)
    return global_state_dict

def is_within_chebyshev_threshold(linear_sum, squared_sum, n_clients, client, chebyshev_k):
    upper_boundary, lower_boundary = chebyshev_bounderies(chebyshev_k, linear_sum, squared_sum, n_clients)
    # Determine similarity based on Euclidean distance
    if np.all(client >= lower_boundary) and np.all(client <= upper_boundary):
        return True
    return False

def silhouette_index(vector, cluster, other_clusters):
    # Calculate the average distance (a) to the same cluster
    a = np.mean(pairwise_distances([vector], cluster)[0])

    # Calculate the average distance (b) to the next closest cluster
    b = float('inf')
    for other_cluster in other_clusters:
        distance_to_other_cluster = np.mean(pairwise_distances([vector], other_cluster)[0])
        b = min(b, distance_to_other_cluster)

    # Calculate the Silhouette Score
    silhouette_score = (b - a) / max(a, b)
    return silhouette_score

def clusters_means(clusters):
    temp_clusters = []
    for cluster in clusters:
        temp_clusters.append(cluster['linear_sum']/cluster['n_clients'])
    return temp_clusters

def closest_cluster_by_silhouette(vector, clusters, labels, k=1):
    unique_labels = np.unique(labels)
    max_silhouette_score = -1
    best_cluster = None

    si_list = []
    for label in unique_labels:
        same_cluster = clusters[labels == label]
        other_clusters = [clusters[labels == other_label] for other_label in unique_labels if other_label != label]

        si = silhouette_index(vector, same_cluster, other_clusters)
        si_list.append(si)
        if si > max_silhouette_score:
            max_silhouette_score = si
            best_cluster = label

    return best_cluster, max_silhouette_score

def update_fl_statistics(new_client_mean, fl_rounds_statistics, fl_round, cluster_index, client_id, n_clients):
    frequency = np.zeros(n_clients, dtype=int)
    frequency[int(float(client_id))] = 1
    
    feature_vector = {
        'linear_sum': new_client_mean,
        'squared_sum': new_client_mean ** 2,
        'n_clients': 1,
        'client_frequency': frequency,
        'cluster_label':  cluster_index
    }
    if fl_round not in fl_rounds_statistics[cluster_index]:
        fl_rounds_statistics[cluster_index][fl_round] = feature_vector
    else:
        fl_rounds_statistics[cluster_index][fl_round]['n_clients'] += feature_vector['n_clients']
        fl_rounds_statistics[cluster_index][fl_round]['linear_sum'] += feature_vector['linear_sum']
        fl_rounds_statistics[cluster_index][fl_round]['squared_sum'] += feature_vector['squared_sum']
        fl_rounds_statistics[cluster_index][fl_round]['client_frequency'] += feature_vector['client_frequency']
        fl_rounds_statistics[cluster_index][fl_round]['cluster_label'] = feature_vector['cluster_label']
    calculate_totals_for_current_cluster(fl_rounds_statistics[cluster_index])

def update_fl_statistics_with_new_concept(client_model, fl_rounds_statistics, fl_round, client_id, expname, active_clusters, n_clients):
    frequency = np.zeros(n_clients, dtype=int)
    frequency[int(float(client_id))] = 1
    cluster_label = len(fl_rounds_statistics)
    active_clusters.add(cluster_label)
    feature_vector = {
        'linear_sum': client_model,
        'squared_sum': client_model ** 2,
        'n_clients': 1,
        'client_frequency': frequency,
        'cluster_label':  cluster_label,
        'active': True
    }
    print("Client: ", client_id, "Cluster: ", feature_vector['cluster_label'])
    fl_rounds_statistics[cluster_label] = {fl_round: feature_vector, 'total': feature_vector}
    with open(f"clusters_{expname}.txt", "a") as f:
        f.write(f"Client {client_id}, Cluster {feature_vector['cluster_label']}, N_clients Cluster {feature_vector['n_clients']}\n")

def add_new_empty_round(fl_round_statistics, fl_round):
    for key in fl_round_statistics.keys():
        fl_round_statistics[key][fl_round] = {}
        
def create_fl_round_statistics(fl_rounds_statistics, cluster_statistics, fl_round):
    for cluster in cluster_statistics:
        fl_rounds_statistics[cluster['cluster_label']] = {fl_round: cluster, 'total': cluster}
    
def get_clusters(fl_rounds_statistics):
    clusters_dict = {}
    for _, cluster in fl_rounds_statistics.items():
        clusters_dict[cluster['total']['cluster_label']] = cluster['total']
    means = clusters_means([v for _, v in clusters_dict.items()])
    return clusters_dict, np.array(means)

def build_global_model(global_model_dict):
    g_model = np.zeros(len(global_model_dict[0]))
    round_count = 0
    for k, v in global_model_dict.items():
        if k == 'global_model':
            continue
        g_model += v
        round_count += 1
    return g_model / round_count

def update_active_clusters(fl_rounds_statistics, active_clusters):
    for cluster_index in fl_rounds_statistics.keys():
        if cluster_index in active_clusters:
            fl_rounds_statistics[cluster_index]['total']['active'] = True
        else:
            fl_rounds_statistics[cluster_index]['total']['active'] = False

def calculate_metrics(clusters: dict, expname: str):
    # clusters = {
    #     "cluster_1": {"n": 100, "linear_sum": np.array([300, 400]), "squared_sum": np.array([10000, 16000])},
    #     "cluster_2": {"n": 150, "linear_sum": np.array([500, 600]), "squared_sum": np.array([20000, 36000])},
    #     "cluster_3": {"n": 120, "linear_sum": np.array([450, 550]), "squared_sum": np.array([18000, 29000])},
    # }

    # Initialize variables
    centroids = {}
    variances = {}
    compactnesses = {}

    # Calculate centroids, variances, and compactness
    for cluster_name, cluster_data in clusters.items():
        n = cluster_data["n_clients"]
        linear_sum = cluster_data["linear_sum"]
        squared_sum = cluster_data["squared_sum"]
        
        # Centroid
        centroid = linear_sum / n
        centroids[cluster_name] = centroid
        
        # Variance
        variance = (squared_sum / n) - (centroid ** 2)
        variances[cluster_name] = variance
        
        # Compactness
        compactness = np.sum(variance)
        compactnesses[cluster_name] = compactness

    # Calculate inter-cluster distances
    cluster_names = list(clusters.keys())
    distances = {}
    for i in range(len(cluster_names)):
        for j in range(i + 1, len(cluster_names)):
            c1, c2 = cluster_names[i], cluster_names[j]
            distance = np.linalg.norm(centroids[c1] - centroids[c2])
            distances[(c1, c2)] = distance

    # Calculate Dunn Index
    min_inter_cluster_distance = min(distances.values())
    max_intra_cluster_compactness = max(np.sqrt(compactnesses[cluster_name]) for cluster_name in cluster_names)
    dunn_index = min_inter_cluster_distance / max_intra_cluster_compactness

    # Calculate Davies-Bouldin Index
    db_index = 0
    for i in range(len(cluster_names)):
        c1 = cluster_names[i]
        max_ratio = 0
        for j in range(len(cluster_names)):
            if i != j:
                c2 = cluster_names[j]
                # Ensure distances are accessed in the correct order
                distance_key = (c1, c2) if (c1, c2) in distances else (c2, c1)
                compactness_sum = np.sqrt(compactnesses[c1]) + np.sqrt(compactnesses[c2])
                ratio = compactness_sum / distances[distance_key] if distances[distance_key] != 0 else np.inf
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    db_index /= len(cluster_names)

    # Calculate Silhouette Scores for each cluster
    silhouette_scores = {}
    for cluster_name in cluster_names:
        # Average intra-cluster distance (compactness approximation)
        a = np.sqrt(compactnesses[cluster_name])
        
        # Average inter-cluster distance (minimum distance to another centroid)
        b = min(distances[(cluster_name, other)] if (cluster_name, other) in distances else distances[(other, cluster_name)]
                for other in cluster_names if other != cluster_name)
        
        # Silhouette score
        s = (b - a) / max(a, b) if max(a, b) != 0 else 0
        silhouette_scores[cluster_name] = s

    # Calculate overall Silhouette Index
    total_instances = sum(clusters[cluster_name]["n_clients"] for cluster_name in cluster_names)
    overall_silhouette_index = sum(
        clusters[cluster_name]["n_clients"] * silhouette_scores[cluster_name] for cluster_name in cluster_names
    ) / total_instances

    # Print results
    print("Centroids:", centroids)
    print("Variances:", variances)
    print("Compactness:", compactnesses)
    print("Inter-cluster Distances:", distances)
    print("Dunn Index:", dunn_index)
    print("Davies-Bouldin Index:", db_index)
    print("Silhouette Scores (Approximation):", silhouette_scores)
    print("Overall Silhouette Index:", overall_silhouette_index)
    with open(f"clusters_{expname}.txt", "a") as f:
        f.write(f"Centroids: {centroids}, Variances: {variances}, Compactness: {compactnesses}, Inter-cluster Distances: {distances}, Dunn Index: {dunn_index}, Davies-Bouldin Index: {db_index}, Silhouette Scores (Approximation): {silhouette_scores}, Overall Silhouette Index: {overall_silhouette_index}\n")

def get_current_global_model(algorithm_name, fl_rounds_statistics, fl_round, training_percentage):
    if algorithm_name in ['FedCluLearn', 'FedCluLearn_Prox']:
        current_global_model = build_current_global_model_eq_3(fl_rounds_statistics)
    elif algorithm_name in ['FedCluLearn_recent', 'FedCluLearn_Prox_recent']:
        current_global_model = build_current_global_model_eq_3_recent_cf(fl_rounds_statistics, fl_round)
    elif algorithm_name in ['FedCluLearn_percentage', 'FedCluLearn_Prox_percentage']:
        current_global_model = build_current_global_model_eq_3_recent_cf_percentage(fl_rounds_statistics, fl_round, training_percentage)
    return current_global_model

def update_model(client_models, fl_rounds_statistics, global_model_dict, fl_round, expname, n_clients, algorithm_name, training_percentage, SI_threshold):
    max_si_scores = []
    if fl_round == 0:
        # Perform k-means clustering after the first training round
        data = kmeans_clustering(client_models, expname)
        labels = data['cluster'].values
        # Calculate statistics for each cluster
        cluster_statistics = calculate_cluster_statistics(data.drop(['cluster'], axis=1), labels, expname, n_clients)
        
        # Represent clusters' statistics in a list of pairs per round
        create_fl_round_statistics(fl_rounds_statistics, cluster_statistics, fl_round)

        clusters, means = get_clusters(fl_rounds_statistics)
        calculate_metrics(clusters, expname)
        # build the clusters' global models
        current_global_model = get_current_global_model(algorithm_name, fl_rounds_statistics, fl_round, training_percentage)
        current_state_dict = client_models[list(client_models.keys())[-1]].state_dict()
        overall_global_model = generate_state_dict(list(current_global_model), current_state_dict)
        # update the global model after the first training round
        global_model_dict["global_model"].load_state_dict(overall_global_model)
    else:
        clusters, means = get_clusters(fl_rounds_statistics)
        active_clusters = set()
        for client_id, client_model in client_models.items():
            clusters, means = get_clusters(fl_rounds_statistics)
            client_model_params = np.concatenate([p.detach().flatten().cpu().numpy() for p in client_model.parameters()])

            # The next lines are about Silhouette index
            best_cluster_index, max_silhouette_score = closest_cluster_by_silhouette(vector=client_model_params, clusters=means, labels=list(clusters.keys()))
            print("Max SI score: ", max_silhouette_score)
            print("Best cluster index: ", best_cluster_index)
            max_si_scores.append(max_silhouette_score)
            if max_silhouette_score >= SI_threshold:
               update_fl_statistics(client_model_params, fl_rounds_statistics, fl_round, best_cluster_index, client_id, len(client_models))
               active_clusters.add(best_cluster_index)
               print(f"Client {client_id}, Cluster {clusters[best_cluster_index]['cluster_label']}, N clients {clusters[best_cluster_index]['n_clients']} SI {max_silhouette_score} SS {clusters[best_cluster_index]['squared_sum']} LS {clusters[best_cluster_index]['linear_sum']} Mean {clusters[best_cluster_index]['linear_sum']/clusters[best_cluster_index]['n_clients']}\n")
               with open(f"clusters_{expname}.txt", "a") as f:
                   f.write(f"Client {client_id}, Cluster {clusters[best_cluster_index]['cluster_label']}, N clients {clusters[best_cluster_index]['n_clients']} SI {max_silhouette_score} SS {clusters[best_cluster_index]['squared_sum']} LS {clusters[best_cluster_index]['linear_sum']} Mean {clusters[best_cluster_index]['linear_sum']/clusters[best_cluster_index]['n_clients']}\n")
            else:
               update_fl_statistics_with_new_concept(client_model_params, fl_rounds_statistics, fl_round, client_id, expname, active_clusters, len(client_models))
               # if there is a new cluster => update the clusters list
               clusters, means = get_clusters(fl_rounds_statistics)

        clusters, means = get_clusters(fl_rounds_statistics)
        calculate_metrics(clusters, expname)
        # update clusters - active/inactive
        update_active_clusters(fl_rounds_statistics, active_clusters)
        # build the clusters' global models
        current_global_model = get_current_global_model(algorithm_name, fl_rounds_statistics, fl_round, training_percentage)
        current_state_dict = client_models[list(client_models.keys())[-1]].state_dict()
        overall_global_model = generate_state_dict(list(current_global_model), current_state_dict)
        
        # update the global model after the first training round
        global_model_dict["global_model"].load_state_dict(overall_global_model)
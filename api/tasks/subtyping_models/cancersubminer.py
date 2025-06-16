import os
import json
import warnings
import zipfile
import subprocess

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

from config import settings
from util.path_untils import source_sample_file_path, target_sample_file_path, job_preprocessing_path, job_result_path, \
    metadata_sample_path

def preprocess_cancersubminer(
        user_id: int,
        project_id: int,
        job_id: int,
        source_file: str,
        target_file: str
):
    source_path = source_sample_file_path(user_id, project_id, source_file)
    target_path = target_sample_file_path(user_id, project_id, target_file)
    preprocessing_path = job_preprocessing_path(user_id, project_id, job_id)
    os.makedirs(preprocessing_path, exist_ok = True)

    missing_value_ratio = 0.2

    source_data = pd.read_csv(source_path, index_col=0)
    target_data = pd.read_csv(target_path, index_col=0)

    if 'subtype' in target_data.columns.tolist() :
        pd.DataFrame(target_data['subtype']).to_csv("target_subtype_info.csv", mode = "w", index = True)
        del target_data['subtype']

    # Remove source samples without subtype label
    source_data = source_data.loc[source_data['subtype'].dropna().index]
    source_data_y = source_data[['subtype']]
    del source_data['subtype']

    # Remove source samples having NAs for all features
    source_data.dropna(how='all', axis=0, inplace=True)

    # Remove features having NAs for all source samples
    # Count the number of NAs for each feature and remove CpGs having missing values more than 20% of samples
    source_data = source_data.T
    num_source_samples = len(source_data.columns)
    cpg_missing_count_df = pd.DataFrame({'count': source_data.isnull().sum(1)})
    cpg_missing_count_df['percentage'] = cpg_missing_count_df['count'] / num_source_samples
    filtered_cpg_list = cpg_missing_count_df[cpg_missing_count_df['percentage'] <= missing_value_ratio].index.tolist()

    source_feature = pd.DataFrame({'cpg': filtered_cpg_list})
    source_data = source_data.T
    source_data = source_data[filtered_cpg_list]
    source_data_y = source_data_y.loc[source_data.index]

    # Remove target samples without batch label
    target_data = target_data.loc[target_data['Batch'].dropna().index]

    target_data_batch_info = target_data[['Batch']]
    del target_data['Batch']

    # Remove target samples having NAs for all features
    # Count the number of NAs for each feature and remove CpGs having missing values more than 20% of samples
    target_data.dropna(how='all', axis=0, inplace=True)

    target_data = target_data.T
    num_target_samples = len(target_data.columns)
    cpg_missing_count_df = pd.DataFrame({'count': target_data.isnull().sum(1)})
    cpg_missing_count_df['percentage'] = cpg_missing_count_df['count'] / num_target_samples
    filtered_cpg_list = cpg_missing_count_df[cpg_missing_count_df['percentage'] <= missing_value_ratio].index.tolist()

    target_data = target_data.T
    target_data = target_data[filtered_cpg_list]

    # Find common features in both source and target
    target_feature = pd.DataFrame({'cpg': filtered_cpg_list})
    common_feature = pd.merge(source_feature, target_feature)['cpg'].tolist()

    source_data = source_data[common_feature]
    target_data = target_data[common_feature]

    # Perform median imputation
    subtype_list = source_data_y['subtype'].unique().tolist()
    res_impute_df = pd.DataFrame()

    for subtype in subtype_list:
        tmp_sample_list = source_data_y[source_data_y['subtype'] == subtype].index.tolist()
        tmp_df = source_data.loc[tmp_sample_list]
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        imputed_values = imp_median.fit_transform(tmp_df)
        imputed_df = pd.DataFrame(imputed_values)
        imputed_df.index = tmp_df.index
        imputed_df.columns = tmp_df.columns
        res_impute_df = pd.concat([res_impute_df, imputed_df], axis=0)

    source_data = res_impute_df.copy()

    res_target_impute_df = pd.DataFrame()
    batch_list = target_data_batch_info['Batch'].unique().tolist()
    for batch in batch_list:
        tmp_sample_list = target_data_batch_info[target_data_batch_info['Batch'] == batch].index.tolist()
        tmp_df = target_data.loc[tmp_sample_list]
        tmp_df.dropna(how='any', axis=1, inplace=True)
        imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
        imputed_values = imp_median.fit_transform(tmp_df)
        imputed_df = pd.DataFrame(imputed_values)
        imputed_df.index = tmp_df.index
        imputed_df.columns = tmp_df.columns
        res_target_impute_df = pd.concat([res_target_impute_df, imputed_df], axis=0)

    res_target_impute_df.dropna(how='any', axis=1, inplace=True)
    '''
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    imputed_values = imp_median.fit_transform(target_data)
    imputed_df = pd.DataFrame(imputed_values)
    imputed_df.index = target_data.index
    imputed_df.columns = target_data.columns
    '''
    target_data = res_target_impute_df.copy()

    if len(target_data.columns) < len(source_data.columns):
        source_data = source_data[target_data.columns]

    # Feature selection-based on k-means clustering and constrcut CpG clusters
    source_data = source_data.T
    num_feature = 3000
    kmeans = KMeans(n_clusters=num_feature, random_state=42, verbose=0)
    kmeans.fit(source_data)

    result_cluster = pd.DataFrame({'cpg': source_data.index.tolist(), 'cluster': kmeans.labels_})
    result_cluster.to_csv(os.path.join(preprocessing_path, "cpg_cluster_info.csv"), mode="w", index=False)

    result_cluster.sort_values(by="cluster", inplace=True)

    tmp = pd.merge(source_data, result_cluster, left_index=True, right_on="cpg")
    tmp.set_index("cpg", inplace=True, drop=True)

    cluster_grouped = tmp.groupby('cluster')
    cluster_data = cluster_grouped.median()
    source_data = cluster_data.T

    source_data = pd.merge(source_data, source_data_y, left_index=True, right_index=True)
    source_data_y = source_data[['subtype']]
    del source_data['subtype']

    y_category_list = source_data_y['subtype'].unique().tolist()
    y_int_list = []
    for i in range(len(y_category_list)):
        y_int_list.append(i)

    category_info_df = pd.DataFrame({'subtype': y_category_list, 'subtype_int': y_int_list})
    category_info_df.to_csv(os.path.join(preprocessing_path, "subtype_category_info.csv"), mode="w", index=False)

    source_data_y['subtype'].replace(y_category_list, y_int_list, inplace=True)
    source_data.to_csv(os.path.join(preprocessing_path, "source_X.csv"), mode="w", index=True)
    source_data_y.to_csv(os.path.join(preprocessing_path, "source_y.csv"), mode="w", index=True)

    target_data = target_data.T
    tmp = pd.merge(target_data, result_cluster, left_index=True, right_on="cpg")
    tmp.set_index("cpg", inplace=True, drop=True)

    cluster_grouped = tmp.groupby('cluster')
    cluster_data = cluster_grouped.median()
    cluster_data = cluster_data.T

    target_data = cluster_data

    target_data = pd.merge(target_data, target_data_batch_info, left_index=True, right_index=True)

    y_category_list = target_data['Batch'].unique().tolist()
    y_int_list = []
    for i in range(len(y_category_list)):
        y_int_list.append(i + 1)

    batch_info_df = pd.DataFrame({'batch': y_category_list, 'domain_idx': y_int_list})
    batch_info_df.to_csv(os.path.join(preprocessing_path, "batch_category_info.csv"), mode="w", index=False)
    target_data['domain_idx'] = target_data['Batch'].copy()
    target_data['domain_idx'].replace(y_category_list, y_int_list, inplace=True)

    target_data.to_csv(os.path.join(preprocessing_path, "target_X.csv"), mode="w", index=True)

def run_cancersubminer(
        user_id: int,
        project_id: int,
        job_id: int,
        model_parameters: list,
        metadata_file: str,
):
    warnings.filterwarnings("ignore")

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    class MyBaseDataset(Dataset):
        def __init__(self, x_data, y_data):
            self.x_data = x_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.x_data.shape[0]

    class UnlabelDataset(Dataset):
        def __init__(self, x_data):
            self.x_data = x_data

        def __getitem__(self, index):
            return self.x_data[index]

        def __len__(self):
            return self.x_data.shape[0]

    class DomainDataset(Dataset):
        def __init__(self, x_data, y_data):
            self.x_data = x_data
            self.y_data = y_data

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.x_data.shape[0]

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"Using {device} device")

    is_automatically_estimation_required = model_parameters[0]
    num_subtype_user_defined = model_parameters[1]

    result_dir = job_result_path(user_id, project_id, job_id)
    os.makedirs(result_dir, exist_ok=True)

    preprocessing_path = job_preprocessing_path(user_id, project_id, job_id)

    x_filename = os.path.join(preprocessing_path, "source_X.csv")
    y_filename = os.path.join(preprocessing_path, "source_y.csv")
    target_filename = os.path.join(preprocessing_path, "target_X.csv")

    raw_x = pd.read_csv(x_filename, index_col=0)
    raw_y = pd.read_csv(y_filename, index_col=0)

    raw_target_x = pd.read_csv(target_filename, index_col=0)
    raw_target_domain_y = raw_target_x['domain_idx'].tolist()

    y_train = raw_y['subtype'].tolist()
    num_subtype_original = len(set(y_train))
    y_train = np.array(y_train)

    del raw_target_x['domain_idx']
    del raw_target_x['Batch']

    raw_target_x_index = np.array(raw_target_x.index.tolist())
    x_train_index = np.array(raw_x.index.tolist())
    domain_x_index = np.concatenate((x_train_index, raw_target_x_index))

    raw_target_x = raw_target_x.values
    x_train = raw_x.values

    domain_x = np.append(x_train, raw_target_x, axis=0)

    raw_source_domain_y = np.zeros(len(y_train), dtype=int)  # TCGA label : 0
    domain_y = np.append(raw_source_domain_y, raw_target_domain_y)

    num_domain = len(set(domain_y))

    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)

    domain_x = torch.from_numpy(domain_x)
    domain_y = torch.from_numpy(domain_y)

    target_x = torch.from_numpy(raw_target_x)

    num_feature = len(x_train[0])
    num_train = len(x_train)
    num_test = len(raw_target_x)

    train_dataset = MyBaseDataset(x_train, y_train)
    domain_dataset = DomainDataset(domain_x, domain_y)

    batch_size = 128
    target_batch_size = 128
    test_target_batch_size = 64

    source_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    domain_dataloader = DataLoader(domain_dataset, batch_size=batch_size, shuffle=True)

    n_fe_embed1 = 1024
    n_fe_embed2 = 512
    n_c_h1 = 256
    n_c_h2 = 64
    n_d_h1 = 256
    n_d_h2 = 64

    class FeatureExtractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.feature_layer = nn.Sequential(
                nn.Linear(num_feature, n_fe_embed1),
                nn.LeakyReLU(),
                nn.Linear(n_fe_embed1, n_fe_embed2),
                nn.LeakyReLU()
            )

        def forward(self, x):
            embedding = self.feature_layer(x)
            return embedding

    class SubtypeClassifier(nn.Module):
        def __init__(self, n_class):
            super().__init__()
            # self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(n_fe_embed2, n_c_h1),
                nn.LeakyReLU(),
                nn.Linear(n_c_h1, n_c_h2),
                nn.LeakyReLU(),
                nn.Linear(n_c_h2, n_class)
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    class DomainDiscriminator(nn.Module):
        def __init__(self):
            super().__init__()
            self.disc_layer = nn.Sequential(
                nn.Linear(n_fe_embed2, n_d_h1),
                nn.LeakyReLU(),
                nn.Linear(n_d_h1, n_d_h2),
                nn.LeakyReLU(),
                nn.Linear(n_d_h2, num_domain)
            )

        def forward(self, x):
            domain_logits = self.disc_layer(x)
            return domain_logits

    feature_extract_model = FeatureExtractor().to(device)
    subtype_pred_model = SubtypeClassifier(num_subtype_original).to(device)
    domain_disc_model = DomainDiscriminator().to(device)

    c_loss = nn.CrossEntropyLoss()  # Already have softmax
    domain_loss = nn.CrossEntropyLoss()

    fe_optimizer = torch.optim.Adam(feature_extract_model.parameters(), lr=1e-5)
    c_optimizer = torch.optim.Adam(subtype_pred_model.parameters(), lr=1e-5)
    d_optimizer = torch.optim.Adam(domain_disc_model.parameters(), lr=1e-6)

    def pretrain_classifier(epoch, dataloader, fe_model, c_model, c_loss, fe_optimizer, c_optimizer):
        size = len(dataloader.dataset)
        correct = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X = X.float()
            extracted_feature = fe_model(X)
            pred = c_model(extracted_feature)
            loss = c_loss(pred, y)
            fe_optimizer.zero_grad()
            c_optimizer.zero_grad()
            loss.backward()
            fe_optimizer.step()
            c_optimizer.step()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss = loss.item()
        correct /= size
        if epoch % 10 == 0:
            print(f"[PT Epoch {epoch + 1}] \tTraining loss: {loss:>5f}, Training Accuracy: {(100 * correct):>0.2f}%")
        return loss, correct

    def get_embed(dataloader, fe_model, c_model):
        fe_model.eval()
        c_model.eval()
        X_embed_list = []
        y_list = []
        raw_X_list = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                X = X.float()
                X_embed = fe_model(X)
                X_embed_list.append(X_embed)
                y_list.append(y)
                raw_X_list.append(X)
        X_embed_list = torch.cat(X_embed_list, 0)
        y_list = torch.cat(y_list, 0)
        raw_X_list = torch.cat(raw_X_list, 0)
        return X_embed_list, y_list, raw_X_list

    def pretest_new_subtype(new_X_raw, new_y, subtype_num):
        new_X_raw, new_y = torch.FloatTensor(new_X_raw), torch.from_numpy(new_y)
        tmp_dataset = MyBaseDataset(new_X_raw, new_y)
        tmp_dataloader = DataLoader(tmp_dataset, batch_size=batch_size, shuffle=False)
        tmp_subtype_model = SubtypeClassifier(subtype_num).to(device)
        tmp_fe_model = FeatureExtractor().to(device)
        tmp_fe_optimizer = torch.optim.Adam(tmp_fe_model.parameters(), lr=1e-4)
        tmp_c_optimizer = torch.optim.Adam(tmp_subtype_model.parameters(), lr=1e-5)
        tmp_acc = 0.0
        epoch = 0
        stop_count = 0
        while (tmp_acc != 1.0) and (stop_count < 10):
            epoch += 1
            tmp_loss, tmp_acc = pretrain_classifier(epoch, tmp_dataloader, tmp_fe_model, tmp_subtype_model, c_loss,
                                                    tmp_fe_optimizer, tmp_c_optimizer)
            if tmp_acc == 1.0:
                stop_count += 1
        tmp_X_embed, tmp_y_torch, tmp_X_raw = get_embed(tmp_dataloader, tmp_fe_model, tmp_subtype_model)
        tmp_X_embed = tmp_X_embed.detach().cpu().numpy()
        tmp_y_torch = tmp_y_torch.detach().cpu().numpy()
        tmp_X_raw = tmp_X_raw.detach().cpu().numpy()
        return tmp_X_embed, tmp_y_torch, tmp_X_raw, tmp_loss

    def cluster_features_for_subtyping(X_embed_torch, y_torch, X_torch):
        is_new_subtype_identified = 0
        X_embed = X_embed_torch.detach().cpu().numpy()
        y = y_torch.detach().cpu().numpy()
        X_raw = X_torch.detach().cpu().numpy()
        prev_silhouette = 0.0
        if is_automatically_estimation_required == 1:
            num_cluster = num_subtype_original - 2  # len(np.unique(y)) - 2
        else:
            num_cluster = num_subtype_user_defined
        for i in range(10):
            kmeans = KMeans(n_clusters=num_cluster, random_state=42).fit(X_embed)
            kmeans_label = kmeans.labels_
            new_silhouette = silhouette_score(X_embed, kmeans_label)
            print("prev silhouette : ", prev_silhouette, "new silhouette : ", new_silhouette)
            if (new_silhouette > prev_silhouette) and (new_silhouette - prev_silhouette > 0.01):
                prev_silhouette = new_silhouette
                y = kmeans_label.copy()
                # if i == 0 :
                #    continue
                # print("Extend subtypes with " + str(num_cluster) + " clusters")
                if is_automatically_estimation_required == 1:
                    num_cluster += 1
                else:
                    break
                is_new_subtype_identified = 1
            else:
                # print("Stop finding with "+ str(num_cluster-1) + " clusters")
                break
        return X_raw, X_embed, y, is_new_subtype_identified

    def cluster_high_entropy_samples(X_embed_torch, y_torch, X_torch, c_model, c_loss):
        is_new_subtype_identified = 0
        # X_embed_torch, y_torch, X_torch = get_embed(train_dataloader, feature_extract_model, subtype_pred_model)
        prev_pred = c_model(X_embed_torch)
        prev_loss = c_loss(prev_pred, y_torch)
        func_sm = nn.Softmax()
        #
        # Calculate the confidence score
        prev_conf_score = func_sm(prev_pred)
        prev_conf_score = pd.DataFrame(prev_conf_score.detach().cpu().numpy())
        # low_conf_df = prev_conf_score[prev_conf_score.max(axis = 1) < 0.95]
        subtype_list = prev_conf_score.columns.tolist()
        #
        # Calculate the entropy
        tmp_entropy_series = (prev_conf_score[subtype_list[0]] * np.log2(prev_conf_score[subtype_list[0]]))
        for subtype in subtype_list[1:]:
            tmp_entropy_series += (prev_conf_score[subtype] * np.log2(prev_conf_score[subtype]))
        tmp_entropy_series = tmp_entropy_series * -1.0
        prev_conf_score['entropy'] = tmp_entropy_series
        #
        # 1. Check whether having high entropy (outlier)
        q1_entropy = prev_conf_score['entropy'].quantile(0.25)
        q3_entropy = prev_conf_score['entropy'].quantile(0.75)
        iqr_entropy = q3_entropy - q1_entropy
        outlier_criteria_entropy = q3_entropy + 1.5 * iqr_entropy
        #
        high_entropy_df = prev_conf_score[prev_conf_score['entropy'] > outlier_criteria_entropy]
        del high_entropy_df['entropy']
        #
        # 2. Select the samples having the confidence score less than 0.95
        high_entropy_low_conf_df = high_entropy_df[high_entropy_df.max(axis=1) < 0.95]
        # tmp_sample_list = high_entropy_low_conf_df.index.tolist()
        tmp_list = []
        for i in range(len(high_entropy_low_conf_df)):
            top_two_sub = high_entropy_low_conf_df.iloc[i].sort_values(ascending=False)[:2].index.sort_values().tolist()
            tmp_list.append(str(top_two_sub[0]) + "_" + str(top_two_sub[1]))
        #
        high_entropy_low_conf_df['result'] = tmp_list
        new_subtype_candidate_list = high_entropy_low_conf_df['result'].unique().tolist()
        #
        #
        X_raw = X_torch.detach().cpu().numpy()
        X_embed = X_embed_torch.detach().cpu().numpy()
        y = y_torch.detach().cpu().numpy()
        num_subtype = len(subtype_list)
        for candidate in new_subtype_candidate_list:
            # candidate = new_subtype_candidate_list[0]
            tmp_new_subtype_candidate_sample_list = high_entropy_low_conf_df[
                high_entropy_low_conf_df['result'] == candidate].index.tolist()
            if len(tmp_new_subtype_candidate_sample_list) < 5:
                continue
            #
            prev_silhouette = silhouette_score(X_embed, y)
            new_y = y.copy()
            new_y[tmp_new_subtype_candidate_sample_list] = num_subtype
            new_silhouette = silhouette_score(X_embed, new_y)
            if new_silhouette > prev_silhouette:
                tmp_X_embed, tmp_y, tmp_X_raw, tmp_loss = pretest_new_subtype(X_raw, new_y, num_subtype + 1)
                if (tmp_loss < prev_loss):
                    num_subtype += 1
                    prev_loss = tmp_loss
                    prev_silhouette = new_silhouette
                    X_embed = tmp_X_embed.copy()
                    y = tmp_y.copy()
                    X_raw = tmp_X_raw.copy()
                    print("Found new subtype on subtype " + str(candidate))
                    is_new_subtype_identified = 1
                else:
                    print("No found on subtype " + str(candidate))
        return X_raw, X_embed, y, is_new_subtype_identified

    def cluster_sources(X_embed_torch, y_torch, X_raw_torch, c_model, c_loss):
        is_new_subtype_identified = 0
        prev_pred = c_model(X_embed_torch)  # c_model
        prev_loss = c_loss(prev_pred, y_torch)
        prev_loss = prev_loss.item()
        #
        X_embed = X_embed_torch.detach().cpu().numpy()
        X_raw = X_raw_torch.detach().cpu().numpy()
        y = y_torch.detach().cpu().numpy()
        prev_silhouette = silhouette_score(X_embed, y)
        subtype_list = np.unique(y)
        num_subtype = len(subtype_list)
        #
        for subtype in subtype_list:
            X_embed = pd.DataFrame(X_embed)
            X_embed['subtype'] = y
            #
            X_embed_subtype = X_embed[X_embed['subtype'] == subtype]
            X_raw_subtype = X_raw[X_embed['subtype'] == subtype]
            #
            del X_embed_subtype['subtype']
            X_embed_others = X_embed[X_embed['subtype'] != subtype]
            X_raw_others = X_raw[X_embed['subtype'] != subtype]
            # clustering
            kmeans = KMeans(n_clusters=2, random_state=42).fit(X_embed_subtype)
            X_embed_subtype['subtype'] = kmeans.labels_
            X_embed_subtype.replace({'subtype': {0: subtype, 1: num_subtype}}, inplace=True)
            #
            new_X_embed = pd.concat([X_embed_subtype, X_embed_others], axis=0)
            new_y = new_X_embed['subtype'].values
            new_X_raw = np.concatenate([X_raw_subtype, X_raw_others], axis=0)
            #
            tmp_X_embed, tmp_y, tmp_X_raw, tmp_loss = pretest_new_subtype(new_X_raw, new_y, num_subtype + 1)
            new_silhouette = silhouette_score(tmp_X_embed, new_y)
            if (tmp_loss < prev_loss) and (prev_silhouette < new_silhouette):
                print("Found new subtype on subtype " + str(subtype))
                print(prev_silhouette, new_silhouette)
                num_subtype += 1
                prev_loss = tmp_loss
                prev_silhouette = new_silhouette
                X_embed = tmp_X_embed.copy()
                y = tmp_y.copy()
                X_raw = tmp_X_raw.copy()
                is_new_subtype_identified = 1
            else:
                del X_embed['subtype']
                print("No found on subtype " + str(subtype))
                print(prev_silhouette, new_silhouette)
        return X_raw, X_embed, y, is_new_subtype_identified

    def test_classifier(dataloader, fe_model, c_model, c_loss):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        fe_model.eval()
        c_model.eval()
        test_loss, test_acc = 0, 0
        pred_subtype_list = []
        label_subtype_list = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                X = X.float()
                extracted_feature = fe_model(X)
                pred = c_model(extracted_feature)
                test_loss += c_loss(pred, y).item()
                test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
                pred_subtype_list.append(pred.argmax(1))
                label_subtype_list.append(y)
        pred_subtype_list = torch.cat(pred_subtype_list, 0)
        label_subtype_list = torch.cat(label_subtype_list, 0)
        test_loss /= num_batches
        test_acc /= size
        print(f"\t\tTesting Accuracy: {(100 * test_acc):>0.3f}%, Avg loss: {test_loss:>5f} \n")
        return pred_subtype_list, label_subtype_list, test_acc

    def adversarial_train_disc(epoch, dataloader, fe_model, d_model, domain_loss, fe_optimizer, d_optimizer):
        size = len(dataloader.dataset)
        correct = 0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X = X.float()
            extracted_feature = fe_model(X)
            pred = d_model(extracted_feature)
            d_loss = domain_loss(pred, y)
            # Backpropagation
            fe_optimizer.zero_grad()
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        d_loss = d_loss.item()
        correct /= size
        if t % 10 == 0:
            print(f"[AT Epoch {epoch + 1}] Disc loss: {d_loss:>5f}, Training Accuracy: {(100 * correct):>0.2f}%",
                  end=", ")

    def adversarial_train_fe(epoch, dataloader, fe_model, d_model, domain_loss, fe_optimizer, d_optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            X = X.float()
            extracted_feature = fe_model(X)
            pred = d_model(extracted_feature)
            fake_y = torch.randint(low=0, high=num_domain, size=(len(y),))
            fake_y = fake_y.to(device)
            g_loss = domain_loss(pred, fake_y)
            # Backpropagation
            fe_optimizer.zero_grad()
            d_optimizer.zero_grad()
            g_loss.backward()
            fe_optimizer.step()
        g_loss = g_loss.item()
        if epoch % 10 == 0:
            print(f"Gen loss: {g_loss:>5f}")

    def ssl_train_classifier(epoch, source_dataloader, target_dataloader, fe_model, c_model, c_loss, fe_optimizer,
                             c_optimizer):
        source_size = len(source_dataloader.dataset)
        target_size = len(target_dataloader.dataset)
        #
        # 1. Obtain the pseudo-label for target dataset
        #
        target_pseudo_label = torch.empty((0), dtype=torch.int64)
        target_pseudo_label = target_pseudo_label.to(device)
        target_X_embed_list = []
        #
        for batch, (target_X, target_y) in enumerate(target_dataloader):
            target_X, target_y = target_X.to(device), target_y.to(device)
            target_X = target_X.float()
            extracted_feature = fe_model(target_X)
            batch_target_pred = c_model(extracted_feature)
            batch_pseudo_label = batch_target_pred.argmax(1)
            target_pseudo_label = torch.cat((target_pseudo_label, batch_pseudo_label), 0)
            target_X_embed_list.append(extracted_feature)
            if batch == 0:
                target_loss = c_loss(batch_target_pred, target_y)
            else:
                target_loss = target_loss + c_loss(batch_target_pred, target_y)
        target_loss = target_loss / (batch + 1)
        target_X_embed_list = torch.cat(target_X_embed_list, 0)
        #
        # Define alpha value
        alpha_f = 0.01
        t1 = 100
        t2 = 200
        if epoch < t1:
            alpha = 0
        elif epoch < t2:
            alpha = (epoch - t1) / (t2 - t1) * alpha_f
        else:
            alpha = alpha_f
        #
        # 2. Calculate the loss for the source dataset
        #
        correct = 0
        for batch, (source_X, source_y) in enumerate(source_dataloader):
            source_X, source_y = source_X.to(device), source_y.to(device)
            source_X = source_X.float()
            source_extracted_feature = fe_model(source_X)
            source_pred = c_model(source_extracted_feature)
            source_loss = c_loss(source_pred, source_y)
            ssl_loss = source_loss + alpha * target_loss
            # Backpropogation
            target_loss.detach_()
            fe_optimizer.zero_grad()
            c_optimizer.zero_grad()
            ssl_loss.backward()  # retain_graph=True
            fe_optimizer.step()
            c_optimizer.step()
            correct += (source_pred.argmax(1) == source_y).type(torch.float).sum().item()
        ssl_loss = ssl_loss.item()
        source_loss = source_loss.item()
        target_loss = target_loss.item()
        correct /= source_size
        if epoch % 10 == 0:
            print(
                f"[SSL Epoch {epoch + 1}] alpha : {alpha:>3f}, SSL loss: {ssl_loss:>5f}, source loss: {source_loss:>5f}, target loss: {target_loss:>4f}, source ACC: {(100 * correct):>0.2f}%\n")
        #
        # Adjust the target labels having mis-alignment using cluster purity
        #
        target_X_embed = target_X_embed_list.detach().cpu().numpy()
        target_pseudo_y = target_pseudo_label.detach().cpu().numpy()
        X_embed_torch, y_torch, X_raw_torch = get_embed(source_dataloader, fe_model, c_model)
        X_embed = X_embed_torch.detach().cpu().numpy()
        y = y_torch.detach().cpu().numpy()
        num_subtype = len(np.unique(y))
        total_embed = np.concatenate((X_embed, target_X_embed), axis=0)
        #
        kmeans = KMeans(n_clusters=num_subtype, random_state=42).fit(total_embed)
        kmeans_label = kmeans.labels_
        #
        y_df = pd.DataFrame({'original': y})
        y_df['Batch'] = 'source'
        target_y_df = pd.DataFrame({'original': target_pseudo_y})
        target_y_df['Batch'] = 'target'
        total_y_df = pd.concat([y_df, target_y_df], axis=0)
        total_y_df['cluster'] = kmeans_label
        target_y_df['new'] = target_y_df['original'].copy()
        #
        cluster_list = np.unique(kmeans_label)
        for cluster in cluster_list:
            tmp_cluster = total_y_df[total_y_df['cluster'] == cluster]
            tmp_cluster_source = tmp_cluster[tmp_cluster['Batch'] == "source"]
            if len(tmp_cluster_source) == 0:
                continue
            subtype_assigned_to_cluster = tmp_cluster_source['original'].value_counts().index[0]
            #
            tmp_cluster_target = tmp_cluster[tmp_cluster['Batch'] == "target"]
            tmp_cluster_target_index = tmp_cluster_target.index.tolist()
            target_y_df.at[tmp_cluster_target_index, 'new'] = subtype_assigned_to_cluster
        #
        target_pseudo_y_adjusted = target_y_df['new'].values
        target_pseudo_y_adjusted = torch.from_numpy(target_pseudo_y_adjusted)
        target_pseudo_y_adjusted = target_pseudo_y_adjusted.to(device)
        return target_pseudo_y_adjusted  # target_pseudo_label

    def class_alignment_train(epoch, domain_x, domain_y, domain_z, fe_model, c_model, fe_optimizer, c_optimizer):
        # for batch, (X, y_domain, z_subtype) in enumerate(domain_dataloader):
        X, y_domain, z_subtype = domain_x.to(device), domain_y.to(device), domain_z.to(device)
        # X = X.float()
        batch_subtype_list = z_subtype.unique()
        num_subtype = len(batch_subtype_list)
        X_embed = fe_model(X)  # feature_extract_model, subtype_pred_model
        pred = c_model(X_embed)
        func_sm = nn.Softmax()
        conf_score = func_sm(pred)
        conf_score = pd.DataFrame(conf_score.detach().cpu().numpy())
        low_conf_df = conf_score[conf_score.max(axis=1) < 0.95]
        low_conf_sample_list = low_conf_df.index.tolist()
        high_conf_df = conf_score[conf_score.max(axis=1) > 0.95]
        high_conf_sample_list = high_conf_df.index.tolist()
        #
        X_embed_high_conf = X_embed[high_conf_sample_list]
        y_domain_high_conf = y_domain[high_conf_sample_list]
        z_subtype_high_conf = z_subtype[high_conf_sample_list]
        #
        align_loss = torch.zeros((1), dtype=torch.float64)
        align_loss = align_loss.to(device)
        #
        for subtype in batch_subtype_list:
            sample_idx_list = (z_subtype_high_conf == subtype).nonzero(as_tuple=True)[0]
            if len(sample_idx_list) < 1:
                continue
            # else :
            tmp_x = X_embed_high_conf[sample_idx_list]
            tmp_y = y_domain_high_conf[sample_idx_list]
            tmp_z = z_subtype_high_conf[sample_idx_list]
            batch_domain_list = tmp_y.unique()
            domain_centroid_stack = []
            for domain in batch_domain_list:
                domain_idx_list = (tmp_y == domain).nonzero(as_tuple=True)[0]
                if len(domain_idx_list) != 1:
                    tmp_x_domain = tmp_x[domain_idx_list]
                    tmp_centroid = torch.div(torch.sum(tmp_x_domain, dim=0), len(domain_idx_list))
                    domain_centroid_stack.append(tmp_centroid)
            if len(domain_centroid_stack) == 0:
                continue
            else:
                domain_centroid_stack = torch.stack(domain_centroid_stack)
            subtype_centroid = torch.mean(domain_centroid_stack, dim=0)
            # Duplicate the subtype centroid to get dist with each domain_centroid
            subtype_centroid_stack = []
            for i in range(len(domain_centroid_stack)):
                subtype_centroid_stack.append(subtype_centroid)
            subtype_centroid_stack = torch.stack(subtype_centroid_stack)
            pdist_stack = nn.L1Loss()(subtype_centroid_stack, domain_centroid_stack)
            align_loss += torch.mean(pdist_stack, dim=0)
        align_known_cluster_loss = align_loss.item()
        #
        # Check whether having better silhouette score
        #
        prev_silhouette = silhouette_score(X_embed.detach().cpu().numpy(), z_subtype.detach().cpu().numpy())
        new_z_subtype = z_subtype.detach().cpu().numpy()
        new_z_subtype[low_conf_sample_list] = num_subtype
        new_silhouette = silhouette_score(X_embed.detach().cpu().numpy(), new_z_subtype)
        #
        pdist_stack = 0.0
        if new_silhouette > prev_silhouette:
            print("Identify new type on target")
            X_embed_low_conf = X_embed[low_conf_sample_list]
            y_domain_low_conf = y_domain[low_conf_sample_list]
            z_subtype_low_conf = z_subtype[low_conf_sample_list]
            #
            tmp_centroid = torch.div(torch.sum(X_embed_low_conf, dim=0), len(X_embed_low_conf))
            subtype_centroid_stack = []
            for i in range(len(X_embed_low_conf)):
                subtype_centroid_stack.append(tmp_centroid)
            subtype_centroid_stack = torch.stack(subtype_centroid_stack)
            pdist_stack = nn.L1Loss()(subtype_centroid_stack, X_embed_low_conf)
            align_loss += pdist_stack
            num_subtype += 1
            new_z_subtype = torch.from_numpy(new_z_subtype)
            new_z_subtype = new_z_subtype.to(device)
            z_subtype = new_z_subtype
            pdist_stack = pdist_stack.item()
            #
        # if align_loss == 0.0 :
        #    continue
        align_loss = align_loss / num_subtype
        fe_optimizer.zero_grad()
        c_optimizer.zero_grad()
        align_loss.backward()
        fe_optimizer.step()
        c_optimizer.step()
        align_loss = align_loss.item()
        if epoch % 10 == 0:
            print(
                f"[CA Epoch {epoch + 1}] align loss: {align_loss:>5f}, known type loss : {align_known_cluster_loss:>5f}, new type loss : {pdist_stack:>5f}\n")
        return X, X_embed, y_domain, z_subtype, conf_score

    def reassign_source(X_embed_torch, y_torch, y_new, c_model):
        # 1. Cluster source samples
        # X_embed_torch = torch.FloatTensor(X_embed)
        # X_embed_torch = X_embed_torch.to(device)
        X_conf_score = c_model(X_embed_torch)
        X_conf_score = nn.Softmax()(X_conf_score)
        X_conf_score = X_conf_score.detach().cpu().numpy()
        X_conf_score = pd.DataFrame(X_conf_score)
        #
        # 2.
        y = y_torch.detach().cpu().numpy()
        tmp_num_cluster = len(np.unique(y_new))
        tmp_y_df = pd.DataFrame({'origin': y, 'cluster': y_new})
        tmp_y_df = pd.concat([tmp_y_df, X_conf_score], axis=1)
        #
        for i in range(tmp_num_cluster):
            tmp_y_cluster = tmp_y_df[tmp_y_df['cluster'] == i]
            tmp_high_conf = tmp_y_cluster[tmp_y_cluster[X_conf_score.columns].max(axis=1) >= 0.95]
            tmp_maj_subtype = tmp_high_conf.value_counts('origin').index[0]
            tmp_low_conf = tmp_y_cluster[tmp_y_cluster[X_conf_score.columns].max(axis=1) < 0.95]
            if len(tmp_low_conf) > 0:
                tmp_low_conf_idx = tmp_low_conf.index.tolist()
                tmp_y_df['origin'][tmp_low_conf_idx] = tmp_maj_subtype
        y_reassigned = tmp_y_df['origin'].tolist()
        return y_reassigned

    def get_embed_domain(domain_dataloader, fe_model, c_model):
        fe_model.eval()
        c_model.eval()
        X_embed_list = []
        domain_list = []
        pred_subtype_list = []
        label_list = []  # Can be used only for source dataset
        with torch.no_grad():
            for batch, (X, y, z) in enumerate(domain_dataloader):
                X, y, z = X.to(device), y.to(device), z.to(device)
                X = X.float()
                X_embed = fe_model(X)
                pred = c_model(X_embed)
                pred_subtype_list.append(pred.argmax(1))
                X_embed_list.append(X_embed)
                domain_list.append(y)
                label_list.append(z)
        X_embed_list = torch.cat(X_embed_list, 0)
        pred_subtype_list = torch.cat(pred_subtype_list, 0)
        domain_list = torch.cat(domain_list, 0)
        label_list = torch.cat(label_list, 0)
        return X_embed_list, domain_list, pred_subtype_list, label_list

    pt_epochs = 800
    ad_train_epochs = 500
    ssl_train_epochs = 300
    ft_epochs = 300

    test_acc_pt = 0.0
    test_acc_at = 0.0
    test_acc_ssl = 0.0
    test_acc_ft = 0.0

    test_target_acc_pt = 0.0
    test_target_acc_ssl = 0.0
    test_target_acc_at = 0.0
    test_target_acc_ft = 0.0

    # 1. Pre-training
    print("=========================================")
    print("[Step 1] Pretraining")
    print("=========================================")
    early_stop = 0
    acc_1_stop = 0
    prev_loss = 100.0
    for t in range(pt_epochs):
        tmp_loss, tmp_acc = pretrain_classifier(t, train_dataloader, feature_extract_model, subtype_pred_model, c_loss,
                                                fe_optimizer, c_optimizer)

    # 2. Cluster the source samples based on the subtype
    print("=========================================================")
    print("[Step 2-1] Clustering source samples based on the k-means")
    print("=========================================================")
    X_embed_torch, y_torch, X_raw_torch = get_embed(source_dataloader, feature_extract_model, subtype_pred_model)

    for epoch in range(8):
        print("EPOCH ", epoch)
        X_raw, X_embed, y_new, is_new_subtype_identified = cluster_features_for_subtyping(X_embed_torch, y_torch,
                                                                                          X_raw_torch)
        y_reassigned = reassign_source(X_embed_torch, y_torch, y_new, subtype_pred_model)
        train_dataset = MyBaseDataset(X_raw, y_reassigned)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        for t in range(pt_epochs):
            tmp_loss, tmp_acc = pretrain_classifier(t, train_dataloader, feature_extract_model, subtype_pred_model,
                                                    c_loss, fe_optimizer, c_optimizer)
            if tmp_acc == 1.0:
                break
        X_embed_torch, y_torch, X_raw_torch = get_embed(train_dataloader, feature_extract_model, subtype_pred_model)

    X_embed_torch, y_torch, X_raw_torch = get_embed(source_dataloader, feature_extract_model, subtype_pred_model)
    X_raw, X_embed, y_new, is_new_subtype_identified = cluster_features_for_subtyping(X_embed_torch, y_torch,
                                                                                      X_raw_torch)
    X_raw_torch = torch.FloatTensor(X_raw)
    y_new_torch = torch.from_numpy(y_new)
    y_new_torch = torch.tensor(y_new_torch, dtype=torch.long)
    train_dataset = MyBaseDataset(X_raw_torch, y_new_torch)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2-3. Retrain the model wit the newly defined subtype using the source dataset
    print("==============================================")
    print("[Step 2-3] Retraining with the source samples")
    print("==============================================")
    num_subtype = len(np.unique(y_new))
    feature_extract_model = FeatureExtractor().to(device)
    subtype_pred_model = SubtypeClassifier(num_subtype).to(device)
    fe_optimizer = torch.optim.Adam(feature_extract_model.parameters(), lr=1e-4)
    c_optimizer = torch.optim.Adam(subtype_pred_model.parameters(), lr=1e-5)
    for t in range(pt_epochs):
        tmp_loss, tmp_acc = pretrain_classifier(t, train_dataloader, feature_extract_model, subtype_pred_model, c_loss,
                                                fe_optimizer, c_optimizer)

    X_embed_torch, y_torch, X_raw_torch = get_embed(source_dataloader, feature_extract_model, subtype_pred_model)
    # X_embed_df = pd.DataFrame(X_embed_torch.detach().cpu().numpy())
    # X_embed_df.index = x_train_index
    # X_embed_df.to_csv(os.path.join(result_dir, "ct_source_embed_group_" + group_num + ".csv"), mode = "w", index = True)
    # np.savetxt(os.path.join(result_dir, "ct_source_pred_group_" + group_num + ".csv"), y_torch.detach().cpu().numpy(), fmt="%.0f", delimiter=",")
    # X_embed_torch, y_torch, X_raw_torch = get_embed(train_dataloader, feature_extract_model, subtype_pred_model)
    # del X_embed_df

    # tmp_silhouette = silhouette_score(X_embed_torch.detach().cpu().numpy(), y_pred_torch)
    # X_raw, X_embed, y_new, is_new_subtype_identified = cluster_features_for_subtyping(X_embed_torch, y_torch, X_raw_torch)

    # 3. Adversarial traning
    print("================================")
    print(" [Step 3] Adversarial training")
    print("================================")
    target_init_y = torch.randint(low=0, high=num_subtype, size=(len(target_x),))
    target_dataset = MyBaseDataset(target_x, target_init_y)
    target_dataloader = DataLoader(target_dataset, batch_size=target_batch_size, shuffle=False)

    for t in range(ad_train_epochs):
        adversarial_train_disc(t, domain_dataloader, feature_extract_model, domain_disc_model, domain_loss,
                               fe_optimizer, d_optimizer)
        adversarial_train_fe(t, domain_dataloader, feature_extract_model, domain_disc_model, domain_loss, fe_optimizer,
                             d_optimizer)

    print("================================")
    print(" [Step 4] SSL training")
    print("================================")

    # 4. SSL training
    for t in range(ssl_train_epochs):
        target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, feature_extract_model,
                                                   subtype_pred_model, c_loss, fe_optimizer, c_optimizer)
        target_dataset = MyBaseDataset(target_x, target_pseudo_label)
        target_dataloader = DataLoader(target_dataset, batch_size=target_batch_size)

    print("================================")
    print(" [Step 5] Fine-tuning")
    print("================================")
    # 5. Fine-tuning
    for t in range(ft_epochs):
        target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, feature_extract_model,
                                                   subtype_pred_model, c_loss, fe_optimizer, c_optimizer)
        target_dataset = MyBaseDataset(target_x, target_pseudo_label)
        target_dataloader = DataLoader(target_dataset, batch_size=target_batch_size)

        target_pseudo_label = target_pseudo_label.to("cpu")
        domain_z = torch.cat((y_new_torch, target_pseudo_label), 0)
        domain_x = np.append(X_raw, raw_target_x, axis=0)
        domain_y = np.append(raw_source_domain_y, raw_target_domain_y)

        domain_x = torch.FloatTensor(domain_x)
        domain_y = torch.from_numpy(domain_y)
        X_final_raw, X_final_embed, y_final_domain, z_final_subtype, X_final_conf_score = class_alignment_train(t,
                                                                                                                domain_x,
                                                                                                                domain_y,
                                                                                                                domain_z,
                                                                                                                feature_extract_model,
                                                                                                                subtype_pred_model,
                                                                                                                fe_optimizer,
                                                                                                                c_optimizer)

    X_final_embed = X_final_embed.detach().cpu().numpy()
    y_final_domain = y_final_domain.detach().cpu().numpy()
    z_final_subtype = z_final_subtype.detach().cpu().numpy()

    reducer = umap.UMAP()
    umap_embed = reducer.fit_transform(X_final_embed)
    tmp_num_cluster = len(np.unique(z_final_subtype))
    kmeans = KMeans(n_clusters=tmp_num_cluster).fit(umap_embed)

    X_final_embed = pd.DataFrame(X_final_embed)
    X_final_embed.index = domain_x_index

    X_final_embed['Batch'] = y_final_domain
    X_final_embed['Pred_subtype'] = kmeans.labels_
    X_final_embed['Pred_subtype'] = X_final_embed['Pred_subtype'] + 1
    X_final_embed['Pred_subtype'] = X_final_embed['Pred_subtype'].astype('string')
    X_final_embed['Pred_subtype'] = 'Cluster ' + X_final_embed['Pred_subtype']

    domain_info = pd.read_csv(os.path.join(preprocessing_path, "batch_category_info.csv"), index_col=1)
    domain_info = domain_info.to_dict()
    domain_info['batch'][0] = 'Source'
    X_final_embed['Batch'] = X_final_embed['Batch'].replace(domain_info['batch'])

    subtype_info = pd.read_csv(os.path.join(preprocessing_path, "subtype_category_info.csv"), index_col=1)
    subtype_info = subtype_info.to_dict()
    X_final_embed['Original_subtype'] = raw_y['subtype']
    X_final_embed['Original_subtype'] = X_final_embed['Original_subtype'].replace(subtype_info['subtype'])
    X_final_embed.to_csv(os.path.join(result_dir, "input_for_viz.csv"), mode="w", index=True)

    X_pred = X_final_embed[['Batch', 'Pred_subtype']]
    X_pred.to_csv(os.path.join(result_dir, "results_subtyping.csv"), mode="w", index=True)

    umap_embed = pd.DataFrame(umap_embed)
    umap_embed.index = X_final_embed.index
    umap_embed['Batch'] = X_final_embed['Batch']
    umap_embed['Pred_subtype'] = X_final_embed['Pred_subtype']
    umap_embed['Original_subtype'] = X_final_embed['Original_subtype']
    umap_embed.to_csv(os.path.join(result_dir, "umap_cancersubminer.csv"), mode="w", index=True)

    # ================================
    # Creating CpG Info Table
    # ================================
    print("Creating CpG info table")

    # Load CpG info and clustering data
    cpg_info = pd.read_csv(settings.cpg_info_file, index_col=0, low_memory=False)
    cluster = pd.read_csv(os.path.join(preprocessing_path, "cpg_cluster_info.csv"), index_col=0)

    # Merge CpG clusters with CpG information and sort by cluster
    cpg_info_table = (
        cluster.merge(cpg_info, left_index=True, right_index=True, how='left')
        .assign(CpG=lambda df: df.index)
        .sort_values(by='cluster')
        .reset_index(drop=True)
    )

    # Save CpG info table
    cpg_info_table.to_csv(os.path.join(result_dir, "cpg_info_table.csv"), mode="w", index=False)
    print("Done")

    # ================================
    # Creating Heatmaps
    # ================================
    print("Creating heatmaps")

    def reduce_correlation_matrix(corr_matrix, num_features=30):
        """
        Reduce the correlation matrix to a 30x30 size by selecting the most variable features.

        Args:
            corr_matrix (pd.DataFrame): Full correlation matrix.
            num_features (int): Number of top features to retain (default: 30).

        Returns:
            pd.DataFrame: Reduced correlation matrix.
        """
        if corr_matrix.shape[0] <= num_features:
            return corr_matrix  # If fewer than 30 features, return as is.

        # Compute variance of each feature in the correlation matrix
        top_features = corr_matrix.var(axis=1).nlargest(num_features).index
        return corr_matrix.loc[top_features, top_features]

    # Load input data
    raw_x = pd.read_csv(x_filename, index_col=0)
    raw_y = pd.read_csv(y_filename, index_col=0)
    raw_x['subtype'] = raw_y['subtype'].replace(subtype_info['subtype'])

    # Load target dataset and remove unnecessary columns
    raw_target_x = pd.read_csv(target_filename, index_col=0)
    if 'domain_idx' in raw_target_x.columns:
        del raw_target_x['domain_idx']

    target_x = raw_target_x.copy()
    target_x['subtype'] = X_final_embed['Pred_subtype']

    all_x = pd.concat([raw_x, target_x], axis=0)
    if 'Batch' in all_x.columns:
        del all_x['Batch']

    # Generate heatmaps for all data (Grouped by subtype)
    for subtype, subtype_df in all_x.groupby('subtype', group_keys=False):
        subtype_df = subtype_df.copy()
        print(f"All_{subtype}: {len(subtype_df)}")

        subtype_df.drop(columns=['subtype'], inplace=True)

        # Compute Spearman Correlation and save heatmap
        reduced = reduce_correlation_matrix(subtype_df.corr(method='spearman'))
        filename = os.path.join(result_dir, f"All_{subtype}_heatmap.csv")
        reduced.to_csv(filename)
        print(f"All_{subtype}_heatmap done")

    # Generate heatmaps for source data (Grouped by subtype)
    for subtype, subtype_df in raw_x.groupby('subtype', group_keys=False):
        subtype_df = subtype_df.copy()
        print(f"Source_{subtype}: {len(subtype_df)}")

        subtype_df.drop(columns=['subtype'], inplace=True)

        # Compute Spearman Correlation and save heatmap
        reduced = reduce_correlation_matrix(subtype_df.corr(method='spearman'))
        filename = os.path.join(result_dir, f"Source_{subtype}_heatmap.csv")
        reduced.to_csv(filename)
        print(f"Source_{subtype}_heatmap done")

    # Generate heatmaps for target data (Grouped by batch and subtype)
    for batch, batch_df in target_x.groupby('Batch', group_keys=False):
        batch_df = batch_df.copy()
        batch_df.drop(columns=['Batch'], inplace=True)

        for subtype, subtype_df in batch_df.groupby('subtype', group_keys=False):
            subtype_df = subtype_df.copy()
            print(f"{batch}_{subtype}: {len(subtype_df)}")

            reduced = reduce_correlation_matrix(
                subtype_df.drop(columns=['subtype'], errors='ignore').corr(method='spearman'))
            filename = os.path.join(result_dir, f"{batch}_{subtype}_heatmap.csv")
            reduced.to_csv(filename)
            print(f"{batch}_{subtype}_heatmap done")

    print("Done")

    # ================================
    # Creating All Beta Values Table
    # ================================
    print("Creating all beta values with labels table")

    # Load source dataset and assign batch label
    raw_x = pd.read_csv(x_filename, index_col=0)
    raw_x['Batch'] = "Source"

    # Merge source and target datasets
    combined = pd.concat([raw_x, raw_target_x])
    combined['subtype'] = X_final_embed['Pred_subtype']

    # Save merged dataset
    combined.to_csv(os.path.join(result_dir, "preprocessed_dataset.csv"), index=True)
    print("Done")

    # ================================
    # Creating UMAP Data
    # ================================
    print("Creating UMAP data")

    # Initialize UMAP reducer
    reducer = umap.UMAP(n_components=2, random_state=42)

    # Generate UMAP for Uncorrected Data
    uncorrected = combined.copy().drop(columns=['Batch', 'subtype'], errors='ignore')
    uncorrected_umap_embedding = reducer.fit_transform(uncorrected)
    uncorrected_umap_df = pd.DataFrame(uncorrected_umap_embedding, columns=['x', 'y'], index=uncorrected.index)
    uncorrected_umap_df[['Batch', 'subtype']] = combined[['Batch', 'subtype']]
    uncorrected_umap_df.to_csv(os.path.join(result_dir, "uncorrected_umap_embedding.csv"))
    # for the batch != Source, their subtype should be Unknown. for the batch == Source, their subtype should be the original

    # Generate UMAP for Corrected Data
    corrected = umap_embed.copy().drop(columns=['Batch', 'Pred_subtype', 'Original_subtype'], errors='ignore')
    corrected_umap_embedding = reducer.fit_transform(corrected)
    corrected_umap_df = pd.DataFrame(corrected_umap_embedding, columns=['x', 'y'], index=corrected.index)
    corrected_umap_df[['Batch', 'subtype']] = X_final_embed[['Batch', 'Pred_subtype']]
    corrected_umap_df.to_csv(os.path.join(result_dir, "corrected_umap_embedding.csv"))

    print("Done")

    # ================================
    # Creating K-means clustering data
    # ================================
    print("Creating K-means clustering data")
    k_mean_x = uncorrected.copy()

    n_samples = len(k_mean_x)
    max_k = min(10, max(2, n_samples // 10))
    k_values = list(range(2, max_k + 1))

    def convert_labels_to_names(kmean_labels):
        return [f"Cluster {label + 1}" for label in kmean_labels]

    if is_automatically_estimation_required == 0:
        k = num_subtype_user_defined
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(k_mean_x)
    else:
        inertias = []
        if len(k_values) < 2:
            k_values = [2]

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(k_mean_x)
            inertias.append(kmeans.inertia_)

        knee = KneeLocator(k_values, inertias, curve='convex', direction='decreasing')
        k_by_elbow = knee.knee
        suggested_k = (
            int(k_by_elbow)
            if (k_by_elbow is not None and k_by_elbow != 2 and k_by_elbow != max(k_values))
            else min(k_values)
        )

        kmeans = KMeans(n_clusters=suggested_k, random_state=42)
        labels = kmeans.fit_predict(k_mean_x)

    # Convert results to DataFrame
    kmeans_df = pd.DataFrame({
        "Cluster": convert_labels_to_names(labels)
    }, index=uncorrected_umap_df.index)

    # Save as CSV
    kmeans_df.to_csv(os.path.join(result_dir, "kmeans.csv"), index=True)
    print("Done")

    # ================================
    # Creating Nemo data
    # ================================
    print("Creating Nemo data")
    def run_nemo_r_script(script_path, auto_estimate, num_clusters, source_path, target_path, output_path,
                          working_dir=result_dir):
        args = [
            str(int(auto_estimate)),
            str(num_clusters),
            source_path,
            target_path,
            output_path
        ]
        command = ["Rscript", script_path] + args

        try:
            result = subprocess.run(
                command,
                cwd=working_dir,
                capture_output=True,
                text=True,
                check=True
            )
            print("R script output:", result.stdout)
            if result.stderr:
                print("R script warnings/errors:", result.stderr)
            return True
        except subprocess.CalledProcessError as e:
            print("R script failed:", e.stderr)
            return False

    run_nemo_r_script(
        script_path=settings.nemo_script_file,
        auto_estimate=is_automatically_estimation_required==1,
        num_clusters=num_subtype_user_defined,
        source_path=x_filename,
        target_path=target_filename,
        output_path=os.path.join(result_dir, "nemo.csv")
    )
    print("Done")

    # ================================
    # Comparing VM Models
    # ================================
    print("Comparing VM models")

    cancersubminer_result = X_final_embed.copy()

    # Create the comparison table with only the filtered rows
    # Batch == Source, subtype should be the original subtypes
    comparison_table = pd.DataFrame(index=cancersubminer_result.index, data={
        "Batch": cancersubminer_result['Batch'],
        "CancerSubminer": cancersubminer_result['Pred_subtype']
    })

    # Load NeMo results
    nemo_df = pd.read_csv(os.path.join(result_dir, "nemo.csv"), index_col=0)

    # Add clustering results to comparison table (aligning by index)
    comparison_table["NeMo"] = nemo_df["Subtype"].reindex(comparison_table.index)
    comparison_table["KMeans"] = kmeans_df["Cluster"].reindex(comparison_table.index)

    # Save comparison table
    comparison_table.to_csv(os.path.join(result_dir, "result_comparison_ml.csv"), index=True)

    print("Done")

    # ================================
    # Creating KM Plot Data
    # ================================
    if metadata_file is not None:
        print("Processing KM Plot Data...")

        # Load metadata file
        metadata_path = metadata_sample_path(user_id, project_id, metadata_file)
        metadata = pd.read_csv(metadata_path, index_col=0)

        # Convert column names to lowercase for easier searching
        metadata.columns = metadata.columns.str.lower()

        # Identify columns containing "time" and "event" or "status"
        time_col = next((col for col in metadata.columns if "time" in col), None)
        event_col = next((col for col in metadata.columns if "event" in col), None)

        # If no event column is found, check for "status"
        if not event_col and "status" in metadata.columns:
            event_col = "status"
            print("Using 'status' as the event column. Mapping values: 1 → 0, 2 → 1")
            metadata["status"] = metadata["status"].map({1: 0, 2: 1})

        if not time_col or not event_col:
            print("Error: Metadata file must contain at least one column with 'time' and one with 'event' or 'status'.")
        else:
            print(f"Using columns: {time_col} (Time), {event_col} (Event)")

            # Rename columns to OS_time and OS_event
            metadata = metadata.rename(columns={time_col: "OS_time", event_col: "OS_event"})

            # Merge metadata with data_X_embed to get Label_subtype and Batch
            km_data = metadata[['OS_time', 'OS_event']].join(
                X_final_embed[['Pred_subtype', 'Batch']].rename(columns={"Pred_subtype": "Label_subtype"}),
                how='inner'
            )
            
            # Save KM plot data for D3.js visualization
            km_plot_path = os.path.join(result_dir, "km_plot_data.csv")
            km_data.to_csv(km_plot_path, index=True)
            print("Done")

    # ================================
    # Zipping result files
    # ================================
    print("Zipping result files...")
    result_files = ["cpg_info_table.csv", "results_target_subtype.csv", "preprocessed_dataset.csv", "batch_corrected_features.csv", "result_comparison_ml.csv"]
    zip_path = os.path.join(result_dir, "results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for file in result_files:
            file_path = os.path.join(result_dir, file)
            if os.path.exists(file_path):
                zipf.write(file_path, arcname=file)

    print("Done")
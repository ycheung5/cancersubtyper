import os
import zipfile

import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from config import settings
from util.path_untils import source_sample_file_path, target_sample_file_path, job_preprocessing_path, job_result_path, \
    metadata_sample_path


def preprocess_bctypefinder(
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

def run_bctypefinder(
        user_id: int,
        project_id: int,
        job_id: int,
        model_parameters: list,
        metadata_file: str,
):
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
        def __init__(self, x_data, y_data, z_data):
            self.x_data = x_data
            self.y_data = y_data
            self.z_data = z_data

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index], self.z_data[index]

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

    result_dir = job_result_path(user_id, project_id, job_id)
    os.makedirs(result_dir, exist_ok=True)

    preprocessing_path = job_preprocessing_path(user_id, project_id, job_id)

    x_filename = os.path.join(preprocessing_path, "source_X.csv")
    y_filename = os.path.join(preprocessing_path, "source_y.csv")
    target_filename = os.path.join(preprocessing_path, "target_X.csv")

    raw_x = pd.read_csv(x_filename, index_col=0)
    raw_y = pd.read_csv(y_filename, index_col=0)

    raw_target_x = pd.read_csv(target_filename, index_col=0)

    sample_id_list = raw_x.index.tolist()
    sample_id_list.extend(raw_target_x.index.tolist())

    raw_target_domain_y = raw_target_x['domain_idx'].tolist()

    y_train = raw_y['subtype'].tolist()
    num_subtype = len(set(y_train))
    y_train = np.array(y_train)

    del raw_target_x['domain_idx']
    del raw_target_x['Batch']

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
    target_init_y = torch.randint(low=0, high=num_subtype, size=(len(target_x),))

    # domain_z : domain_subtype
    domain_z = torch.cat((y_train, target_init_y), 0)

    num_feature = len(x_train[0])
    num_train = len(x_train)
    num_test = len(raw_target_x)

    train_dataset = MyBaseDataset(x_train, y_train)
    domain_dataset = DomainDataset(domain_x, domain_y, domain_z)
    target_dataset = MyBaseDataset(target_x, target_init_y)

    batch_size = 128
    target_batch_size = 128
    test_target_batch_size = 64

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    domain_dataloader = DataLoader(domain_dataset, batch_size=batch_size, shuffle=True)
    target_dataloader = DataLoader(target_dataset, batch_size=target_batch_size, shuffle=False)

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

    class SubtypeClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            # self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(n_fe_embed2, n_c_h1),
                nn.LeakyReLU(),
                nn.Linear(n_c_h1, n_c_h2),
                nn.LeakyReLU(),
                nn.Linear(n_c_h2, num_subtype)
            )

        def forward(self, x):
            logits = self.linear_relu_stack(x)
            return logits

    feature_extract_model = FeatureExtractor().to(device)
    domain_disc_model = DomainDiscriminator().to(device)
    subtype_pred_model = SubtypeClassifier().to(device)

    c_loss = nn.CrossEntropyLoss()  # Already have softmax
    domain_loss = nn.CrossEntropyLoss()  # Already have softmax

    fe_optimizer = torch.optim.Adam(feature_extract_model.parameters(), lr=1e-4)
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

    def class_alignment_train(epoch, domain_dataloader, fe_model, fe_optimizer, c_optimizer):
        for batch, (X, y_domain, z_subtype) in enumerate(domain_dataloader):
            X, y_domain, z_subtype = X.to(device), y_domain.to(device), z_subtype.to(device)
            X = X.float()
            batch_subtype_list = z_subtype.unique()
            X_embed = fe_model(X)
            #
            align_loss = torch.zeros((1), dtype=torch.float64)
            align_loss = align_loss.to(device)
            #
            for subtype in batch_subtype_list:
                sample_idx_list = (z_subtype == subtype).nonzero(as_tuple=True)[0]
                if len(sample_idx_list) < 1:
                    continue
                # else :
                tmp_x = X_embed[sample_idx_list]
                tmp_y = y_domain[sample_idx_list]
                tmp_z = z_subtype[sample_idx_list]
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
            if align_loss == 0.0:
                continue
            align_loss = align_loss / len(batch_subtype_list)
            fe_optimizer.zero_grad()
            c_optimizer.zero_grad()
            align_loss.backward()
            fe_optimizer.step()
            c_optimizer.step()
        align_loss = align_loss.item()
        if epoch % 10 == 0:
            print(f"[CA Epoch {epoch + 1}] align loss: {align_loss:>5f}\n")

    def ssl_train_classifier(epoch, source_dataloader, target_dataloader, fe_model, c_model, c_loss, fe_optimizer,
                             c_optimizer):
        source_size = len(source_dataloader.dataset)
        target_size = len(target_dataloader.dataset)
        #
        # 1. Obtain the pseudo-label for target dataset
        #
        target_pseudo_label = torch.empty((0), dtype=torch.int64)
        target_pseudo_label = target_pseudo_label.to(device)
        #
        for batch, (target_X, target_y) in enumerate(target_dataloader):
            target_X, target_y = target_X.to(device), target_y.to(device)
            target_X = target_X.float()
            extracted_feature = fe_model(target_X)
            batch_target_pred = c_model(extracted_feature)
            batch_pseudo_label = batch_target_pred.argmax(1)
            target_pseudo_label = torch.cat((target_pseudo_label, batch_pseudo_label), 0)
            if batch == 0:
                target_loss = c_loss(batch_target_pred, target_y)
            else:
                target_loss = target_loss + c_loss(batch_target_pred, target_y)
        target_loss = target_loss / (batch + 1)
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
        return target_pseudo_label

    def adversarial_train_disc(epoch, dataloader, fe_model, d_model, domain_loss, fe_optimizer, d_optimizer):
        size = len(dataloader.dataset)
        correct = 0
        for batch, (X, y, z_subtype) in enumerate(dataloader):
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
        for batch, (X, y, z_subtype) in enumerate(dataloader):
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

    def get_embed(dataloader, fe_model, c_model):
        fe_model.eval()
        c_model.eval()
        X_embed_list = []
        y_list = []
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                X = X.float()
                X_embed = fe_model(X)
                X_embed_list.append(X_embed)
                y_list.append(y)
        X_embed_list = torch.cat(X_embed_list, 0)
        y_list = torch.cat(y_list, 0)
        return X_embed_list, y_list

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
    ssl_train_epochs = 500
    ft_epochs = 800

    # pt_epochs = 20
    # ad_train_epochs = 20
    # ssl_train_epochs = 20
    # ft_epochs = 20

    # 1. Pre-training
    for t in range(pt_epochs):
        pretrain_classifier(t, train_dataloader, feature_extract_model, subtype_pred_model, c_loss, fe_optimizer,
                            c_optimizer)

    # 2. Adversarial training
    for t in range(ad_train_epochs):
        adversarial_train_disc(t, domain_dataloader, feature_extract_model, domain_disc_model, domain_loss,
                               fe_optimizer, d_optimizer)
        adversarial_train_fe(t, domain_dataloader, feature_extract_model, domain_disc_model, domain_loss, fe_optimizer,
                             d_optimizer)

    # 3. SSL training
    for t in range(ssl_train_epochs):
        target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, feature_extract_model,
                                                   subtype_pred_model, c_loss, fe_optimizer, c_optimizer)
        target_dataset = MyBaseDataset(target_x, target_pseudo_label)
        target_dataloader = DataLoader(target_dataset, batch_size=target_batch_size)

    # 4. Fine-tuning
    for t in range(ft_epochs):
        # SSL
        target_pseudo_label = ssl_train_classifier(t, train_dataloader, target_dataloader, feature_extract_model,
                                                   subtype_pred_model, c_loss, fe_optimizer, c_optimizer)
        target_dataset = MyBaseDataset(target_x, target_pseudo_label)
        target_dataloader = DataLoader(target_dataset, batch_size=target_batch_size)

        # CA
        target_pseudo_label = target_pseudo_label.to("cpu")
        domain_z = torch.cat((y_train, target_pseudo_label), 0)
        domain_dataset = DomainDataset(domain_x, domain_y, domain_z)
        domain_dataloader = DataLoader(domain_dataset, batch_size=batch_size, shuffle=True)
        class_alignment_train(t, domain_dataloader, feature_extract_model, fe_optimizer, c_optimizer)

    domain_dataset = DomainDataset(domain_x, domain_y, domain_z)
    domain_dataloader = DataLoader(domain_dataset, batch_size=batch_size, shuffle=False)

    data_X_embed, domain_label, pred_subtype, label_subtype = get_embed_domain(domain_dataloader, feature_extract_model,
                                                                               subtype_pred_model)
    data_X_embed = data_X_embed.detach().cpu().numpy()
    domain_label = domain_label.detach().cpu().numpy()
    pred_subtype = pred_subtype.detach().cpu().numpy()
    label_subtype = label_subtype.detach().cpu().numpy()

    data_X_embed = pd.DataFrame(data_X_embed)
    data_X_embed['Batch'] = domain_label
    data_X_embed['Pred_subtype'] = pred_subtype
    data_X_embed['Label_subtype'] = label_subtype

    data_X_embed.index = sample_id_list
    domain_info = pd.read_csv(os.path.join(preprocessing_path, "batch_category_info.csv"), index_col=1)
    subtype_info = pd.read_csv(os.path.join(preprocessing_path, "subtype_category_info.csv"), index_col=1)
    domain_info = domain_info.to_dict()
    domain_info['batch'][0] = 'Source'
    subtype_info = subtype_info.to_dict()

    data_X_embed['Pred_subtype'] = data_X_embed['Pred_subtype'].replace(subtype_info['subtype'])
    data_X_embed['Label_subtype'] = data_X_embed['Label_subtype'].replace(subtype_info['subtype'])
    data_X_embed['Batch'] = data_X_embed['Batch'].replace(domain_info['batch'])

    data_X_embed.to_csv(os.path.join(result_dir, "batch_corrected_features.csv"), mode="w", index=True)

    target_pred = data_X_embed[['Batch', 'Pred_subtype']]
    target_pred = target_pred[target_pred['Batch'] != 'Source']
    target_pred.to_csv(os.path.join(result_dir, "results_target_subtype.csv"), mode="w", index=True)

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
    target_x['subtype'] = data_X_embed['Label_subtype']

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
    combined['subtype'] = data_X_embed['Label_subtype']

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

    # Generate UMAP for Corrected Data
    corrected = data_X_embed.copy().drop(columns=['Batch', 'Pred_subtype', 'Label_subtype'], errors='ignore')
    corrected_umap_embedding = reducer.fit_transform(corrected)
    corrected_umap_df = pd.DataFrame(corrected_umap_embedding, columns=['x', 'y'], index=corrected.index)
    corrected_umap_df[['Batch', 'subtype']] = data_X_embed[['Batch', 'Label_subtype']]
    corrected_umap_df.to_csv(os.path.join(result_dir, "corrected_umap_embedding.csv"))

    print("Done")

    # ================================
    # Comparing VM Models
    # ================================
    print("Comparing VM models")

    # Load training and target data
    raw_x = pd.read_csv(x_filename, index_col=0)
    raw_y = pd.read_csv(y_filename, index_col=0)
    target_raw_x = pd.read_csv(target_filename, index_col=0).drop(columns=['Batch', 'domain_idx'], errors='ignore')

    # Standardize features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(raw_x)
    y_train = raw_y['subtype']
    x_target = scaler.transform(target_raw_x)

    # Filter out rows where 'Batch' is 'Source'
    filtered_data = data_X_embed.loc[data_X_embed['Batch'] != "Source"]

    # Create the comparison table with only the filtered rows
    comparison_table = pd.DataFrame(index=filtered_data.index, data={
        "Batch": filtered_data['Batch'],
        "BCtypeFinder": filtered_data['Label_subtype']
    })

    # Define models for evaluation
    models = {
        "SVM": SVC(kernel='linear', random_state=42),
        "RF": RandomForestClassifier(random_state=42),
        "LogReg": LogisticRegression(random_state=42)
    }

    # Train and evaluate models
    for name, model in models.items():
        model.fit(x_train, y_train)
        predictions = pd.Series(model.predict(x_target), index=target_raw_x.index).replace(subtype_info['subtype'])
        comparison_table[name] = predictions

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
            km_data = metadata[['OS_time', 'OS_event']].join(data_X_embed[['Label_subtype', 'Batch']], how='inner')

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
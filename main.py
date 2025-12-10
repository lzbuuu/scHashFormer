import os
from datetime import datetime
from os.path import exists
from easydict import EasyDict
import psutil
from scanpy import AnnData
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import wandb
import gc
from config import get_config, parse_args
from data import normalize, load_dataset, MyDataset
from scRNA_workflow import *
from models import scHashFormer, LSH
from utils import setup_seed, evaluate, kmeans
import utils
from time import time


def pretraining(cfg, X, sf, raw, Y, model, device):
    print(f'Pretraining Hashformer model for {cfg.pretraining_epoch} epochs...')
    type_key = cfg['type_key']
    save_model_path = cfg.save_model_path

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)  #
    min_loss = np.inf
    best_epoch = 0
    train_dataset = Data.TensorDataset(X, sf, raw)
    train_loader = Data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    epoch = cfg.pretraining_epoch
    y_true = Y.cpu().detach().numpy()

    for epoch_id in range(2):
        start = datetime.now()
        model.train()
        train_loss = 0
        for i, (xbatch, sfbatch, rawbatch) in enumerate(train_loader):
            xbatch = xbatch.to(device)
            sfbatch = sfbatch.to(device)
            rawbatch = rawbatch.to(device)
            # _, _, meanbatch, dispbatch, pibatch = model(xbatch)
            # re_loss = model.ZINB_loss(sfbatch, rawbatch, meanbatch, dispbatch, pibatch)
            _, ebatch, meanbatch, dispbatch, pibatch = model(xbatch)
            s = torch.sign(ebatch)
            powers_of_two = 2 ** torch.arange(s.size(-1) - 1, -1, -1).to(s.device)
            bucket_index = torch.sum(s * powers_of_two, dim=-1)  # [batch_size, n_hashes] sample_id, bucket_id
            bucket_index = bucket_index.to(torch.long)
            loss = model.loss(ebatch, bucket_index, sfbatch, rawbatch, meanbatch, dispbatch, pibatch)
            
            optimizer.zero_grad()
            # re_loss.backward()
            loss.backward()
            optimizer.step()
            # train_loss += re_loss.cpu().detach().item()
            train_loss += loss.cpu().detach().item()

        train_loss = train_loss / len(train_loader)
        with torch.no_grad():
            emb, _ = model.encodeBatch(X, cfg.batch_size, device=device)
            y_pred, _ = kmeans(emb.cpu().detach().numpy(), cfg.num_classes)
            acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred, cfg.num_classes)
        

        print(f'Epoch: {epoch_id}, Train Loss: {train_loss}')
        print(f'ACC: {acc}, F1: {f1}, NMI: {nmi}, ARI: {ari}, Homo: {homo}, Comp: {comp}')
        if train_loss < min_loss:
            min_loss = train_loss
            best_epoch = epoch_id
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch_id,
            }
            path = save_model_path + '/pretrained/'
            if os.path.exists(path) is False:
                os.makedirs(path)
            torch.save(state, path + f'{cfg.dataset_name}_trained_hashmodel.pth')
        end = datetime.now()
        print(f'Epoch {epoch_id}: {end-start}s')


def fine_tuning(cfg, X, sf, raw, Y, model, hash_model, device, cell_type_map):
    # cfg.hops = 256
    type_key = cfg.type_key
    raw_adata = AnnData(X=raw.cpu().detach().numpy(), obs=pd.DataFrame({type_key: Y.cpu().detach().numpy()}))
    save_model_path = cfg.save_model_path
    path = save_model_path + '/pretrained/'
    # hash_model.load_state_dict(torch.load(path + f'{cfg.dataset_name}_trained_hashmodel10.pth')['net'])
    
    # path = save_model_path + '/fine_tuned/'
    # model.load_state_dict(torch.load(path + f'{cfg.dataset_name}_trained_scHashFormer.pth')['net'])
    start = time()

    with torch.no_grad():
        _, e = hash_model.encodeBatch(X, 128, device=device)
        s = torch.where(torch.sign(e)==-1, 0, 1).cpu()
        powers_of_two = 2 ** torch.arange(s.size(-1) - 1, -1, -1)
        bucket_index = torch.sum(s * powers_of_two, dim=-1)  # Convert binary index to decimal index [N, 1]
        index_mapping = utils.get_index_mapping(bucket_index)  # {bucket_id: decimal_index}

    end = time()
    print(f'Hashing time: {end - start} seconds')
    print(f'Memory: {psutil.Process(os.getpid()).memory_info().rss/1024/1024}MB')
    features = X
    labels = Y

    model = model.to(device)
    
    fine_dataset = Data.TensorDataset(torch.tensor(list(index_mapping.keys())))
    fine_loader = Data.DataLoader(fine_dataset, batch_size=cfg.fine_batch_size, shuffle=True)

    with torch.no_grad():
        emb = []
        # y_true = []
        index_list = []
        for batch_id, data in enumerate(fine_loader):
            index_batch = [index_mapping[i.item()] for i in data[0]]
            splits = [len(index_batch[i]) for i in range(len(index_batch))]
            index_batch = np.concatenate(index_batch, axis=0)
            x = features[index_batch].to(device)
            z = model.embedding(x, splits)
            emb.append(z.detach().cpu().numpy())
            index_list.append(index_batch)

    emb = np.concatenate(emb, axis=0)
    index = np.concatenate(index_list, axis=0)
    y_true = np.array(labels[index])
    # y_true = np.concatenate(y_true, axis=0)

    # Initializing cluster centers
    y_pred, centers = kmeans(emb, cfg.num_classes)
    acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred, cfg.num_classes)

    reversed_emb = np.zeros_like(emb)
    reversed_y_pred = np.zeros_like(y_pred)
    reversed_emb[index] = emb
    reversed_y_pred[index] = y_pred
    emb_adata = AnnData(X=reversed_emb, obs=pd.DataFrame({type_key: cell_type_map[Y], 'scHashFormer': cell_type_map[reversed_y_pred]}), raw=raw_adata)
    path = cfg.save_embedding_path
    if args.save:
        emb_adata.write_h5ad(path+f'/{cfg.dataset_name}_embedding10.h5ad')

    model.set_centers(centers)
    print(f'Initial Clustering Accuracy: {acc}, F1 Score: {f1}, NMI: {nmi}, ARI: {ari}')
    del emb, y_true, y_pred, centers
    gc.collect()

    # Start training
    print('Start training')
    # aug_features = aug_feature_shuffle(features)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    max_acc = 0
    # start = time.now()
    for epoch in range(cfg.fine_tune_epoch):
        t0 = time()
        model.train()
        train_loss = 0
        
        for batch_id, data in enumerate(fine_loader):
            index = [index_mapping[i.item()] for i in data[0]]
            splits = [len(index[i]) for i in range(len(index))]
            index = np.concatenate(index, axis=0)
            x = features[index].to(device)
            emb_batch, meanbatch, dispbatch, pibatch = model(x, splits)

            sf_batch = sf[index].to(device)
            raw_batch = raw[index].to(device)
            loss = model.loss(emb_batch, sf_batch, raw_batch, meanbatch, dispbatch, pibatch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().detach().item()
        train_loss = train_loss / len(fine_loader)
        GPU_allocated = torch.cuda.memory_allocated(device)/1024/1024
        print("GPU Memory: {} MB".format(GPU_allocated))
        with torch.no_grad():
            z_test = []
            label_list = []
            index_list = []
            for batch_id, data in enumerate(fine_loader):
                index_batch = [index_mapping[i.item()] for i in data[0]]
                splits = [len(index_batch[i]) for i in range(len(index_batch))]
                index_batch = np.concatenate(index_batch, axis=0)
                x = features[index_batch].to(device)
                z = model.embedding(x, splits)
                z_test.append(z)
                index_list.append(index_batch)
                label_list.append(Y[index_batch])

            emb = torch.cat(z_test, dim=0)
            index = np.concatenate(index_list, axis=0)
            y_true = Y[index]

            y_pred, _ = kmeans(emb, -1, model.centers.data)
            # y_pred, _ = kmeans(emb, cfg.num_classes)
        acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred, cfg.num_classes)
        results = {'ACC': acc, 'F1': f1, 'NMI': nmi, 'ARI': ari, 'Homo': homo, 'Comp': comp}
            
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, '
              f'ACC: {acc}, F1: {f1}, NMI: {nmi}, ARI: {ari}, Homo: {homo}, Comp: {comp}')
        print(f'Epoch : {epoch} training time: %d seconds.' % int(time() - t0))
        
        if max_acc < acc:
            max_acc = acc
            best_epoch = epoch
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'results': results
            }
            path = save_model_path + '/fine_tuned/'
            if os.path.exists(path) is False:
                os.makedirs(path)
            if args.save:
                torch.save(state, path + f'{cfg.dataset_name}_trained_scHashFormer10.pth')
            emb = emb.cpu().detach().numpy()
            reversed_emb = np.zeros_like(emb)
            reversed_y_pred = np.zeros_like(y_pred)
            reversed_emb[index] = emb
            reversed_y_pred[index] = y_pred
            emb_adata = AnnData(X=reversed_emb, obs=pd.DataFrame({type_key: cell_type_map[Y], 'scHashFormer': cell_type_map[reversed_y_pred]}), raw=raw_adata)
            path = cfg.save_embedding_path
            if args.save:
                emb_adata.write_h5ad(path+f'/{cfg.dataset_name}_embedding.h5ad')

    print(f'Best Epoch: {best_epoch}, Best ACC: {max_acc}')


def pretrain(cfg, X, sf, raw, Y, model, hash_model, device):
    # cfg.hops = 256
    type_key = cfg.type_key
    raw_adata = AnnData(X=raw.cpu().detach().numpy(), obs=pd.DataFrame({type_key: Y.cpu().detach().numpy()}))
    save_model_path = cfg.save_model_path
    path = save_model_path + '/pretrained/'

    start = time()
    decimal_indexes = []
    with torch.no_grad():
        for _, (x, y, _) in enumerate(test_loader):
            x = x.to(device)
            h = hash_model(x)
            h = h.view(h.size(0), cfg.n_buckets)  # [batch_size, num_buckets]
            binary_index = torch.round(h)  # Round function to get the index for the buckets
            powers_of_two = 2 ** torch.arange(binary_index.size(1) - 1, -1, -1, device=device)
            decimal_index = torch.sum(binary_index * powers_of_two, dim=1)  # Convert binary index to decimal index
            decimal_indexes.append(decimal_index.cpu())
        decimal_indexes = torch.cat(decimal_indexes, dim=0)
    index_mapping = utils.get_index_mapping(decimal_indexes)  # {bucket_id: decimal_index}
    end = time()
    print(f'Hashing time: {end - start} seconds')
    features = X
    labels = Y

    # Initializing cluster centers
    model = model.to(device)
    
    fine_dataset = Data.TensorDataset(torch.tensor(list(index_mapping.keys())))
    fine_loader = Data.DataLoader(fine_dataset, batch_size=cfg.fine_batch_size, shuffle=True)

    # Start training
    print('Start pretraining')
    # aug_features = aug_feature_shuffle(features)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    # start = time.now()
    min_loss = np.inf
    for epoch in range(1):
        t0 = time()
        model.train()
        train_loss = 0
        for batch_id, data in enumerate(fine_loader):
            index = [index_mapping[i.item()] for i in data[0]]
            splits = [len(index[i]) for i in range(len(index))]
            index = np.concatenate(index, axis=0)
            x = features[index].to(device)
            emb_batch, meanbatch, dispbatch, pibatch = model(x, splits)

            sf_batch = sf[index].to(device)
            raw_batch = raw[index].to(device)
            loss = model.pretrain_loss(sf_batch, raw_batch, meanbatch, dispbatch, pibatch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.cpu().detach().item()
        train_loss = train_loss / len(fine_loader)
        with torch.no_grad():
            emb = []
            # y_true = []
            index_list = []
            for batch_id, data in enumerate(fine_loader):
                index_batch = [index_mapping[i.item()] for i in data[0]]
                splits = [len(index_batch[i]) for i in range(len(index_batch))]
                index_batch = np.concatenate(index_batch, axis=0)
                x = features[index_batch].to(device)
                z = model.embedding(x, splits)
                emb.append(z.detach().cpu().numpy())
                index_list.append(index_batch)

        emb = np.concatenate(emb, axis=0)
        index = np.concatenate(index_list, axis=0)
        y_true = np.array(labels[index])
        y_pred, centers = kmeans(emb, cfg.num_classes)
        acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred, cfg.num_classes)

        results = {'ACC': acc, 'F1': f1, 'NMI': nmi, 'ARI': ari, 'Homo': homo, 'Comp': comp}
            
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, '
              f'ACC: {acc}, F1: {f1}, NMI: {nmi}, ARI: {ari}, Homo: {homo}, Comp: {comp}')
        print(f'Epoch : {epoch} pretraining time: %d seconds.' % int(time() - t0))
        if train_loss < min_loss:
            min_loss = train_loss
            best_epoch = epoch
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'results': results
            }
            path = save_model_path + '/pretrain/'
            if os.path.exists(path) is False:
                os.makedirs(path)
            torch.save(state, path + f'{cfg.dataset_name}_pretrained_autoencoder.pth')


if __name__ == '__main__':
    args = parse_args()
    device = args.device
    cfg_path = args.cfg_path + f'/{args.dataset}.yml'
    cfg = get_config(cfg_path)
    seed = args.seed
    for key, value in vars(args).items():
        cfg[key] = value
    if cfg.wandb:
        if not os.path.exists("./wandb/"):
            os.makedirs("./wandb")
        wandb.init(config=cfg,
                   project="SGFormer",
                   name="scHashFormer_{}".format(cfg.dataset_name),
                   dir="./wandb/",
                   job_type="training",
                   reinit=True)
    setup_seed(seed)

    X, sf, raw, adata = load_dataset(cfg['data_dir'])
    print(X.shape)
    type_key = cfg.type_key
    cell_name = np.array(adata.obs[type_key])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    adata.obs['Group'] = cell_label
    Y = torch.from_numpy(np.array(adata.obs['Group'])).to(torch.long)

    input_dim = X.shape[1]
    
    hash_model = LSH(cfg.n_buckets, cfg.n_hashes, input_dim).to(device)

    pretrain_path = cfg.save_model_path + f'/pretrained/{cfg.dataset_name}_trained_hashmodel.pth'
    if os.path.exists(pretrain_path):
        print('Load pretrained HashModel')
        hash_model.load_state_dict(torch.load(pretrain_path)['net'])
    else:
        print('Start pretraining')
        pretrain_dataset = Data.TensorDataset(X, sf, raw, Y)
        pretrain_loader = Data.DataLoader(pretrain_dataset, batch_size=cfg.batch_size, shuffle=True)
        pretraining(cfg, X, sf, raw, Y, hash_model, device)
    
    if cfg.finetune:
        print('Start fine-tuning')
        model = scHashFormer(cfg, input_dim).to(device)

        test_dataset = MyDataset(X, Y)
        test_loader = Data.DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        fine_tuning(cfg, X, sf, raw, Y, model, hash_model, device, cell_type)
        # fine_tuning(cfg, X, sf, raw, Y, model, hash_model, device, train_loader, cell_type)

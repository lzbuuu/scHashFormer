import os.path
import os
import wandb
import torch
import numpy as np
import scSimGCL
from data import load_dataset
import torch.utils.data as Data
from utils import setup_seed, loader_construction, evaluate, hop2token, clustering
import scFormer
from sklearn.cluster import KMeans
from config import config, get_config, parse_args


def pretrain(cfg, train_loader, model):
    epoch = cfg.pretraining_epoch
    lr = cfg.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_epoch = 0
    max_acc = 0
    for each_epoch in range(epoch):

        train_loss = 0
        model.train()
        for step, (data, batch_y) in enumerate(train_loader):
            batch_x = data[:, 0, :, :].squeeze().float().to(device)
            aug_batch_x = data[:, 1, :, :].squeeze().float().to(device)

            loss_cl = model.pretrain(batch_x, aug_batch_x)
            train_batch_loss = loss_cl
            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()
            train_loss += train_batch_loss.cpu().detach().item()
        train_loss = train_loss / len(train_loader)
        with torch.no_grad():
            test_loss = 0
            model.eval()
            z_test = []
            y_test = []
            for step, (data, batch_y) in enumerate(train_loader):
                batch_x = data[:, 0, :, :].squeeze().float().to(device)
                z = model.embedding(batch_x)
                z_test.append(z.cpu().detach().numpy())
                y_test.append(batch_y)
            test_loss /= len(train_loader)
            emb = np.concatenate(z_test, axis=0)
            y_true = np.concatenate(y_test, axis=0)
            clustering_model = KMeans(init="k-means++", n_clusters=cfg.num_classes, random_state=cfg.seed)
            clustering_model.fit(emb)
            y_pred = clustering_model.labels_
            model.centers.data = torch.tensor(clustering_model.cluster_centers_).to(device)
            acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred)
            results = {'ACC': acc, 'F1': f1, 'NMI': nmi, 'ARI': ari, 'Homo': homo, 'Comp': comp}

        print(f'Epoch: {each_epoch}, Train Loss: {train_loss}, '
                f'ACC: {acc}, F1: {f1}, NMI: {nmi}, ARI: {ari}, Homo: {homo}, Comp: {comp}')

        if max_acc < acc:
            max_acc = acc
            best_epoch = each_epoch
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': each_epoch,
                'results': results
            }
            path = save_model_path+f'/pretrained/{cfg.dataset_name}_'
            if os.path.exists(path) is False:
                os.makedirs(path)
            torch.save(state, path+'best_state.pth')
    print(f'Best Epoch: {best_epoch}, Best ACC: {max_acc}')
    return best_epoch, max_acc, results


def fine_tune(cfg, train_loader, model):
    model._pretraining = False
    epoch = cfg.fine_tune_epoch
    lr = cfg.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.eval()
    best_epoch = 0
    max_acc = 0

    for each_epoch in range(epoch):
        model.train()
        train_loss = 0
        for step, (data, batch_y) in enumerate(train_loader):
            batch_x = data[:, 0, :, :].squeeze().float().to(device)
            aug_batch_x = data[:, 1, :, :].squeeze().float().to(device)
            train_batch_loss = model.fine_tune(batch_x, aug_batch_x)
            optimizer.zero_grad()
            train_batch_loss.backward()
            optimizer.step()
            train_loss += train_batch_loss.cpu().detach().item()
        train_loss = train_loss / len(train_loader)
        with (torch.no_grad()):
            model.eval()
            x_test = []
            y_test = []
            for step, (data, batch_y) in enumerate(train_loader):
                batch_x = data[:, 0, :, :].squeeze().float().to(device)
                z = model.embedding(batch_x)
                x_test.append(z)
                y_test.append(batch_y.cpu().detach().numpy())

            x = torch.cat(x_test, dim=0)
            y_true = np.concatenate(y_test, axis=0)
            centers = model.centers
            y_pred, _ = clustering(x, len(centers), centers.data)
            acc, f1, nmi, ari, homo, comp = evaluate(y_true, y_pred)
            results = {'ACC': acc, 'F1': f1, 'NMI': nmi, 'ARI': ari, 'Homo': homo, 'Comp': comp}

        print(f'Epoch: {each_epoch}, Train Loss: {train_loss}, '
                f'ACC: {acc}, F1: {f1}, NMI: {nmi}, ARI: {ari}, Homo: {homo}, Comp: {comp}')
        wandb.log({'Train Loss': train_loss, 'Test ACC': acc, 'Test F1': f1, 'Test NMI': nmi, 'Test ARI': ari})

        if max_acc < acc:
            max_acc = acc
            best_epoch = each_epoch
            state = {
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': each_epoch
            }
            path = save_model_path+f'/fine_tuned/{cfg.dataset_name}_'
            if os.path.exists(path) is False:
                os.makedirs(path)
            torch.save(state, path+'best_state.pth')

    return best_epoch, max_acc, results


def test(z_test_epoch,
         y_test_epoch,
         best_epoch,
         n_clusters,
         seed):

    z_test = z_test_epoch[best_epoch]
    y_test = y_test_epoch[best_epoch]

    z_test = np.vstack(z_test)
    y_test = np.hstack(y_test)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20).fit(z_test)
    y_kmeans_test = kmeans.labels_

    acc, f1, nmi, ari, homo, comp = evaluate(y_test, y_kmeans_test)
    results = {'CA': acc, 'NMI': nmi, 'ARI':ari}

    return results


if __name__ == '__main__':
    args = parse_args()

    device = args.device
    graph_head = args.graph_head
    phi = args.phi
    gcn_dim = args.gcn_dim
    mlp_dim = args.mlp_dim
    lambda_cl = args.lambda_cl
    dropout = args.dropout
    data_path = args.data_path
    save_model_path = args.save_model_path

    cfg_path = args.cfg_path+f'/{args.dataset}.yml'
    cfg = get_config(cfg_path)
    seed = cfg.seed

    setup_seed(seed)
    features, Y = load_dataset(cfg)
    re_features, adj, graph = hop2token(features, cfg.hops, cfg.k, cfg.pe_dim)

    if not os.path.exists("./wandb/"):
        os.makedirs("./wandb")
    wandb.init(config=args,
               project="SGFormer",
               name="scFormer_{}".format('HRCA'),
               dir="./wandb/",
               job_type="training",
               reinit=True)

    train_dataset = Data.TensorDataset(re_features, torch.from_numpy(Y))
    # train_loader = torch.utils.data.DataLoader(list(range(features.shape[0])), batch_size=cfg.batch_size, shuffle=False)
    train_loader = Data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)

    input_dim = re_features.shape[-1]
    model = scFormer.Model(cfg, input_dim)
    model = model.to(device)
    # train_loader, test_loader, input_dim = loader_construction(data_path)
    pretrained_file = f'{save_model_path}/pretrained/{cfg.dataset_name}_model.pth'
    if args.mode == 'pretrain':
        print('Start pretraining...')
        pretrain(cfg, train_loader, model)
        torch.save(model.state_dict(), pretrained_file)
    elif args.mode == 'fine_tune':
        print('Start fine-tuning...')
        fine_tuned_file = f'{save_model_path}/fine_tuned/{cfg.dataset_name}_model.pth'
        model.load_state_dict(torch.load(pretrained_file))
        fine_tune(cfg, train_loader, model)
        torch.save(model.state_dict(), pretrained_file)


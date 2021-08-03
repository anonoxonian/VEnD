import torch
from argparse import ArgumentParser
from models.vend import VEND
from tqdm import tqdm
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset

GPU_ID = 0

def my_loss(z, scores):
    return (z * scores).sum()

def get_dataloader(ordered, dataset):
    # batch size has to be 1 because it's the most that the GPUs can handle
    data = torch.load("../tokd/datasets/{}_{}.pt".format(dataset, ordered))
    ids, toks = data
    ds = TensorDataset(ids, toks)
    return DataLoader(ds, batch_size=1)

def delta(entities, entities_pred):
    batch_size, num_neighbours = entities_pred.shape
    return torch.Tensor([[1 if entities[i] == entities_pred[i, j] else -1 for j in range(num_neighbours)] for i in range(batch_size)])

def modify_index(entities_pred, entities, distances, index, repulsion_rate, attraction_rate, embedding_size, positions):
    batch_size, num_neighbours = entities_pred.shape
    repulsion_coefficient = np.array([[-attraction_rate if entities[i] == entities_pred[i, j] else repulsion_rate for j in range(num_neighbours)] for i in range(batch_size)])
    repulsion_distances = repulsion_coefficient / distances # repulsion is inversely proportional to distance squared
    transformations = {}
    for i in range(batch_size):
        for j in range(num_neighbours):
            entity_id = entities_pred[i, j]
            if entity_id not in transformations.keys():
                transformations[entity_id] = (np.zeros((embedding_size,)), index.lookup(entity_id))
            entity_displacement, entity_obj = transformations[entity_id]
            entity_displacement += repulsion_distances[i, j] * (entity_obj.pos - positions[i])
            transformations[entity_id] = entity_displacement, entity_obj
    new_entity_embeddings = {}
    for k, v in transformations:
        entity_displacement, entity_obj = v
        new_pos = entity_obj.pos + entity_displacement
        new_entity_embeddings[k] = new_pos
    index.remove(entities_pred)
    ids = np.array(new_entity_embeddings.keys())
    new_pos = np.array(new_entity_embeddings.values())
    index.add(ids, new_pos)
    
def rebuild_index(index):
    index.rebuild()

def load_model(saved_model):
    embedding_dim, state_dict, index = torch.load(saved_model)
    model = VEND(embedding_dim, index, 4) # num_neighbours should be a hyperparam!!
    model.load_state_dict(state_dict)
    return model, index
    
def construct_model(embedding_dim):
    # load the default index
    index = torch.load("../models/vend/default_index_{}.pt".format(embedding_dim))
    model = VEND(embedding_dim, index, 4) # num_neighbours should be a hyperparam!!
    return model, index

def train(embedding_dim, index_period, repulsion_rate, attraction_rate, ordered, saved_model, dataset, num_epochs):
    dl = get_dataloader(ordered, dataset)
    if saved_model is not None:
        model, index = load_model(saved_model)
    else:
        model, index = construct_model(embedding_dim)
    num_samples_processed_since_rebuild = 0
    model = model.to(GPU_ID)
    optimizer = torch.optim.Adam(model.parameters()).to(GPU_ID)
    model.train()
    for epoch in range(num_epochs):
        train_iter = tqdm(dl, desc="Epoch {}, Loss: N/A".format(epoch))
        epoch_loss = 0
        samples_processed = 0
        for batch in train_iter:
            entities, inputs = batch
            inputs_gpu = {k:v.to(GPU_ID) for k,v in inputs.items()}
            optimizer.zero_grad()
            num_samples_in_batch = entities.shape[0]
            outputs = model(**inputs_gpu)
            entities_pred, positions, scores, distances = outputs["entities"], outputs["positions"], outputs["scores"], outputs["distances"]
            z = delta(entities, entities_pred).to(GPU_ID)
            loss = my_loss(z, scores)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().detach().item()
            modify_index(entities_pred, entities, distances, index, repulsion_rate, attraction_rate, positions)
            num_samples_processed_since_rebuild += num_samples_in_batch
            samples_processed += num_samples_in_batch
            train_iter.set_description(desc="Epoch {}, Loss: {}".format(epoch, epoch_loss/samples_processed))
            if num_samples_processed_since_rebuild > index_period:
                num_samples_processed_since_rebuild = 0
                rebuild_index(index)

if __name__ == "__main__":
    parser = ArgumentParser(description='Training options')
    parser.add_argument('--embedding_dim', type=int,
                        help='dimension of embedding space', required=True)
    parser.add_argument('--index_period', type=int,
                        help='num training samples between index rebuildings', required=True)
    parser.add_argument('--repulsion_rate', type=float,
                        help='repulsion rate for incorrect predictions', required=True)
    parser.add_argument('--attraction_rate', type=float,
                        help='attraction rate for correct predictions', required=True)
    parser.add_argument('--ordered', type=bool, action='store_true',
                        help='ordered data', default=False)
    parser.add_argument('--saved_model',
                        help='path to model dict and index, for loading a model and index trained on the BLINK dataset', default=None)
    parser.add_argument('--dataset', choices=["AY2", "WIKI", "CWEB", "ALL"],
                        help='dataset for training', default="BLINK")
    parser.add_argument('--num_epochs', type=int, required=True )
    args = parser.parse_args()

    # basic validation
    # if saved_model, then dataset must not be "BLINK"
    assert((args.saved_model is None) or (args.dataset != "BLINK"))
    train(args.embedding_dim, args.index_period, args.repulsion_rate, args.attraction_rate, args.ordered, args.saved_model, args.dataset, args.num_epochs)

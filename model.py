import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, mf_dim, layers):
        super(NeuMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.mf_dim = mf_dim
        self.layers = layers
        
        assert(layers[0] % 2 == 0)
        mlp_emb_dim = layers[0] // 2

        self.mf_embedding_user = nn.Embedding(num_users, mf_dim)
        self.mf_embedding_item = nn.Embedding(num_items, mf_dim)
        self.mlp_embedding_user = nn.Embedding(num_users, mlp_emb_dim)
        self.mlp_embedding_item = nn.Embedding(num_items, mlp_emb_dim)

        nn.init.normal_(self.mf_embedding_user.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mf_embedding_item.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mlp_embedding_user.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.mlp_embedding_item.weight, mean=0.0, std=0.01)
        
        num_layers = len(layers)
        self.mlp_layers = nn.ModuleList()
        for index in range(num_layers-1):
            self.mlp_layers.append(nn.Linear(layers[index], layers[index+1]))

        self.final_prediction_layer = nn.Linear(layers[num_layers-1] + mf_dim, 1)      

    def forward(self, user_input, item_input):
        mf_emb_user = self.mf_embedding_user(user_input)
        mf_emb_item = self.mf_embedding_item(item_input)
        mf_vector = mf_emb_user * mf_emb_item

        mlp_emb_user = self.mlp_embedding_user(user_input)
        mlp_emb_item = self.mlp_embedding_item(item_input)
        mlp_vector = torch.cat((mlp_emb_user, mlp_emb_item), dim=1)

        for mlp_layer in self.mlp_layers:
            mlp_vector = F.relu(mlp_layer(mlp_vector))

        prediction = torch.sigmoid(self.final_prediction_layer(torch.cat((mf_vector, mlp_vector), dim=1)))

        return prediction



import torch
from torch import nn
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,

)
from modules.ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from .conv import (
    MLP,
    DropoutIfTraining,
    MetaLayer,
    AtomEncoder,
    GINConv,        
    MLPwoLastAct,
)
import torch.nn.functional as F
from .utils import GradMultiply

_REDUCER_NAMES = {"sum": global_add_pool, "mean": global_mean_pool, "max": global_max_pool}

class GNN(nn.Module):
    def __init__(
        self,
        mlp_hidden_size: int = 512,
        mlp_layers: int = 2,
        latent_size: int = 128,
        use_layer_norm: bool = False,
        num_message_passing_steps: int = 8,
        global_reducer: str = "sum",
        node_reducer: str = "sum",
        face_reducer: str = "sum",
        dropedge_rate: float = 0.1,
        dropnode_rate: float = 0.1,
        ignore_globals: bool = True,
        use_face: bool = True,
        dropout: float = 0.1,
        dropnet: float = 0.1,
        init_face: bool = False,
        graph_pooling: str = "sum",
        use_outer: bool = False,
        residual: bool = False,
        layernorm_before: bool = False,
        parallel: bool = False,
        num_tasks: int = 1,
        layer_drop: float = 0.0,
        pooler_dropout: float = 0.0,
        encoder_dropout: float = 0.0,
        use_bn: bool = False,
        global_attn: bool = False,
        node_attn: bool = False,
        face_attn: bool = False,        
        som_mode:bool=False,
        flag: bool = False,
        ddi: bool = False,
        gradmultiply: float = -1,
        ap_hid_size: int = None,
        ap_mlp_layers: int = None,        
    ):
        super().__init__()

        self.som_mode = som_mode        
        self.encoder_edge = MLPwoLastAct(
            sum(get_bond_feature_dims()),
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
            use_bn=use_bn,
        )
        
        n_atom_dim = sum(get_atom_feature_dims()) 
        self.encoder_node = MLPwoLastAct(
            n_atom_dim,
            [mlp_hidden_size] * mlp_layers + [latent_size],
            use_layer_norm=use_layer_norm,
            use_bn=use_bn,
        )
        self.global_init = nn.Parameter(torch.zeros((1, latent_size)))
        if use_face:
            self.encoder_face = MLPwoLastAct(
                latent_size * (3 if init_face else 1),
                [mlp_hidden_size] * mlp_layers + [latent_size],
                use_layer_norm=use_layer_norm,
                use_bn=use_bn,
                layernorm_before=layernorm_before,
            )
        else:
            self.encoder_face = None

        self.gnn_layers = nn.ModuleList()
        for i in range(num_message_passing_steps):
            edge_model = DropoutIfTraining(
                p=dropedge_rate,
                submodule=MLP(
                    latent_size * (6 if use_face else 4),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            node_model = DropoutIfTraining(
                p=dropnode_rate,
                submodule=MLP(
                    latent_size * (5 if use_face else 4),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                ),
            )
            if i == num_message_passing_steps - 1:
                global_model = None
            else:

                global_model = MLP(
                    latent_size * (4 if use_face else 3),
                    [mlp_hidden_size] * mlp_layers + [latent_size],
                    use_layer_norm=use_layer_norm,
                    layernorm_before=layernorm_before,
                    dropout=encoder_dropout,
                    use_bn=use_bn,
                )
            if use_face:
                if i == num_message_passing_steps - 1:
                    face_model = None
                else:
                    face_model = MLP(
                        latent_size * 5,
                        [mlp_hidden_size] * mlp_layers + [latent_size],
                        use_layer_norm=use_layer_norm,
                        layernorm_before=layernorm_before,
                        dropout=encoder_dropout,
                        use_bn=use_bn,
                    )
            else:
                face_model = None
            sublayer = MetaLayer
            self.gnn_layers.append(
                sublayer(
                    edge_model=edge_model,
                    node_model=node_model,
                    face_model=face_model,
                    global_model=global_model,
                    aggregate_edges_for_node_fn=_REDUCER_NAMES[node_reducer],
                    aggregate_edges_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_nodes_for_globals_fn=_REDUCER_NAMES[global_reducer],
                    aggregate_edges_for_face_fn=_REDUCER_NAMES[face_reducer],
                    face_attn=face_attn,
                    node_attn=node_attn,
                    global_attn=global_attn,
                    embed_dim=latent_size,)
            )
        self.ddi = ddi
        ap_hid_size = mlp_hidden_size if ap_hid_size is None else ap_hid_size
        ap_mlp_layers = mlp_layers if ap_mlp_layers is None else ap_mlp_layers
        self.decoder = MLPwoLastAct(
            latent_size if not self.ddi else 2 * latent_size,
            [ap_hid_size] * ap_mlp_layers + [num_tasks],
            use_layer_norm=False,
            dropout=pooler_dropout,
            use_bn=use_bn,
        )

        self.use_face = use_face
        self.latent_size = latent_size
        self.aggregate_nodes_for_globals_fn = _REDUCER_NAMES[global_reducer]
        self.aggregate_edges_for_node_fn = _REDUCER_NAMES[node_reducer]
        self.aggregate_edges_for_face_fn = _REDUCER_NAMES[face_reducer]
        self.ignore_globals = ignore_globals
        self.pooling = _REDUCER_NAMES[graph_pooling]

        self.dropnet = dropnet
        self.init_face = init_face
        self.use_outer = use_outer
        self.residual = residual
        self.layer_drop = layer_drop
        self.dropout = dropout
        self.encoder_pe = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 3))
        self.reset_parameters()
        self.gradmultiply = gradmultiply

    def reset_parameters(self):
        if isinstance(self.encoder_node, nn.Linear):
            nn.init.xavier_uniform_(self.encoder_node.weight.data)
            nn.init.xavier_uniform_(self.encoder_edge.weight.data)

    def forward(self, batch, perturb=None, perturb_edge=None):
        (
            x,
            edge_index,
            edge_attr,
            node_batch,
            face_mask,
            face_index,
            num_nodes,
            num_faces,
            num_edges,
            num_graphs,
            nf_node,
            nf_face,            
        ) = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.ring_mask,
            batch.ring_index,
            batch.n_nodes,
            batch.num_rings,
            batch.n_edges,
            batch.num_graphs,
            batch.nf_node.view(-1),
            batch.nf_ring.view(-1),
            
        )
        if self.use_outer:
            face_mask = face_mask.fill_(False)
        x = one_hot_atoms(x)
        edge_attr = one_hot_bonds(edge_attr)

        graph_idx = torch.arange(num_graphs).to(x.device)
        edge_batch = torch.repeat_interleave(graph_idx, num_edges, dim=0)

        u = self.global_init.expand(num_graphs, -1)

        x = self.encoder_node(x)        
        x_encoder_node = x.clone()
        
        if perturb is not None:
            x = x + perturb
        edge_attr = self.encoder_edge(edge_attr)        
        edge_attr_encoder_edge = edge_attr.clone()

        if perturb_edge is not None:
            edge_attr = edge_attr + perturb_edge

        if self.use_face:
            face_batch = torch.repeat_interleave(graph_idx, num_faces, dim=0)
            if self.init_face:
                node_attributes = self.aggregate_edges_for_face_fn(
                    x[nf_node], nf_face, size=num_faces.sum().item()
                )
                sent_attributes = self.aggregate_edges_for_face_fn(
                    edge_attr, face_index[0], size=num_faces.sum().item()
                )
                received_attributes = self.aggregate_edges_for_face_fn(
                    edge_attr, face_index[1], size=num_faces.sum().item()
                )
                feat = torch.cat([node_attributes, sent_attributes, received_attributes], dim=1)
                feat = torch.where(face_mask.unsqueeze(1), feat.new_zeros((feat.shape[0], 1)), feat)
                face = self.encoder_face(feat)
            else:
                face = x.new_zeros((num_faces.sum().item(), self.latent_size))
                face = self.encoder_face(face)
        else:
            face = None
            face_batch = None
            face_index = None

        droplayer_probs = torch.empty(len(self.gnn_layers)).uniform_()
        for i, layer in enumerate(self.gnn_layers):            
            if self.training and self.layer_drop > 0 and droplayer_probs[i] < self.layer_drop:
                continue
            x_1, edge_attr_1, u_1, face_1 = layer(
                x,
                edge_index,
                edge_attr,
                u,
                node_batch,
                edge_batch,
                face_batch,
                face,
                face_mask,
                face_index,
                num_nodes,
                num_faces,
                num_edges,
                nf_node,
                nf_face,
                batch.batch
            )
            if self.residual:
                x = x_1
                edge_attr = edge_attr_1
                u = u_1
                if face is not None:
                    face = face_1
            else:
                x = F.dropout(x_1, p=self.dropout, training=self.training) + x
                edge_attr = (
                    F.dropout(edge_attr_1, p=self.dropout, training=self.training) + edge_attr
                )
                u = F.dropout(u_1, p=self.dropout, training=self.training) + u
                if face is not None:
                    face = F.dropout(face_1, p=self.dropout, training=self.training) + face
        if self.som_mode:
            mol_feat_atom = self.pooling(x, node_batch, size=num_graphs)
            mol_feat_bond = self.pooling(edge_attr, edge_batch, size=num_graphs)
            mol_feat_ring = self.pooling(face, face_batch, size=num_graphs)

            return mol_feat_atom, mol_feat_bond, mol_feat_ring, x, edge_attr, u, x_encoder_node, edge_attr_encoder_edge
        else:
            return self.pooling(x, node_batch, size=num_graphs)
        

    def get_last_layer(self, input):
        assert self.ddi
        return self.decoder(input)


def one_hot_atoms(atoms):
    vocab_sizes = get_atom_feature_dims()
    one_hots = []
    for i in range(atoms.shape[1]):
        one_hots.append(
            F.one_hot(atoms[:, i], num_classes=vocab_sizes[i]).to(atoms.device).to(torch.float32)
        )
    return torch.cat(one_hots, dim=1)


def one_hot_bonds(bonds):
    vocab_sizes = get_bond_feature_dims()
    one_hots = []
    for i in range(bonds.shape[1]):
        one_hots.append(
            F.one_hot(bonds[:, i], num_classes=vocab_sizes[i]).to(bonds.device).to(torch.float32)
        )
    return torch.cat(one_hots, dim=1)

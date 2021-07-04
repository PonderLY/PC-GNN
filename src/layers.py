import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable


from operator import itemgetter
import math

"""
	PC-GNN Layers
	Paper: Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection
	Modified from https://github.com/YingtongDou/CARE-GNN
"""


class InterAgg(nn.Module):

	def __init__(self, features, feature_dim, train_pos,
				 embed_dim, adj_lists, intraggs,
				 inter='GNN', cuda=True):
		"""
		Initialize the inter-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feature_dim: the input dimension
		:param embed_dim: the output dimension
		:param adj_lists: a list of adjacency lists for each single-relation graph
		:param intraggs: the intra-relation aggregators used by each single-relation graph
		:param inter: the aggregator type: 'Att', 'Weight', 'Mean', 'GNN'
		:param cuda: whether to use GPU
		"""
		super(InterAgg, self).__init__()

		self.features = features
		self.dropout = 0.6
		self.adj_lists = adj_lists
		self.intra_agg1 = intraggs[0]
		self.intra_agg2 = intraggs[1]
		self.intra_agg3 = intraggs[2]
		self.embed_dim = embed_dim
		self.feat_dim = feature_dim
		self.inter = inter
		self.cuda = cuda
		self.intra_agg1.cuda = cuda
		self.intra_agg2.cuda = cuda
		self.intra_agg3.cuda = cuda
		self.train_pos = train_pos

		# initial filtering thresholds
		self.thresholds = [0.5, 0.5, 0.5]

		# the activation function used by attention mechanism
		self.leakyrelu = nn.LeakyReLU(0.2)

		# parameter used to transform node embeddings before inter-relation aggregation
		self.weight = nn.Parameter(torch.FloatTensor(self.embed_dim, self.feat_dim))
		init.xavier_uniform_(self.weight)

		# weight parameter for each relation used by CARE-Weight
		self.alpha = nn.Parameter(torch.FloatTensor(self.embed_dim, 3))
		init.xavier_uniform_(self.alpha)

		# parameters used by attention layer
		self.a = nn.Parameter(torch.FloatTensor(2 * self.embed_dim, 1))
		init.xavier_uniform_(self.a)

		# label predictor for similarity measure
		self.label_clf = nn.Linear(self.feat_dim, 2)

		# initialize the parameter logs
		self.weights_log = []
		self.thresholds_log = [self.thresholds]
		self.relation_score_log = []

	def forward(self, nodes, labels, train_flag=True):
		"""
		:param nodes: a list of batch node ids
		:param labels: a list of batch node labels
		:param train_flag: indicates whether in training or testing mode
		:return combined: the embeddings of a batch of input node features
		:return center_scores: the label-aware scores of batch nodes
		"""

		# extract 1-hop neighbor ids from adj lists of each single-relation graph
		to_neighs = []
		for adj_list in self.adj_lists:
			to_neighs.append([set(adj_list[int(node)]) for node in nodes])

		# find unique nodes and their neighbors used in current batch
		unique_nodes = set.union(set.union(*to_neighs[0]), set.union(*to_neighs[1]),
								 set.union(*to_neighs[2], set(nodes)))

		# calculate label-aware scores
		if self.cuda:
			batch_features = self.features(torch.cuda.LongTensor(list(unique_nodes)))
		else:
			batch_features = self.features(torch.LongTensor(list(unique_nodes)))
		batch_scores = self.label_clf(batch_features)
		if self.cuda:
			pos_features = self.features(torch.cuda.LongTensor(list(self.train_pos)))
		else:
			pos_features = self.features(torch.LongTensor(list(self.train_pos)))
		pos_scores = self.label_clf(pos_features)
		id_mapping = {node_id: index for node_id, index in zip(unique_nodes, range(len(unique_nodes)))}

		# the label-aware scores for current batch of nodes
		center_scores = batch_scores[itemgetter(*nodes)(id_mapping), :]

		# get neighbor node id list for each batch node and relation
		r1_list = [list(to_neigh) for to_neigh in to_neighs[0]]
		r2_list = [list(to_neigh) for to_neigh in to_neighs[1]]
		r3_list = [list(to_neigh) for to_neigh in to_neighs[2]]

		# assign label-aware scores to neighbor nodes for each batch node and relation
		r1_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r1_list]
		r2_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r2_list]
		r3_scores = [batch_scores[itemgetter(*to_neigh)(id_mapping), :].view(-1, 2) for to_neigh in r3_list]

		# count the number of neighbors kept for aggregation for each batch node and relation
		r1_sample_num_list = [math.ceil(len(neighs) * self.thresholds[0]) for neighs in r1_list]
		r2_sample_num_list = [math.ceil(len(neighs) * self.thresholds[1]) for neighs in r2_list]
		r3_sample_num_list = [math.ceil(len(neighs) * self.thresholds[2]) for neighs in r3_list]

		# intra-aggregation steps for each relation
		# Eq. (8) in the paper
		r1_feats, r1_scores = self.intra_agg1.forward(nodes, labels, r1_list, center_scores, r1_scores, pos_scores, r1_sample_num_list)
		r2_feats, r2_scores = self.intra_agg2.forward(nodes, labels, r2_list, center_scores, r2_scores, pos_scores, r2_sample_num_list)
		r3_feats, r3_scores = self.intra_agg3.forward(nodes, labels, r3_list, center_scores, r3_scores, pos_scores, r3_sample_num_list)

		# concat the intra-aggregated embeddings from each relation
		neigh_feats = torch.cat((r1_feats, r2_feats, r3_feats), dim=0)

		# get features or embeddings for batch nodes
		if self.cuda and isinstance(nodes, list):
			index = torch.LongTensor(nodes).cuda()
		else:
			index = torch.LongTensor(nodes)
		self_feats = self.features(index)

		# number of nodes in a batch
		n = len(nodes)

		# inter-relation aggregation steps
		
		if self.inter == 'Att':
			# 1) Att Inter-relation Aggregator
			combined, attention = att_inter_agg(len(self.adj_lists), self.leakyrelu, self_feats, neigh_feats, self.embed_dim,
												self.weight, self.a, n, self.dropout, self.training, self.cuda)
		elif self.inter == 'Weight':
			# 2) Weight Inter-relation Aggregator
			combined = weight_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, self.alpha, n, self.cuda)
			gem_weights = F.softmax(torch.sum(self.alpha, dim=0), dim=0).tolist()
			if train_flag:
				print(f'Weights: {gem_weights}')
		elif self.inter == 'Mean':
			# 3) Mean Inter-relation Aggregator
			combined = mean_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, n, self.cuda)
		elif self.inter == 'GNN':
			# 4) GNN Inter-relation Aggregator
			combined = threshold_inter_agg(len(self.adj_lists), self_feats, neigh_feats, self.embed_dim, self.weight, self.thresholds, n, self.cuda)

		return combined, center_scores


class IntraAgg(nn.Module):

	def __init__(self, features, feat_dim, train_pos, rho, cuda=False):
		"""
		Initialize the intra-relation aggregator
		:param features: the input node features or embeddings for all nodes
		:param feat_dim: the input dimension
		:param cuda: whether to use GPU
		"""
		super(IntraAgg, self).__init__()

		self.features = features
		self.cuda = cuda
		self.feat_dim = feat_dim
		self.train_pos = train_pos
		self.rho = rho

	def forward(self, nodes, batch_labels, to_neighs_list, batch_scores, neigh_scores, pos_scores, sample_list):
		"""
		Code partially from https://github.com/williamleif/graphsage-simple/
		:param nodes: list of nodes in a batch
		:param to_neighs_list: neighbor node id list for each batch node in one relation
		:param batch_scores: the label-aware scores of batch nodes
		:param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
		:param sample_list: the number of neighbors kept for each batch node in one relation
		:return to_feats: the aggregated embeddings of batch nodes neighbors in one relation
		:return samp_scores: the average neighbor distances for each relation after filtering
		"""

		# filer neighbors under given relation
		samp_neighs, samp_scores = choose_step_neighs(batch_scores, batch_labels, neigh_scores, to_neighs_list, pos_scores, self.train_pos, sample_list, self.rho)

		# find the unique nodes among batch nodes and the filtered neighbors
		unique_nodes_list = list(set.union(*samp_neighs))
		unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}

		# intra-relation aggregation only with sampled neighbors
		mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
		column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
		row_indices = [i for i in range(len(samp_neighs)) for _ in range(len(samp_neighs[i]))]
		mask[row_indices, column_indices] = 1
		if self.cuda:
			mask = mask.cuda()
		num_neigh = mask.sum(1, keepdim=True)
		mask = mask.div(num_neigh)
		if self.cuda:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
		else:
			embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
		to_feats = mask.mm(embed_matrix)
		to_feats = F.relu(to_feats)
		return to_feats, samp_scores


def choose_step_neighs(center_scores, center_labels, neigh_scores, neighs_list, minor_scores, minor_list, sample_list, sample_rate):
    """
    Filter neighbors according label predictor result with adaptive thresholds
    :param center_scores: the label-aware scores of batch nodes
    :param neigh_scores: the label-aware scores 1-hop neighbors each batch node in one relation
    :param neighs_list: neighbor node id list for each batch node in one relation
    :param sample_list: the number of neighbors kept for each batch node in one relation
	:para sample_rate: the ratio of the oversample neighbors for the minority class
    :return samp_neighs: the neighbor indices and neighbor simi scores
    :return samp_score_diff: the average neighbor distances for each relation after filtering
    """
    samp_neighs = []
    samp_score_diff = []
    for idx, center_score in enumerate(center_scores):
        center_score = center_scores[idx][0]
        neigh_score = neigh_scores[idx][:, 0].view(-1, 1)
        center_score_neigh = center_score.repeat(neigh_score.size()[0], 1)
        neighs_indices = neighs_list[idx]
        num_sample = sample_list[idx]

        # compute the L1-distance of batch nodes and their neighbors
        
        score_diff_neigh = torch.abs(center_score_neigh - neigh_score).squeeze()
        sorted_score_diff_neigh, sorted_neigh_indices = torch.sort(score_diff_neigh, dim=0, descending=False)
        selected_neigh_indices = sorted_neigh_indices.tolist()

        # top-p sampling according to distance ranking and thresholds
        
        if len(neigh_scores[idx]) > num_sample + 1:
            selected_neighs = [neighs_indices[n] for n in selected_neigh_indices[:num_sample]]
            selected_score_diff = sorted_score_diff_neigh.tolist()[:num_sample]
        else:
            selected_neighs = neighs_indices
            selected_score_diff = score_diff_neigh.tolist()
            if isinstance(selected_score_diff, float):
                selected_score_diff = [selected_score_diff]

        if center_labels[idx] == 1:
            num_oversample = int(num_sample * sample_rate)
            center_score_minor = center_score.repeat(minor_scores.size()[0], 1)
            score_diff_minor = torch.abs(center_score_minor - minor_scores[:, 0].view(-1, 1)).squeeze()
            sorted_score_diff_minor, sorted_minor_indices = torch.sort(score_diff_minor, dim=0, descending=False)
            selected_minor_indices = sorted_minor_indices.tolist()
            selected_neighs.extend([minor_list[n] for n in selected_minor_indices[:num_oversample]])
            selected_score_diff.extend(sorted_score_diff_minor.tolist()[:num_oversample])

        samp_neighs.append(set(selected_neighs))
        samp_score_diff.append(selected_score_diff)

    return samp_neighs, samp_score_diff


def mean_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, n, cuda):
	"""
	Mean inter-relation aggregator
	:param num_relations: number of relations in the graph
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	:return: inter-relation aggregated node embeddings
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = weight.mm(self_feats.t())
	neigh_h = weight.mm(neigh_feats.t())

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(embed_dim, n)).cuda()
	else:
		aggregated = torch.zeros(size=(embed_dim, n))

	# sum neighbor embeddings together
	for r in range(num_relations):
		aggregated += neigh_h[:, r * n:(r + 1) * n]

	# sum aggregated neighbor embedding and batch node embedding
	# take the average of embedding and feed them to activation function
	combined = F.relu((center_h + aggregated) / 4.0)

	return combined


def weight_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, alpha, n, cuda):
	"""
	Weight inter-relation aggregator
	Reference: https://arxiv.org/abs/2002.12307
	:param num_relations: number of relations in the graph
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param alpha: weight parameter for each relation used by CARE-Weight
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	:return: inter-relation aggregated node embeddings
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = weight.mm(self_feats.t())
	neigh_h = weight.mm(neigh_feats.t())

	# compute relation weights using softmax
	w = F.softmax(alpha, dim=0)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(embed_dim, n)).cuda()
	else:
		aggregated = torch.zeros(size=(embed_dim, n))

	# add weighted neighbor embeddings in each relation together
	for r in range(num_relations):
		aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1, n), neigh_h[:, r * n:(r + 1) * n])

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu(center_h + aggregated)

	return combined


def att_inter_agg(num_relations, att_layer, self_feats, neigh_feats, embed_dim, weight, a, n, dropout, training, cuda):
	"""
	Attention-based inter-relation aggregator
	Reference: https://github.com/Diego999/pyGAT
	:param num_relations: num_relations: number of relations in the graph
	:param att_layer: the activation function used by the attention layer
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param a: parameters used by attention layer
	:param n: number of nodes in a batch
	:param dropout: dropout for attention layer
	:param training: a flag indicating whether in the training or testing mode
	:param cuda: whether use GPU
	:return combined: inter-relation aggregated node embeddings
	:return att: the attention weights for each relation
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = self_feats.mm(weight.t())
	neigh_h = neigh_feats.mm(weight.t())

	# compute attention weights
	combined = torch.cat((center_h.repeat(3, 1), neigh_h), dim=1)
	e = att_layer(combined.mm(a))
	attention = torch.cat((e[0:n, :], e[n:2 * n, :], e[2 * n:3 * n, :]), dim=1)
	ori_attention = F.softmax(attention, dim=1)
	attention = F.dropout(ori_attention, dropout, training=training)

	# initialize the final neighbor embedding
	if cuda:
		aggregated = torch.zeros(size=(n, embed_dim)).cuda()
	else:
		aggregated = torch.zeros(size=(n, embed_dim))

	# add neighbor embeddings in each relation together with attention weights
	for r in range(num_relations):
		aggregated += torch.mul(attention[:, r].unsqueeze(1).repeat(1, embed_dim), neigh_h[r * n:(r + 1) * n, :])

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu((center_h + aggregated).t())

	# extract the attention weights
	att = F.softmax(torch.sum(ori_attention, dim=0), dim=0)

	return combined, att


def threshold_inter_agg(num_relations, self_feats, neigh_feats, embed_dim, weight, threshold, n, cuda):
	"""
	GNN inter-relation aggregator
	
	:param num_relations: number of relations in the graph
	:param self_feats: batch nodes features or embeddings
	:param neigh_feats: intra-relation aggregated neighbor embeddings for each relation
	:param embed_dim: the dimension of output embedding
	:param weight: parameter used to transform node embeddings before inter-relation aggregation
	:param threshold: the neighbor filtering thresholds used as aggregating weights
	:param n: number of nodes in a batch
	:param cuda: whether use GPU
	:return: inter-relation aggregated node embeddings
	"""

	# transform batch node embedding and neighbor embedding in each relation with weight parameter
	center_h = weight.mm(self_feats.t())
	neigh_h = weight.mm(neigh_feats.t())

	if cuda:
		# use thresholds as aggregating weights
		w = torch.FloatTensor(threshold).repeat(weight.size(0), 1).cuda()

		# initialize the final neighbor embedding
		aggregated = torch.zeros(size=(embed_dim, n)).cuda()
	else:
		w = torch.FloatTensor(threshold).repeat(weight.size(0), 1)
		aggregated = torch.zeros(size=(embed_dim, n))

	# add weighted neighbor embeddings in each relation together
	for r in range(num_relations):
		aggregated += torch.mul(w[:, r].unsqueeze(1).repeat(1, n), neigh_h[:, r * n:(r + 1) * n])

	# sum aggregated neighbor embedding and batch node embedding
	# feed them to activation function
	combined = F.relu(center_h + aggregated)

	return combined

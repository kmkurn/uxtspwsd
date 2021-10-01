# Copyright (c) 2021 Kemal Kurniawan

from typing import Dict, List, Optional, Set, Tuple
import warnings

from torch import BoolTensor, LongTensor, Tensor
from torch_struct import DependencyCRF, LinearChainCRF
from torch_struct.deptree import convert, unconvert
import numpy
import torch

from matrix_tree import compute_log_partitions, compute_marginals


class DepTreeCRF:
    """Dependency tree CRF.

    This CRF defines a (conditional) probability distribution over labeled dependency
    trees. A labeled dependency tree is represented as a sequence of head positions
    and a sequence of dependency types for the corresponding arc. The first position
    in the sequence (position 0) is assumed to be the tree's root.

    Args:
        scores: Tensor of shape (B, N, N, L) containing scores of all labeled
            head-dependent arcs.
        mask: Boolean tensor of shape (B, N) indicating valid positions.
        projective: Whether to operate in the space of projective trees.
        multiroot: Whether to consider multi-root case, where the (symbolic) root can have
            more than one child.

    Note:
        B = batch size, N = sequence length, L = number of dependency labels/types.
    """

    ROOT = 0

    def __init__(
        self,
        scores: Tensor,
        mask: Optional[BoolTensor] = None,
        projective: bool = False,
        multiroot: bool = True,
    ) -> None:
        assert scores.dim() == 4
        bsz, slen = scores.shape[:2]
        assert scores.size(2) == slen
        assert mask is None or mask.shape == (bsz, slen)

        if mask is None:
            mask = scores.new_full([bsz, slen], 1).bool()  # type: ignore

        self.scores = scores
        self.mask = mask
        self.proj = projective
        self.multiroot = multiroot

    def log_probs(
        self, heads: LongTensor, types: LongTensor, score_only: bool = False
    ) -> Tensor:
        """Compute the log probability of a labeled dependency tree.

        Args:
            heads: Tensor of shape (B, N) containing the index/position of the head of
                each word.
            types: Tensor of shape (B, N) containing the dependency types for the
                corresponding head-dependent relation.
            score_only: Whether to compute only the score of the tree. Useful for training
                with cross-entropy loss.

        Returns:
            1-D tensor of length B containing the log probabilities.
        """
        assert heads.dim() == 2
        assert types.shape == heads.shape
        assert self.mask is not None

        scores = self.scores
        bsz, slen, _, n_types = self.scores.shape

        # broadcast over types
        heads = heads.unsqueeze(2).expand(bsz, slen, n_types)  # type: ignore
        # shape: (bsz, slen, n_types)
        scores = scores.gather(1, heads.unsqueeze(1)).squeeze(1)
        # shape: (bsz, slen)
        scores = scores.gather(2, types.unsqueeze(2)).squeeze(2)
        # mask scores from invalid dependents
        scores = scores.masked_fill(~self.mask, 0)
        # mask scores of root as dependents
        scores = scores.masked_fill(torch.arange(slen).to(scores.device) == self.ROOT, 0)

        return scores.sum(dim=1) - (0 if score_only else self.log_partitions())

    def argmax(self) -> Tuple[LongTensor, LongTensor]:
        """Compute the most probable labeled dependency tree.

        Returns:
            - Tensor of shape (B, N) containing the head positions of the best tree.
            - Tensor of shape (B, N) containing the dependency types for the
              corresponding head-dependent relation.
        """
        assert self.mask is not None

        # each shape: (bsz, slen, slen)
        scores, best_types = self.scores.max(dim=3)
        lengths = self.mask.long().sum(dim=1)

        if self.proj:
            crf = DependencyCRF(unconvert(scores), lengths - 1, multiroot=self.multiroot)
            # shape: (bsz, slen)
            _, pred_heads = convert(crf.argmax).max(dim=1)
            pred_heads[:, self.ROOT] = self.ROOT
        else:
            if not self.multiroot:
                warnings.warn(
                    "argmax for non-projective is still multiroot although multiroot=False"
                )
            # shape: (bsz, slen)
            pred_heads = find_mst(scores, lengths.tolist())

        # shape: (bsz, slen)
        pred_types = best_types.gather(1, pred_heads.unsqueeze(1)).squeeze(1)

        return pred_heads, pred_types  # type: ignore

    def log_partitions(self) -> Tensor:
        """Compute the log partition function.

        Returns:
            1-D tensor of length B containing the log partition functions.
        """
        assert self.mask is not None

        if self.proj:
            lengths = self.mask.long().sum(dim=1)
            crf = DependencyCRF(unconvert(self.scores), lengths - 1, multiroot=self.multiroot)
            return crf.partition

        return compute_log_partitions(self.scores, self.mask, self.multiroot)

    def marginals(self) -> Tensor:
        """Compute the arc marginal probabilities.

        Returns:
            Tensor of shape (B, N, N, L) containing the arc marginal probabilities.
        """
        assert self.mask is not None

        if self.proj:
            lengths = self.mask.long().sum(dim=1)
            crf = DependencyCRF(unconvert(self.scores), lengths - 1, multiroot=self.multiroot)
            margs = convert(crf.marginals)

            # marginals of incoming arcs to root are zero
            margs[:, :, self.ROOT] = 0
            # marginals of self-loops are zero
            self_loop_mask = torch.eye(margs.size(1)).to(margs.device).unsqueeze(2).bool()
            margs = margs.masked_fill(self_loop_mask, 0)

            return margs

        return compute_marginals(self.scores, self.mask, self.multiroot)


class LinearCRF:
    def __init__(self, scores: Tensor, mask: Optional[BoolTensor] = None) -> None:
        """Linear-chain CRF.

        This CRF defines a (conditional) probability over chains/sequences of tags.

        Args:
            scores: Tensor of shape (B, N, T, T) containing the score of two consecutive
                tags at every position in the sequence (dim=2 is next tag).
            mask: Boolean tensor of shape (B, N) indicating valid positions.
        """
        bsz, slen = scores.shape[:2]
        if mask is None:
            mask = scores.new_full([bsz, slen], 1).bool()  # type: ignore
        assert mask is not None
        lengths = mask.type_as(scores).sum(dim=1)
        self._crf = LinearChainCRF(scores, lengths=lengths + 1)
        self._mask = mask

    def log_probs(self, tags: Tensor) -> Tensor:
        """Compute the log probability of a tag sequence.

        Args:
            tags: Tensor of shape (B, N+1) containing the tag indices for each position
                in the sequence.

        Returns:
            1-D tensor of length B containing the log probabilities.
        """
        n_tags = self._crf.event_shape[1]
        return self._crf.log_prob(self._crf.to_event(tags, n_tags, self._crf.lengths.long()))

    def log_partitions(self) -> Tensor:
        """Compute the log partition function.

        Returns:
            1-D tensor of length B containing the log partition functions.
        """
        return self._crf.partition

    def argmax(self) -> LongTensor:
        """Compute the most probable tag sequence.

        Returns:
            Tensor of shape (B, N+1) containing the best tag indices for each position.
        """
        amax = self._crf.argmax
        bsz, slen, n_next_tags, n_tags = amax.shape
        assert n_next_tags == n_tags

        lengths = self._mask.long().sum(dim=1)
        if (lengths != slen).any():
            raise NotImplementedError("argmax is only implemented for same-length sequences")

        return self._crf.from_event(amax)[0].to(amax.device)

    def marginals(self) -> Tensor:
        """Compute the consecutive tag pair marginal probabilities.

        Returns:
            Tensor of shape (B, N, T, T) containing the marginal probabilities of each possible
                consecutive tag pairs at each position.
        """
        return self._crf.marginals


def find_mst(scores: Tensor, slens: Optional[List[int]] = None, root: int = 0) -> LongTensor:
    """Find maximum spanning tree with Tarjan's implementation of Edmond's algorithm.

    Args:
        scores: Tensor of shape (B, N, N) containing the scores of all possible arcs.
        slens: List of sequence lengths.
        root: Index/position of the root.

    Returns:
        Tensor of shape (B, N) containing the head positions of the maximum tree.
    """
    bsz, maxlen, _ = scores.shape
    heads = scores.new_zeros((bsz, maxlen)).long()

    for b in range(bsz):
        slen = maxlen if slens is None else slens[b]
        hs, _ = decode_mst(scores[b].cpu().numpy(), slen, has_labels=False)
        heads[b] = torch.from_numpy(hs).to(heads)

    heads[:, root] = root

    return heads  # type: ignore


### From https://raw.githubusercontent.com/allenai/allennlp/master/allennlp/nn/chu_liu_edmonds.py
def decode_mst(
    energy: numpy.ndarray, length: int, has_labels: bool = True
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Note: Counter to typical intuition, this function decodes the _maximum_
    spanning tree.

    Decode the optimal MST tree with the Chu-Liu-Edmonds algorithm for
    maximum spanning arborescences on graphs.

    # Parameters

    energy : `numpy.ndarray`, required.
        A tensor with shape (num_labels, timesteps, timesteps)
        containing the energy of each edge. If has_labels is `False`,
        the tensor should have shape (timesteps, timesteps) instead.
    length : `int`, required.
        The length of this sequence, as the energy may have come
        from a padded batch.
    has_labels : `bool`, optional, (default = `True`)
        Whether the graph has labels or not.
    """
    if has_labels and energy.ndim != 3:
        raise ValueError("The dimension of the energy array is not equal to 3.")
    elif not has_labels and energy.ndim != 2:
        raise ValueError("The dimension of the energy array is not equal to 2.")
    input_shape = energy.shape
    max_length = input_shape[-1]

    # Our energy matrix might have been batched -
    # here we clip it to contain only non padded tokens.
    if has_labels:
        energy = energy[:, :length, :length]
        # get best label for each edge.
        label_id_matrix = energy.argmax(axis=0)
        energy = energy.max(axis=0)
    else:
        energy = energy[:length, :length]
        label_id_matrix = None
    # get original score matrix
    original_score_matrix = energy
    # initialize score matrix to original score matrix
    score_matrix = numpy.array(original_score_matrix, copy=True)

    old_input = numpy.zeros([length, length], dtype=numpy.int32)
    old_output = numpy.zeros([length, length], dtype=numpy.int32)
    current_nodes = [True for _ in range(length)]
    representatives: List[Set[int]] = []

    for node1 in range(length):
        original_score_matrix[node1, node1] = 0.0
        score_matrix[node1, node1] = 0.0
        representatives.append({node1})

        for node2 in range(node1 + 1, length):
            old_input[node1, node2] = node1
            old_output[node1, node2] = node2

            old_input[node2, node1] = node2
            old_output[node2, node1] = node1

    final_edges: Dict[int, int] = {}

    # The main algorithm operates inplace.
    chu_liu_edmonds(
        length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives
    )

    heads = numpy.zeros([max_length], numpy.int32)
    if has_labels:
        head_type = numpy.ones([max_length], numpy.int32)
    else:
        head_type = None

    for child, parent in final_edges.items():
        heads[child] = parent
        if has_labels:
            head_type[child] = label_id_matrix[parent, child]

    return heads, head_type


### From https://raw.githubusercontent.com/allenai/allennlp/master/allennlp/nn/chu_liu_edmonds.py
def chu_liu_edmonds(
    length: int,
    score_matrix: numpy.ndarray,
    current_nodes: List[bool],
    final_edges: Dict[int, int],
    old_input: numpy.ndarray,
    old_output: numpy.ndarray,
    representatives: List[Set[int]],
):
    """
    Applies the chu-liu-edmonds algorithm recursively
    to a graph with edge weights defined by score_matrix.

    Note that this function operates in place, so variables
    will be modified.

    # Parameters

    length : `int`, required.
        The number of nodes.
    score_matrix : `numpy.ndarray`, required.
        The score matrix representing the scores for pairs
        of nodes.
    current_nodes : `List[bool]`, required.
        The nodes which are representatives in the graph.
        A representative at it's most basic represents a node,
        but as the algorithm progresses, individual nodes will
        represent collapsed cycles in the graph.
    final_edges : `Dict[int, int]`, required.
        An empty dictionary which will be populated with the
        nodes which are connected in the maximum spanning tree.
    old_input : `numpy.ndarray`, required.
    old_output : `numpy.ndarray`, required.
    representatives : `List[Set[int]]`, required.
        A list containing the nodes that a particular node
        is representing at this iteration in the graph.

    # Returns

    Nothing - all variables are modified in place.

    """
    # Set the initial graph to be the greedy best one.
    parents = [-1]
    for node1 in range(1, length):
        parents.append(0)
        if current_nodes[node1]:
            max_score = score_matrix[0, node1]
            for node2 in range(1, length):
                if node2 == node1 or not current_nodes[node2]:
                    continue

                new_score = score_matrix[node2, node1]
                if new_score > max_score:
                    max_score = new_score
                    parents[node1] = node2

    # Check if this solution has a cycle.
    has_cycle, cycle = _find_cycle(parents, length, current_nodes)
    # If there are no cycles, find all edges and return.
    if not has_cycle:
        final_edges[0] = -1
        for node in range(1, length):
            if not current_nodes[node]:
                continue

            parent = old_input[parents[node], node]
            child = old_output[parents[node], node]
            final_edges[child] = parent
        return

    # Otherwise, we have a cycle so we need to remove an edge.
    # From here until the recursive call is the contraction stage of the algorithm.
    cycle_weight = 0.0
    # Find the weight of the cycle.
    index = 0
    for node in cycle:
        index += 1
        cycle_weight += score_matrix[parents[node], node]

    # For each node in the graph, find the maximum weight incoming
    # and outgoing edge into the cycle.
    cycle_representative = cycle[0]
    for node in range(length):
        if not current_nodes[node] or node in cycle:
            continue

        in_edge_weight = float("-inf")
        in_edge = -1
        out_edge_weight = float("-inf")
        out_edge = -1

        for node_in_cycle in cycle:
            if score_matrix[node_in_cycle, node] > in_edge_weight:
                in_edge_weight = score_matrix[node_in_cycle, node]
                in_edge = node_in_cycle

            # Add the new edge score to the cycle weight
            # and subtract the edge we're considering removing.
            score = (
                cycle_weight
                + score_matrix[node, node_in_cycle]
                - score_matrix[parents[node_in_cycle], node_in_cycle]
            )

            if score > out_edge_weight:
                out_edge_weight = score
                out_edge = node_in_cycle

        score_matrix[cycle_representative, node] = in_edge_weight
        old_input[cycle_representative, node] = old_input[in_edge, node]
        old_output[cycle_representative, node] = old_output[in_edge, node]

        score_matrix[node, cycle_representative] = out_edge_weight
        old_output[node, cycle_representative] = old_output[node, out_edge]
        old_input[node, cycle_representative] = old_input[node, out_edge]

    # For the next recursive iteration, we want to consider the cycle as a
    # single node. Here we collapse the cycle into the first node in the
    # cycle (first node is arbitrary), set all the other nodes not be
    # considered in the next iteration. We also keep track of which
    # representatives we are considering this iteration because we need
    # them below to check if we're done.
    considered_representatives: List[Set[int]] = []
    for i, node_in_cycle in enumerate(cycle):
        considered_representatives.append(set())
        if i > 0:
            # We need to consider at least one
            # node in the cycle, arbitrarily choose
            # the first.
            current_nodes[node_in_cycle] = False

        for node in representatives[node_in_cycle]:
            considered_representatives[i].add(node)
            if i > 0:
                representatives[cycle_representative].add(node)

    chu_liu_edmonds(
        length, score_matrix, current_nodes, final_edges, old_input, old_output, representatives
    )

    # Expansion stage.
    # check each node in cycle, if one of its representatives
    # is a key in the final_edges, it is the one we need.
    found = False
    key_node = -1
    for i, node in enumerate(cycle):
        for cycle_rep in considered_representatives[i]:
            if cycle_rep in final_edges:
                key_node = node
                found = True
                break
        if found:
            break

    previous = parents[key_node]
    while previous != key_node:
        child = old_output[parents[previous], previous]
        parent = old_input[parents[previous], previous]
        final_edges[child] = parent
        previous = parents[previous]


### From https://raw.githubusercontent.com/allenai/allennlp/master/allennlp/nn/chu_liu_edmonds.py
def _find_cycle(
    parents: List[int], length: int, current_nodes: List[bool]
) -> Tuple[bool, List[int]]:

    added = [False for _ in range(length)]
    added[0] = True
    cycle = set()
    has_cycle = False
    for i in range(1, length):
        if has_cycle:
            break
        # don't redo nodes we've already
        # visited or aren't considering.
        if added[i] or not current_nodes[i]:
            continue
        # Initialize a new possible cycle.
        this_cycle = set()
        this_cycle.add(i)
        added[i] = True
        has_cycle = True
        next_node = i
        while parents[next_node] not in this_cycle:
            next_node = parents[next_node]
            # If we see a node we've already processed,
            # we can stop, because the node we are
            # processing would have been in that cycle.
            if added[next_node]:
                has_cycle = False
                break
            added[next_node] = True
            this_cycle.add(next_node)

        if has_cycle:
            original = next_node
            cycle.add(original)
            next_node = parents[original]
            while next_node != original:
                cycle.add(next_node)
                next_node = parents[next_node]
            break

    return has_cycle, list(cycle)

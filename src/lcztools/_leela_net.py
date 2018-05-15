# -*- coding: utf-8 -*-
import numpy as np

from collections import OrderedDict

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

class LeelaNet:
    def __init__(self, model):
        self.model = model

    def _legal_policy(self, full_policy, leela_board=None, legal_moves=None):
        """A policy dict mapping legal moves to selection probabilities."""
        if legal_moves is None:
            legal_moves = leela_board.generate_legal_moves()
        # Knight promotions are represented without a suffix in leela-chess
        legal_uci = [m.uci().rstrip('n') for m in legal_moves]
        if legal_uci:
            legal_indexes = leela_board.lcz_uci_to_idx(legal_uci)
            softmaxed = _softmax(full_policy[legal_indexes])
            policy_legal = OrderedDict(sorted(zip(legal_uci, softmaxed),
                                        key = lambda mp: (mp[1], mp[0]),
                                        reverse=True))
        else:
            policy_legal = OrderedDict()
        return policy_legal

    def evaluate(self, leela_board):
        features = leela_board.features()
        policy, value = self.model(features)
        if not isinstance(policy, np.ndarray):
            # Assume it's a torch tensor
            policy = policy.cpu().numpy()
            value = value.cpu().numpy()
        policy, value = policy[0], value[0][0]
        policy_legal = self._legal_policy(policy, leela_board=leela_board)
        value = value / 2 + 0.5
        return policy_legal, value

    def evaluate_batch(self, leela_boards):
        """Evaluate a batch of boards.

        The boards are each evaluated only once and in order so `leela_boards`
        may be a generator.
        """
        features = []
        legal_moves_list = []
        for leela_board in leela_boards:
            features.append(leela_board.features())
            legal_moves_list.append(leela_board.generate_legal_moves())
        features = np.array(features)
        policies, values = self.model(features)
        if not isinstance(policies, np.ndarray):
            # Assume it's a torch tensor
            policies = policies.cpu().numpy()
            values = values.cpu().numpy()
        values = values.squeeze(-1) / 2 + 0.5
        return [
            (self._legal_policy(policy, legal_moves=legal_moves), value)
            for policy, legal_moves, value
            in zip(policies, legal_moves_list, values)]


def load_network(backend, filename):
    backends = ('tensorflow', 'pytorch', 'pytorch_orig', 'pytorch_cuda')
    if backend not in backends:
        raise Exception("Supported backends are {}".format(backends))
    kwargs = {}
    if backend == 'tensorflow':
        from lcztools._leela_tf_net import LeelaLoader
    elif backend == 'pytorch':
        from lcztools._leela_torch_eval_net import LeelaLoader
    elif backend == 'pytorch_orig':
        from lcztools._leela_torch_net import LeelaLoader
    elif backend == 'pytorch_cuda':
        from lcztools._leela_torch_eval_net import LeelaLoader
        kwargs['cuda'] = True
    return LeelaNet(LeelaLoader.from_weights_file(filename, **kwargs))

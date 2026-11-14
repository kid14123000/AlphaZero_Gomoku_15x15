# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""

import numpy as np
import copy


def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5.0, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.last_root_q = 0.0      # 每次 get_move_probs 后更新（value 软标签用）
        # ── 根先验 Dirichlet 噪声（AGZ 论文式，2026-11-14 起）──────────────
        # 上游 junxiaosong 把噪声加在"落子抽样"上：(1-ε)p + ε·Dir，树里的根先验不加噪。
        # 后果: 搜索从不探索 → 访问分布很尖；而落子可能被噪声拽到另一个点 ⇒
        #       同一局面「标签说走 A、实际走了 B」，A 的信息全废，策略头目标比搜索实际相信的更自信。
        # 现在按论文改成: 每手把噪声注入**根子节点的 _P**，之后 N 次模拟共享（与 v24 第 326-337 行同口径）。
        self.root_noise_applied = 0     # 注入次数（行为指纹用）
        self.last_root_noise_stats = None

    def _add_root_dirichlet(self, epsilon, alpha):
        """把 Dirichlet 噪声混进【根节点各子节点的先验 P】，返回是否注入成功。

        与 AGZ/v24 一致: P ← (1-ε)·P + ε·Dir(α)。只在根的子节点上做一次，
        后续 N 次模拟都在这份加过噪的先验下搜索 ⇒ 访问分布本身变宽，
        训练标签（访问数）与落子因此同源。
        """
        if not self._root._children:
            return False                # 根还没展开（首次 playout 前），调用方稍后重试
        acts = list(self._root._children.keys())
        k = len(acts)
        noise = np.random.dirichlet([alpha] * k)
        for i, a in enumerate(acts):
            node = self._root._children[a]
            node._P = (1.0 - epsilon) * node._P + epsilon * float(noise[i])
        ps = np.array([n._P for n in self._root._children.values()],
                      dtype=np.float64)
        self.last_root_noise_stats = {
            'k': k, 'sum': float(ps.sum()), 'max': float(ps.max()),
            'top1_share': float(ps.max() / max(ps.sum(), 1e-12)),
        }
        self.root_noise_applied += 1
        return True


    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        action_probs, leaf_value = self._policy(state)
        # Check for end of game.
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == -1:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == state.get_current_player() else -1.0
                )

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3, root_noise=None):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        root_noise: None 或 (epsilon, alpha)。给定时把 Dirichlet 噪声注入根子节点
            先验（只注入一次，之后全部模拟共享）—— AGZ/v24 口径。
            注入时机: 根已有子节点就在第一次模拟之前，否则在第一次模拟把根
            展开之后立刻注入（新树的首手必然走这条分支）。
        """
        _noised = False
        for n in range(self._n_playout):
            if root_noise is not None and not _noised and self._root._children:
                _noised = self._add_root_dirichlet(*root_noise)
            state_copy = state.clone()
            self._playout(state_copy)
        # 兜底: n_playout 很小或首轮展开发生在最后一步时，补注入一次
        if root_noise is not None and not _noised and self._root._children:
            self._add_root_dirichlet(*root_noise)

        # 根节点的搜索价值（**根节点该走方视角**）。
        # 供 value 头的软标签使用：搜索结果比网络单次前向可靠，
        # 拿它当目标相当于「用搜索蒸馏网络」。
        #
        # 🚨 必须取负号：_playout 末尾是 update_recursive(-leaf_value)，
        #    因为子节点 Q 的语义是「落子方（父节点玩家）视角」——select() 里
        #    父节点要最大化子节点 Q，所以子节点存的是对父有利的程度。
        #    根节点没有父节点，却吃同一个 -leaf_value，于是根 _Q = -(该走方视角)。
        #    不取负会把标签符号弄反：实测 batch600 第 17 手黑已必胜，_Q=-0.9878。
        self.last_root_q = (-float(self._root._Q)
                            if self._root._n_visits > 0 else 0.0)

        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        v = np.array(visits, dtype=np.float64)
        act_probs = softmax(1.0/temp * np.log(v + 1e-10))

        # 🚨 训练标签必须和落子温度解耦：
        #    落子可以用 temp 调度（前几手 1.0 探索，之后 0.01 近似 argmax），
        #    但存进训练数据的 π 永远是「访问数归一化」，不含温度。
        #    否则 temp=0.01 时标签退化成 one-hot，policy 头就学不到
        #    「哪些点也不错」这种软信息，只剩「只有一个点是对的」的硬标签。
        #    temp=1.0 时 softmax(log(visits)) 恰好等于 visits/sum，所以这样写
        #    对原有行为零影响（向后兼容）。
        self.last_label = v / max(v.sum(), 1e-12)

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, policy_value_function,
                 c_puct=5.0, n_playout=2000, is_selfplay=0,
                 dirichlet_eps=0.25, dirichlet_alpha=0.3, root_noise=False):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)
        self._is_selfplay = is_selfplay
        # 自我对弈探索噪声（AlphaZero 原论文 ε=0.25, α=0.3）。
        # 🚨 ε=0.25 对 19×19 围棋（200+ 手）合适，但对 9×9 五子棋（均手 20）
        #    意味着每局平均 5 手由纯随机噪声决定。实测 665 手里 28.0% 的落子
        #    网络先验 <1%，与 ε=0.25 吻合 —— 短局里这么多废棋足以直接决定胜负。
        #    实测把 ε 降到 0.10 后探索手比例降到 ~11%。
        self._dirichlet_eps = dirichlet_eps
        self._dirichlet_alpha = dirichlet_alpha
        # root_noise=True: 噪声注入根先验（AGZ 论文式 / v24 口径）
        # root_noise=False: 旧行为，噪声只混进"落子抽样"分布
        self._root_noise = bool(root_noise)
        # 不对称搜索预算（课程学习用）：{1: 黑方sims, 2: 白方sims}；None = 统一 n_playout
        self.sims_by_player = None

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=0):
        # 按执色切换搜索预算（每局开始时由 worker 设置 self.sims_by_player）
        if self.sims_by_player:
            _ns = self.sims_by_player.get(board.current_player)
            if _ns and int(_ns) != self.mcts._n_playout:
                self.mcts._n_playout = int(_ns)
        sensible_moves = board.availables
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(board.width*board.height)
        if len(sensible_moves) > 0:
            # root_noise=True → 噪声进根先验（搜索会去探索，标签=访问数，与落子同源）
            _rn = ((self._dirichlet_eps, self._dirichlet_alpha)
                   if (self._is_selfplay and self._root_noise) else None)
            acts, probs = self.mcts.get_move_probs(board, temp, root_noise=_rn)
            # 训练标签用不含温度的访问分布（last_label），落子用带温度的 probs。
            # 温度只影响「怎么选」，不该影响「学什么」。
            move_probs[list(acts)] = getattr(self.mcts, 'last_label', probs)
            if self._is_selfplay:
                if self._root_noise:
                    # 噪声已经进根先验（每手一次，N 次模拟共享）⇒ 落子只按
                    # 温度采样 probs，绝不再叠一次噪声（否则等于双重加噪）。
                    move = np.random.choice(acts, p=probs)
                else:
                    # 旧行为: 噪声只混进落子抽样，树里不加噪
                    _eps = getattr(self, '_dirichlet_eps', 0.25)
                    _alpha = getattr(self, '_dirichlet_alpha', 0.3)
                    _noise = np.random.dirichlet(_alpha * np.ones(len(probs)))
                    move = np.random.choice(
                        acts,
                        p=(1.0 - _eps) * probs + _eps * _noise
                    )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
#                location = board.move_to_location(move)
#                print("AI move: %d,%d\n" % (location[0], location[1]))

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            print("WARNING: the board is full")

    def __str__(self):
        return "MCTS {}".format(self.player)

import numpy as np

# Maximum number of legal moves in any chess position (theoretical max is 218).
# Sparse policy storage uses this as the fixed width of the index/value arrays.
_MAX_LEGAL = 218


class _SumTree:
    """Binary sum-tree for O(log N) priority-weighted sampling.

    Leaves are stored at indices [capacity, 2*capacity).
    Internal nodes store partial sums; tree[1] is the total.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def update(self, leaf_idx, priority):
        idx = leaf_idx + self.capacity
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx > 1:
            idx >>= 1
            self.tree[idx] += delta

    def total(self):
        return self.tree[1]

    def sample(self, value):
        """Return 0-based leaf index for the given cumulative-priority value."""
        # Clamp to avoid floating-point overshoot landing on an empty leaf
        value = min(float(value), self.tree[1] * (1.0 - 1e-8))
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

    def get_priority(self, leaf_idx):
        return self.tree[leaf_idx + self.capacity]

    def _rebuild(self):
        """Recompute all internal nodes from leaves — vectorized O(capacity) numpy ops."""
        t   = self.tree
        cap = self.capacity
        lvl = int(np.log2(cap)) if cap > 1 else 0
        while lvl >= 1:
            start = 1 << lvl                          # first node at this depth
            end   = min(1 << (lvl + 1), cap)          # one past the last node
            t[start:end] = t[2*start:2*end:2] + t[2*start + 1:2*end:2]
            lvl -= 1
        # Root node (index 1) is never touched by the loop above — update it explicitly.
        # Without this, total() returns 0 after every grow(), breaking PER sampling.
        t[1] = t[2] + t[3]


class PrioritizedReplayBuffer:
    """Replay buffer with Prioritized Experience Replay (PER).

    Samples are drawn proportional to |TD error|^alpha.
    Importance-sampling weights (exponent beta) correct for the sampling bias.

    Parameters
    ----------
    alpha      : prioritization exponent  (0 = uniform, 1 = full priority). Default 0.6.
    beta_start : initial IS exponent, annealed toward 1 by the training loop. Default 0.4.
    epsilon    : small floor added to every priority so nothing is never sampled. Default 1e-6.
    """

    def __init__(
        self,
        capacity,
        state_shape=(20, 8, 8),
        action_size=4672,
        alpha=0.3,
        beta_start=0.4,
        epsilon=1e-6,
    ):
        self.capacity    = capacity
        self.state_shape = state_shape
        self.action_size = action_size
        self.alpha       = alpha
        self.beta        = beta_start
        self.epsilon     = epsilon

        # Pre-allocated data arrays.
        # Policies are stored sparsely as (index, value) pairs — one row per position,
        # _MAX_LEGAL columns wide.  Padding slots have value 0.0 and index 0.
        # This cuts policy memory from ~17 GB/1M entries to ~1 GB/1M entries.
        self.states          = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.policy_indices  = np.zeros((capacity, _MAX_LEGAL),   dtype=np.uint16)
        self.policy_values   = np.zeros((capacity, _MAX_LEGAL),   dtype=np.float32)
        self.values          = np.zeros((capacity, 1),            dtype=np.float32)

        self._tree = _SumTree(capacity)

        self.index = 0
        self.size  = 0

    # ── Writing ───────────────────────────────────────────────────────────────

    def add(self, state, policy, value):
        self.states[self.index] = state
        # Pack dense policy → sparse (nonzero entries only, zero-padded to _MAX_LEGAL)
        nz = np.nonzero(policy)[0]
        k  = min(len(nz), _MAX_LEGAL)
        self.policy_indices[self.index, :k]  = nz[:k]
        self.policy_values[self.index,  :k]  = policy[nz[:k]]
        # Clear any stale data beyond k from a previous circular-buffer occupant
        if k < _MAX_LEGAL:
            self.policy_indices[self.index, k:] = 0
            self.policy_values[self.index,  k:] = 0.0
        self.values[self.index] = value
        # New entries get priority 1.0 so they are sampled at least once,
        # but do NOT inherit the historically-maximum priority — that would
        # cause every new batch of positions (especially after a buffer grow)
        # to dominate sampling until their errors are first measured.
        self._tree.update(self.index, 1.0)
        self.index = (self.index + 1) % self.capacity
        self.size  = min(self.size + 1, self.capacity)

    def add_batch(self, states, policies, values):
        for s, p, v in zip(states, policies, values):
            self.add(s, p, v)

    # ── Sparse policy helpers ─────────────────────────────────────────────────

    def _unpack_policies(self, indices):
        """Reconstruct dense (batch, action_size) policy matrix from sparse storage."""
        dense    = np.zeros((len(indices), self.action_size), dtype=np.float32)
        idx_rows = self.policy_indices[indices]   # (batch, _MAX_LEGAL)
        val_rows = self.policy_values[indices]    # (batch, _MAX_LEGAL)
        rows, cols = np.where(val_rows > 0)
        dense[rows, idx_rows[rows, cols]] = val_rows[rows, cols]
        return dense

    # ── Sampling ──────────────────────────────────────────────────────────────

    def sample(self, batch_size):
        """Return a prioritized batch.

        Returns
        -------
        states, policies, values : numpy arrays
        indices                  : int64 array of buffer indices (needed for update_priorities)
        is_weights               : float32 array of importance-sampling weights in [0, 1]
        """
        total    = self._tree.total()
        segment  = total / batch_size

        indices    = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            value        = np.random.uniform(segment * i, segment * (i + 1))
            idx          = self._tree.sample(value)
            indices[i]   = idx
            priorities[i] = self._tree.get_priority(idx)

        # IS weights: w_i = (N * p_i / total)^{-beta}, normalised by max weight
        probs      = np.maximum(priorities / (total + 1e-10), 1e-10)
        is_weights = (self.size * probs) ** (-self.beta)
        is_weights /= is_weights.max()   # rescale so max weight = 1

        return (
            self.states[indices],
            self._unpack_policies(indices),
            self.values[indices],
            indices,
            is_weights.astype(np.float32),
        )

    # ── Priority update ───────────────────────────────────────────────────────

    def update_priorities(self, indices, td_errors):
        """Update leaf priorities after computing per-sample TD errors."""
        for idx, err in zip(indices, td_errors):
            priority = (abs(float(err)) + self.epsilon) ** self.alpha
            self._tree.update(int(idx), priority)

    # ── Beta annealing ────────────────────────────────────────────────────────

    def set_beta(self, beta):
        self.beta = float(beta)

    # ── Checkpointing helpers ─────────────────────────────────────────────────

    def get_tree_snapshot(self):
        """Return tree_array for saving."""
        return self._tree.tree.copy()

    def restore_tree(self, tree_array):
        """Restore priority tree from a saved snapshot."""
        self._tree.tree[:] = tree_array

    def restore_uniform(self):
        """Assign priority 1.0 to all live entries (fallback for old checkpoints)."""
        for i in range(self.size):
            self._tree.update(i, 1.0)

    # ── Dynamic capacity growth ───────────────────────────────────────────────

    def grow(self, new_capacity):
        """Expand the buffer to new_capacity, preserving all data and priorities in-place.

        Existing entries stay in their same slot indices; new slots are empty (priority 0).
        This is O(new_capacity) due to the SumTree rebuild but is vectorized via numpy.
        """
        if new_capacity <= self.capacity:
            return

        old_cap = self.capacity

        # ── Grow data arrays ──────────────────────────────────────────────────
        new_states         = np.zeros((new_capacity, *self.state_shape), dtype=np.float32)
        new_policy_indices = np.zeros((new_capacity, _MAX_LEGAL),        dtype=np.uint16)
        new_policy_values  = np.zeros((new_capacity, _MAX_LEGAL),        dtype=np.float32)
        new_values         = np.zeros((new_capacity, 1),                 dtype=np.float32)
        new_states[:old_cap]         = self.states
        new_policy_indices[:old_cap] = self.policy_indices
        new_policy_values[:old_cap]  = self.policy_values
        new_values[:old_cap]         = self.values
        self.states         = new_states
        self.policy_indices = new_policy_indices
        self.policy_values  = new_policy_values
        self.values         = new_values

        # ── Rebuild SumTree at new capacity ───────────────────────────────────
        # Copy old leaf priorities (slots 0..old_cap-1) into the new tree's leaf layer.
        old_leaves = self._tree.tree[old_cap:2 * old_cap].copy()
        new_tree   = _SumTree(new_capacity)
        new_tree.tree[new_capacity:new_capacity + old_cap] = old_leaves
        new_tree._rebuild()
        self._tree = new_tree

        self.capacity = new_capacity
        # self.index and self.size are unchanged — the circular buffer cursor is still valid

    # ── Misc ──────────────────────────────────────────────────────────────────

    def __len__(self):
        return self.size


# Backwards-compatible alias so any code that still imports ReplayBuffer keeps working
ReplayBuffer = PrioritizedReplayBuffer

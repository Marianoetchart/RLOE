from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Deque, List, Tuple, Callable
import operator

import numpy as np
import random
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_n_step_info(n_step_buffer: Deque,
                    gamma: float
                    ) -> Tuple[np.int64, np.ndarray, bool]:
    """Return n step reward, next state, and done."""
    # info of the last transition
    reward, next_state, done = n_step_buffer[-1][-3:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_s, d = transition[-3:]

        reward = r + gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done


def get_n_step_info_from_demo(demo: List,
                              n_step: int,
                              gamma: float
                              ) -> Tuple[List, List]:
    """Return 1 step and n step demos."""
    assert n_step > 1

    demos_1_step = list()
    demos_n_step = list()
    n_step_buffer: Deque = deque(maxlen=n_step)

    for transition in demo:
        n_step_buffer.append(transition)

        if len(n_step_buffer) == n_step:
            # add a single step transition
            demos_1_step.append(n_step_buffer[0])

            # add a multi step transition
            curr_state, action = n_step_buffer[0][:2]
            reward, next_state, done = get_n_step_info(n_step_buffer, gamma)
            transition = (curr_state, action, reward, next_state, done)
            demos_n_step.append(transition)

    return demos_1_step, demos_n_step


class BaseBuffer(ABC):
    """Abstract Buffer used for replay buffer."""

    @abstractmethod
    def add(self, transition: Tuple[Any, ...]) \
            -> Tuple[Any, ...]:
        pass

    @abstractmethod
    def sample(self) \
            -> Tuple[np.ndarray, ...]:
        pass

    @abstractmethod
    def __len__(self) \
            -> int:
        pass


class ReplayBuffer(BaseBuffer):
    """Fixed-size buffer to store experience tuples.

    Attributes:
        obs_buf (np.ndarray): observations
        acts_buf (np.ndarray): actions
        rews_buf (np.ndarray): rewards
        next_obs_buf (np.ndarray): next observations
        done_buf (np.ndarray): dones
        n_step_buffer (deque): recent n transitions
        n_step (int): step size for n-step transition
        gamma (float): discount factor
        max_len (int): size of buffers
        batch_size (int): batch size for training
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(self,
                 max_len: int,
                 batch_size: int,
                 gamma: float = 0.99,
                 n_step: int = 1,
                 demo: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = None,
                 ):
        """Initialize a ReplayBuffer object.

        Args:
            max_len (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            gamma (float): discount factor
            n_step (int): step size for n-step transition
            demo (list): transitions of human play
        """
        assert 0 < batch_size <= max_len
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= max_len

        self.obs_buf: np.ndarray = None
        self.acts_buf: np.ndarray = None
        self.rews_buf: np.ndarray = None
        self.next_obs_buf: np.ndarray = None
        self.done_buf: np.ndarray = None

        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.max_len = max_len
        self.batch_size = batch_size
        self.demo_size = len(demo) if demo else 0
        self.demo = demo
        self.length = 0
        self.idx = self.demo_size

        # demo may have empty tuple list [()]
        if self.demo and self.demo[0]:
            self.max_len += self.demo_size
            self.length += self.demo_size
            for idx, d in enumerate(self.demo):
                state, action, reward, next_state, done = d
                if idx == 0:
                    action = (
                        np.array(action).astype(np.int64)
                        if isinstance(action, int)
                        else action
                    )
                    self._initialize_buffers(state, action)
                self.obs_buf[idx] = state
                self.acts_buf[idx] = np.array(action)
                self.rews_buf[idx] = reward
                self.next_obs_buf[idx] = next_state
                self.done_buf[idx] = done

    def add(self,
            transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
            ) -> Tuple[Any, ...]:
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        assert len(transition) == 5, "Inappropriate transition size"
        assert isinstance(transition[0], np.ndarray)
        assert isinstance(transition[1], np.ndarray)

        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0:
            state, action = transition[:2]
            self._initialize_buffers(state, action)

        # add a multi step transition
        reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]

        self.obs_buf[self.idx] = curr_state
        self.acts_buf[self.idx] = action
        self.rews_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_state
        self.done_buf[self.idx] = done

        self.idx += 1
        self.idx = self.demo_size if self.idx % self.max_len == 0 else self.idx
        self.length = min(self.length + 1, self.max_len)

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def extend(self,
               transitions: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]
               ):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(transition)

    def sample(self,
               indices: List[int] = None
               ) -> Tuple[np.ndarray, ...]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        states = self.obs_buf[indices]
        actions = self.acts_buf[indices]
        rewards = self.rews_buf[indices].reshape(-1, 1)
        next_states = self.next_obs_buf[indices]
        dones = self.done_buf[indices].reshape(-1, 1)

        return states, actions, rewards, next_states, dones

    def _initialize_buffers(self,
                            state: np.ndarray,
                            action: np.ndarray) -> None:
        """Initialze buffers for state, action, resward, next_state, done."""
        # In case action of demo is not np.ndarray
        self.obs_buf = np.zeros([self.max_len] + list(state.shape), dtype=state.dtype)
        self.acts_buf = np.zeros(
            [self.max_len] + list(action.shape), dtype=action.dtype
        )
        self.rews_buf = np.zeros([self.max_len], dtype=float)
        self.next_obs_buf = np.zeros(
            [self.max_len] + list(state.shape), dtype=state.dtype
        )
        self.done_buf = np.zeros([self.max_len], dtype=float)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.length


class BufferWrapper(BaseBuffer):
    """Abstract BufferWrapper used for buffer wrapper.

    Attributes:
        buffer (Buffer): Hold replay buffer as am attribute
    """

    def __init__(self,
                 base_buffer: BaseBuffer):
        """Initialize a ReplayBuffer object.

        Args:
            base_buffer (int): ReplayBuffer which should be hold
        """
        self.buffer = base_buffer

    def add(self,
            transition: Tuple[Any, ...]
            ) -> Tuple[Any, ...]:
        return self.buffer.add(transition)

    def sample(self) -> Tuple[np.ndarray, ...]:
        return self.buffer.sample()

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.buffer)


class PrioritizedBufferWrapper(BufferWrapper):
    """Prioritized Experience Replay wrapper for Buffer.


    Refer to OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py

    Attributes:
        buffer (Buffer): Hold replay buffer as an attribute
        alpha (float): alpha parameter for prioritized replay buffer
        epsilon_d (float): small positive constants to add to the priorities
        tree_idx (int): next index of tree
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight
        _max_priority (float): max priority
    """

    def __init__(self,
                 base_buffer: BaseBuffer,
                 alpha: float = 0.6,
                 epsilon_d: float = 1.0
                 ):
        """Initialize.

        Args:
            base_buffer (Buffer): ReplayBuffer which should be hold
            alpha (float): alpha parameter for prioritized replay buffer
            epsilon_d (float): small positive constants to add to the priorities

        """
        BufferWrapper.__init__(self, base_buffer)
        assert alpha >= 0
        self.alpha = alpha
        self.epsilon_d = epsilon_d
        self.tree_idx = 0

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.buffer.max_len:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self._max_priority = 1.0

        # for init priority of demo
        self.tree_idx = self.buffer.demo_size
        for i in range(self.buffer.demo_size):
            self.sum_tree[i] = self._max_priority ** self.alpha
            self.min_tree[i] = self._max_priority ** self.alpha

    def add(
        self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
    ) -> Tuple[Any, ...]:
        """Add experience and priority."""
        n_step_transition = self.buffer.add(transition)
        if n_step_transition:
            self.sum_tree[self.tree_idx] = self._max_priority ** self.alpha
            self.min_tree[self.tree_idx] = self._max_priority ** self.alpha

            self.tree_idx += 1
            if self.tree_idx % self.buffer.max_len == 0:
                self.tree_idx = self.buffer.demo_size

        return n_step_transition

    def _sample_proportional(self,
                             batch_size: int
                             ) -> list:
        """Sample indices based on proportional."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self.buffer))
        segment = p_total / batch_size

        i = 0
        while len(indices) < batch_size:
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            if idx > len(self.buffer):
                print(
                    f"[WARNING] Index for sampling is out of range: {len(self.buffer)} < {idx}"
                )
                continue
            indices.append(idx)
            i += 1
        return indices

    def sample(self,
               beta: float = 0.4
               ) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of experiences."""
        assert len(self.buffer) >= self.buffer.batch_size
        assert beta > 0

        indices = self._sample_proportional(self.buffer.batch_size)

        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self.buffer)) ** (-beta)

        # calculate weights
        weights_, eps_d = [], []
        for i in indices:
            eps_d.append(self.epsilon_d if i < self.buffer.demo_size else 0.0)
            p_sample = self.sum_tree[i] / self.sum_tree.sum()
            weight = (p_sample * len(self.buffer)) ** (-beta)
            weights_.append(weight / max_weight)

        weights = np.array(weights_)
        eps_d = np.array(eps_d)
        experiences = self.buffer.sample(indices)

        return experiences + (weights, indices, eps_d)

    def update_priorities(self,
                          indices: list,
                          priorities: np.ndarray
                          ):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self.buffer)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)


class SegmentTree:
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    Attributes:
        capacity (int)
        tree (list)
        operation (function)

    """

    def __init__(self,
                 capacity: int,
                 operation: Callable,
                 init_value: float
                 ):
        """Initialize.

        Args:
            capacity (int)
            operation (function)
            init_value (float)

        """
        assert (
            capacity > 0 and capacity & (capacity - 1) == 0
        ), "capacity must be positive and a power of 2."
        self.capacity = capacity
        self.tree = [init_value for _ in range(2 * capacity)]
        self.operation = operation

    def _operate_helper(self,
                        start: int,
                        end: int,
                        node: int,
                        node_start: int,
                        node_end: int
                        ) -> float:
        """Returns result of operation in segment."""
        if start == node_start and end == node_end:
            return self.tree[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._operate_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._operate_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self.operation(
                    self._operate_helper(start, mid, 2 * node, node_start, mid),
                    self._operate_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end),
                )

    def operate(self,
                start: int = 0,
                end: int = 0
                ) -> float:
        """Returns result of applying `self.operation`."""
        if end <= 0:
            end += self.capacity
        end -= 1

        return self._operate_helper(start, end, 1, 0, self.capacity - 1)

    def __setitem__(self,
                    idx: int,
                    val: float
                    ):
        """Set value in tree."""
        idx += self.capacity
        self.tree[idx] = val

        idx //= 2
        while idx >= 1:
            self.tree[idx] = self.operation(self.tree[2 * idx], self.tree[2 * idx + 1])
            idx //= 2

    def __getitem__(self,
                    idx: int
                    ) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]


class SumSegmentTree(SegmentTree):
    """ Create SumSegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialize.

        Args:
            capacity (int)

        """
        super(SumSegmentTree, self).__init__(
            capacity=capacity, operation=operator.add, init_value=0.0
        )

    def sum(self,
            start: int = 0,
            end: int = 0
            ) -> float:
        """Returns arr[start] + ... + arr[end]."""
        return super(SumSegmentTree, self).operate(start, end)

    def retrieve(self,
                 upperbound: float
                 ) -> int:
        """Find the highest index `i` about upper bound in the tree."""
        # TODO: Check assert case and fix bug
        assert 0 <= upperbound <= self.sum() + 1e-5, "upperbound: {}".format(upperbound)

        idx = 1

        while idx < self.capacity:  # while non-leaf
            left = 2 * idx
            right = left + 1
            if self.tree[left] > upperbound:
                idx = 2 * idx
            else:
                upperbound -= self.tree[left]
                idx = right
        return idx - self.capacity


class MinSegmentTree(SegmentTree):
    """ Create SegmentTree.

    Taken from OpenAI baselines github repository:
    https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

    """

    def __init__(self, capacity: int):
        """Initialize.

        Args:
            capacity (int)

        """
        super(MinSegmentTree, self).__init__(
            capacity=capacity, operation=min, init_value=float("inf")
        )

    def min(self,
            start: int = 0,
            end: int = 0
            ) -> float:
        """Returns min(arr[start], ...,  arr[end])."""
        return super(MinSegmentTree, self).operate(start, end)

import numpy as np
from src.hnsw import VectorStore, Node
import uuid


def test_calc_dist():
    dim = 10
    vec_num = 20
    query = np.random.rand(dim)
    vectors = np.random.rand(vec_num, dim)
    assert isinstance(VectorStore.calc_dist(query, vectors), np.ndarray)
    assert VectorStore.calc_dist(query, vectors).shape[0] == vec_num


def test_random_level():
    generator = np.random.default_rng(12345)
    level_1 = VectorStore._random_level([0.999, 0.0009, 0.00009, 0.000009], generator)
    assert isinstance(level_1, int)


def test_compute_probabilities():
    node_1 = Node(Id=uuid.uuid4(), vector=[1, 2, 3], level=3)
    node_2 = Node(Id=uuid.uuid4(), vector=[1, 2, 3], level=3, neighbors=[[node_1]])
    store = VectorStore(32, [node_1, node_2])
    store.compute_probabilities()
    assert isinstance(store.assigned_probabilities, list)
    assert isinstance(store.level_num, int)
    assert store.level_num == 6
    assert store.assigned_probabilities == [0.96875, 0.030273437499999986, 0.0009460449218749991,
                                            2.956390380859371e-05, 9.23871994018553e-07, 2.887099981307982e-08]


def test_initialize():
    node_1 = Node(Id=uuid.uuid4(), vector=[0.1, 0.2, 0.3], level=3)
    node_2 = Node(Id=uuid.uuid4(), vector=[1.0, 2.0, 3.0], level=3)
    store = VectorStore(32, [node_1, node_2])
    store._initialize()
    assert len(store.store) == len(store.assigned_probabilities) + 1 == store.level_num + 1
    assert store.entry_node.level == store.level_num + 1
    assert store.entry_node == store.store[-1][0]
    assert store.initials[0].neighbors[0][0].neighbors[0][0] == store.initials[0]
    assert len(store.initials[1].neighbors) == store.level_num

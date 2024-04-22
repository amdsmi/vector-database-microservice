from typing import List, Optional
import numpy as np
from pydantic import BaseModel
import uuid


class Node(BaseModel):
    Id: uuid.UUID
    vector: list
    level: Optional[int] = None
    pyload: Optional[dict] = None
    neighbors: List[List['Node']] = []


class VectorStore:

    def __init__(self, max_neighbors: int, initial_nodes: List[Node]):
        self.initials: List[Node] = initial_nodes
        self.max_neighbors: int = max_neighbors
        self.entry_node: Node | None = None
        self.vector_size: int | None = None
        self.assigned_probabilities: list | None = None
        self.generator = np.random.default_rng(12345)
        self.store: List[List[Node]] = list()
        self.level_num: int | None = None

    def _initialize(self):
        self.compute_probabilities()
        self.put_first_nodes()

    def start(self):
        self._initialize()

    def stop(self):
        pass

    def put_first_nodes(self):
        self.initials[0].level = self.level_num
        self.initials[1].level = self.level_num - 1
        for level in range(self.level_num - 1):
            self.store.append([self.initials[0], self.initials[1]])
            self.initials[0].neighbors.append([self.initials[1]])
            self.initials[1].neighbors.append([self.initials[0]])

        self.store.append([self.initials[0]])
        self.vector_size = len(self.initials[0].vector)
        self.entry_node = self.initials[0]
        self.assigned_probabilities = self.assigned_probabilities[:-1]
        self.level_num -= 1

    def compute_probabilities(self):
        m_l = 1 / np.log(self.max_neighbors)
        level = 0
        assigned_probabilities = []
        while True:
            proba = np.exp(-level / m_l) * (1 - np.exp(-1 / m_l))
            if proba < 1e-9:
                break
            assigned_probabilities.append(proba)
            level += 1
        self.assigned_probabilities = assigned_probabilities
        self.level_num = len(assigned_probabilities)

    @staticmethod
    def _random_level(assigned_probabilities: List[float], generator: np.random.default_rng) -> int:
        num = generator.uniform()
        for level in range(len(assigned_probabilities)):
            if num < assigned_probabilities[level]:
                return level
            num -= assigned_probabilities[level]
        return len(assigned_probabilities) - 1

    def insert_node(self, point: Node):
        print('store len', [len(s) for s in self.store])
        level = self._random_level(self.assigned_probabilities, self.generator)
        print('level', level)
        point.level = level
        point.neighbors = [[] for _ in range(level)]
        next_start_node = self.entry_node
        for idx in range(len(self.assigned_probabilities) - 1, -1, -1):

            if idx > level:

                next_start_node = self.search(point, next_start_node, idx)
            else:
                print('neighbor neighbor', [len(n) for n in next_start_node.neighbors])
                next_start_node = self.search(point, next_start_node, idx)

                self.set_neighbors(node, next_start_node, idx)
                self.store[idx].append(node)

    def set_neighbors(self, query_node: Node, nearest_node: Node, level: int) -> Node:
        query_vector = np.array(query_node.vector)
        neighbors_vector = np.array([point.vector for point in nearest_node.neighbors[level]])
        neighbors_distance = self.calc_dist(query_vector, neighbors_vector)
        query_neighbors = []
        for idx, dist in enumerate(list(neighbors_distance)):
            neighbor = nearest_node.neighbors[level][idx]
            if len(neighbor.neighbors[level]) < self.max_neighbors:
                query_neighbors.append(neighbor)
                neighbor.neighbors[level].append(query_node)
            else:
                dist_query_neighbors = self.calc_dist(
                    query_vector,
                    np.array([point.vector for point in neighbor.neighbors[level]])
                )
                if np.max(dist_query_neighbors) > self.calc_dist(
                        query_vector, np.array(neighbor.vector
                                               ).reshape(1, -1)):

                    query_neighbors.append(neighbor)
                    neighbor.neighbors[level][np.argmax(dist_query_neighbors)] = query_node
        if len(query_neighbors) == 0:
            print('neighbor=====================>', len(query_neighbors), query_node.Id)

        query_node.neighbors.append(query_neighbors)
        return query_node

    def query(self, query_point: Node, k: int) -> List[Node]:
        start_point = self.entry_node
        for idx in range(len(self.assigned_probabilities) - 1, -1, -1):
            start_point = self.search(query_point, start_point, idx - self.level_num)
        return start_point.neighbors[0][:k]

    def search(self, query_point: Node, start_point: Node, level: int) -> Node:
        current_point = start_point
        while True:

            distance = self.calc_dist(np.array(query_point.vector), np.array(current_point.vector).reshape(1, -1))
            neighbors_vector = np.array([point.vector for point in current_point.neighbors[level]])

            neighbors_distance = self.calc_dist(np.array(query_point.vector), neighbors_vector)

            if np.min(neighbors_distance) > distance.item():
                return current_point
            current_point = current_point.neighbors[level][np.argmin(neighbors_distance)]

    @staticmethod
    def calc_dist(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        return np.linalg.norm(vectors - query, axis=1)


if __name__ == "__main__":
    node = Node(Id=uuid.uuid4(), vector=list(np.random.rand(4)))
    node1 = Node(Id=uuid.uuid4(), vector=list(np.random.rand(4)))

    store = VectorStore(8, [node, node1])
    store.start()

    for i in range(10000):
        print(i, i, i, i, i, i, i, i, i, i, i, i, i)
        vector = np.random.rand(4)
        node = Node(Id=uuid.uuid4(), vector=list(vector))
        store.insert_node(node)

    for i in range(len(store.store)):
        print(len(store.store[i]))

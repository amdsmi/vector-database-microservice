from src.hnsw import Node
import uuid
from pydantic import ValidationError
import pytest


def test_node():
    node_1 = Node(Id=uuid.uuid4(), vector=[1, 2, 3], level=3)
    node_2 = Node(Id=uuid.uuid4(), vector=[1, 2, 3], level=3, neighbors=[[node_1]])
    node_3 = Node(Id=uuid.uuid4(), vector=[1, 2, 3], level=3, pyload={'amd': 'smi'}, neighbors=[[node_1, node_2]])
    assert isinstance(node_1, Node)
    assert isinstance(node_2, Node)
    assert isinstance(node_3, Node)
    with pytest.raises(ValidationError):
        Node(Id=uuid.uuid4(), vector=[1, 2, 3], level=3, pyload={'amd': 'smi'}, neighbors=[[node_1, node_2, 3]])

    with pytest.raises(ValidationError):
        Node(Id=uuid.uuid4(), vector=[1, 2, 3], level=3, pyload=['amd'], neighbors=[[node_1, node_2]])

    with pytest.raises(ValidationError):
        Node(Id=uuid.uuid4(), vector=[1, 2, 3], level='i', pyload={'amd': 'smi'}, neighbors=[[node_1, node_2]])

    with pytest.raises(ValidationError):
        Node(Id=uuid.uuid4(), vector=1, level=3, pyload={'amd': 'smi'}, neighbors=[[node_1, node_2]])


def test_node_equal():
    node = Node(Id=uuid.uuid4(), vector=[1, 2, 3], level=3)
    assert node == node

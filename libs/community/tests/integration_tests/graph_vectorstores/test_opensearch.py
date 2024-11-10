"""Test of Astra DB graph vector store class `OpenSearchGraphVectorStore`.

Refer to `test_vectorstores.py` for the requirements to run.
"""

import pytest
import os
from typing import Any
from dotenv import load_dotenv
import json

from langchain_community.graph_vectorstores import OpenSearchGraphVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.graph_vectorstores.base import Node
from langchain_community.graph_vectorstores.links import Link, add_links

#region Embeddings
class ParserEmbeddings(Embeddings):
    """Parse input texts: if they are json for a List[float], fine.
    Otherwise, return all zeros and call it a day.
    """

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(txt) for txt in texts]

    def embed_query(self, text: str) -> list[float]:
        try:
            vals = json.loads(text)
        except json.JSONDecodeError:
            return [0.0] * self.dimension
        else:
            assert len(vals) == self.dimension
            return vals

@pytest.fixture
def embedding() -> Embeddings:
    return ParserEmbeddings(dimension=2)
#endregion

@pytest.fixture()
def graph_vector_store(embedding: Embeddings) -> OpenSearchGraphVectorStore:
        
        load_dotenv()
        
         # Access the variables
        host = os.getenv("OPENSEARCH_HOST", "localhost")
        port = int(os.getenv("OPENSEARCH_PORT", 9201))
        url = f"http://{host}:{port}"

        auth = (
            os.getenv("OPENSEARCH_USER", "admin"),
            os.getenv("OPENSEARCH_PASSWORD", "admin"),
        )
        index_name = os.getenv("OPENSEARCH_INDEX_NAME", "langchain")
        embedding_model = embedding
        print(f"Host: {host}, Port: {port}, Auth: {auth}, Index: {index_name}")
        return OpenSearchGraphVectorStore(
            opensearch_url=url,
            http_auth=auth,
            index_name=index_name,
            embedding=embedding_model,
            reset_index=True)

@pytest.fixture
def graph_vector_store_docs() -> list[Document]:
    docs_a = [
        Document(id="AL", page_content="[-1, 9]", metadata={"label": "AL"}),
        Document(id="A0", page_content="[0, 10]", metadata={"label": "A0"}),
        Document(id="AR", page_content="[1, 9]", metadata={"label": "AR"}),
    ]
    docs_b = [
        Document(id="BL", page_content="[9, 1]", metadata={"label": "BL"}),
        Document(id="B0", page_content="[10, 0]", metadata={"label": "B0"}),
        Document(id="BR", page_content="[9, -1]", metadata={"label": "BR"}),
    ]
    docs_f = [
        Document(id="FL", page_content="[1, -9]", metadata={"label": "FL"}),
        Document(id="F0", page_content="[0, -10]", metadata={"label": "F0"}),
        Document(id="FR", page_content="[-1, -9]", metadata={"label": "FR"}),
    ]
    docs_t = [
        Document(id="TL", page_content="[-9, -1]", metadata={"label": "TL"}),
        Document(id="T0", page_content="[-10, 0]", metadata={"label": "T0"}),
        Document(id="TR", page_content="[-9, 1]", metadata={"label": "TR"}),
    ]
    for doc_a, suffix in zip(docs_a, ["l", "0", "r"]):
        add_links(doc_a, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.outgoing(kind="at_example", tag=f"tag_{suffix}"))
        add_links(doc_a, Link.incoming(kind="af_example", tag=f"tag_{suffix}"))
    for doc_b, suffix in zip(docs_b, ["l", "0", "r"]):
        add_links(doc_b, Link.bidir(kind="ab_example", tag=f"tag_{suffix}"))
    for doc_t, suffix in zip(docs_t, ["l", "0", "r"]):
        add_links(doc_t, Link.incoming(kind="at_example", tag=f"tag_{suffix}"))
    for doc_f, suffix in zip(docs_f, ["l", "0", "r"]):
        add_links(doc_f, Link.outgoing(kind="af_example", tag=f"tag_{suffix}"))
    return docs_a + docs_b + docs_f + docs_t

@pytest.fixture(scope="function")
def populated_graph_vector_store(graph_vector_store: OpenSearchGraphVectorStore,graph_vector_store_docs: list[Document]) -> OpenSearchGraphVectorStore:
    graph_vector_store.add_documents(graph_vector_store_docs)
    return graph_vector_store
class TestOpenSearchGraphVectorStore:
    
    def test_add_documents(self, graph_vector_store: OpenSearchGraphVectorStore, graph_vector_store_docs: list[Document] ) -> None:
        """Test adding nodes to the graph vector store."""
        for r in graph_vector_store_docs:
            print(r.id)
        try :
            graph_vector_store.add_documents(graph_vector_store_docs)
        except Exception as e:
            pytest.fail(f"Exception occurred while inserting documents: {e}")    
            
        results = graph_vector_store.get_documents(k=100)    
        for r in results:
            print(r.id) 
            
        assert len(results) == len(graph_vector_store_docs)

    def test_similarity_search_sync(
            self,
            populated_graph_vector_store: OpenSearchGraphVectorStore,
        ) -> None:
        """Simple (non-graph) similarity search on a graph vector g_store."""
        ss_response = populated_graph_vector_store.similarity_search(query="[2, 10]", k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        
        ss_by_v_response = populated_graph_vector_store.similarity_search_by_vector(query_vector=[2, 10], k=2)
        ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
        assert ss_by_v_labels == ["AR", "A0"]
        
    async def test_similarity_search_async(
        self,
        populated_graph_vector_store: OpenSearchGraphVectorStore,
    ) -> None:
        """Simple (non-graph) similarity search on a graph vector store."""
        ss_response = await populated_graph_vector_store.asimilarity_search(query="[2, 10]", k=2)
        ss_labels = [doc.metadata["label"] for doc in ss_response]
        assert ss_labels == ["AR", "A0"]
        ss_by_v_response = await populated_graph_vector_store.asimilarity_search_by_vector(
            embedding=[2, 10], k=2
        )
        ss_by_v_labels = [doc.metadata["label"] for doc in ss_by_v_response]
        assert ss_by_v_labels == ["AR", "A0"]

    def test_metadata_search_sync(
        self,
        populated_graph_vector_store: OpenSearchGraphVectorStore,
    ) -> None:
        """Metadata search on a graph vector store."""
        mt_response = populated_graph_vector_store.search_by_metadata(
            metadata={"label": "T0"},
            k=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"
        
    async def test_metadata_search_async(
        self,
        populated_graph_vector_store: OpenSearchGraphVectorStore,
    ) -> None:
        """Metadata search on a graph vector store."""
        mt_response = await populated_graph_vector_store.ametadata_search(
            filter={"label": "T0"},
            n=2,
        )
        doc: Document = next(iter(mt_response))
        assert doc.page_content == "[-10, 0]"
        links: set[Link] = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "in"
        assert link.kind == "at_example"
        assert link.tag == "tag_0"
        
    def test_document_id_sync(
        self,
        populated_graph_vector_store: OpenSearchGraphVectorStore,
    ) -> None:
        """Get by document_id on a graph vector store."""
        doc = populated_graph_vector_store.search_by_id(document_id="FL")
        assert doc is not None
        assert doc.page_content == "[1, -9]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"

        invalid_doc = populated_graph_vector_store.search_by_id(document_id="invalid")
        assert invalid_doc is None

    async def test_document_id_async(
        self,
        populated_graph_vector_store: OpenSearchGraphVectorStore,
    ) -> None:
        """Get by document_id on a graph vector store."""
        doc = await populated_graph_vector_store.asearch_by_id(document_id="FL")
        assert doc is not None
        assert doc.page_content == "[1, -9]"
        links = doc.metadata["links"]
        assert len(links) == 1
        link: Link = links.pop()
        assert isinstance(link, Link)
        assert link.direction == "out"
        assert link.kind == "af_example"
        assert link.tag == "tag_l"

        invalid_doc = await g_store.aget_by_document_id(document_id="invalid")
        assert invalid_doc is None

    def test_traversal_search_sync(
        self,
        populated_graph_vector_store: OpenSearchGraphVectorStore,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        ts_response = populated_graph_vector_store.traversal_search(query="[2, 10]", k=2, depth=2)
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        ts_labels = {doc.metadata["label"] for doc in ts_response}
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

        # verify the same works as a retriever
        retriever = populated_graph_vector_store.as_retriever(
            search_type="traversal", search_kwargs={"k": 2, "depth": 2}
        )

        ts_labels = {
            doc.metadata["label"]
            for doc in retriever.get_relevant_documents(query="[2, 10]")
        }
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

    async def test_traversal_search_async(
        self,
        populated_graph_vector_store: OpenSearchGraphVectorStore,
    ) -> None:
        """Graph traversal search on a graph vector store."""
        ts_labels = set()
        async for doc in populated_graph_vector_store.atraversal_search(query="[2, 10]", k=2, depth=2):
            ts_labels.add(doc.metadata["label"])
        # this is a set, as some of the internals of trav.search are set-driven
        # so ordering is not deterministic:
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

        # verify the same works as a retriever
        retriever = populated_graph_vector_store.as_retriever(
            search_type="traversal", search_kwargs={"k": 2, "depth": 2}
        )

        ts_labels = {
            doc.metadata["label"]
            for doc in await retriever.aget_relevant_documents(query="[2, 10]")
        }
        assert ts_labels == {"AR", "A0", "BR", "B0", "TR", "T0"}

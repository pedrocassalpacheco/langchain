"""Opensearch graph vector store integration."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, is_dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    cast,
)

from langchain.vectorstores import OpenSearchVectorSearch
from langchain_core._api import beta
from langchain_core.documents import Document
from typing_extensions import override

from langchain_community.graph_vectorstores.base import GraphVectorStore, Node
from langchain_community.graph_vectorstores.content_graph import ContentGraph
from langchain_community.graph_vectorstores.links import METADATA_LINKS_KEY, Link

CGVST = TypeVar("GVST", bound="OpenSearchVectorSearch")

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


# region Link serialization and deserialization
def _serialize_links(links: list[Link]) -> str:
    class SetAndLinkEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if not isinstance(obj, type) and is_dataclass(obj):
                return asdict(obj)

            if isinstance(obj, Iterable):
                return list(obj)

            # Let the base class default method raise the TypeError
            return super().default(obj)

    return json.dumps(links, cls=SetAndLinkEncoder)

def _restore_links(doc: Document) -> Document:
    """Restores the links in the document by deserializing them from metadata.

    Args:
        doc: A single Document

    Returns:
        The same Document with restored links.
    """
    links = _deserialize_links(doc.metadata.get(METADATA_LINKS_KEY))
    doc.metadata[METADATA_LINKS_KEY] = links
    return doc

def _deserialize_links(json_blob: str | None) -> set[Link]:
    return {
        Link(kind=link["kind"], direction=link["direction"], tag=link["tag"])
        for link in cast(list[dict[str, Any]], json.loads(json_blob or "[]"))
    }

def _metadata_link_key(link: Link) -> str:
    return f"link:{link.kind}:{link.tag}"

def _metadata_link_value() -> str:
    return "link"

def _metadata_link_key(link: Link) -> str:
    return f"link:{link.kind}:{link.tag}"

def _doc_to_node(doc: Document) -> Node:
    metadata = doc.metadata.copy()
    return Node(
        id=doc.id,
        text=doc.page_content,
        metadata=metadata,
        links=metadata.get(METADATA_LINKS_KEY),
    )

class AdjacentNode:
    """Helper class for link manipulation.

    Attributes:
        id (str): The identifier of the node.
        links (list[Link]): A list of links associated with the node.
        embedding (list[float]): A list representing the node's embedding.

    Methods:

        __init__(node: Node, embedding: list[float]) -> None:
            Initializes an AdjacentNode instance with the given node and embedding.

    """

    id: str
    links: list[Link]
    embedding: list[float]

    def __init__(self, node: Node, embedding: list[float]) -> None:
        """Create an Adjacent Node."""
        self.id = node.id or ""
        self.links = node.links
        self.embedding = embedding

def _incoming_links(node: Node | AdjacentNode) -> set[Link]:
    return {link for link in node.links if link.direction in ["in", "bidir"]}

def _outgoing_links(node: Node | AdjacentNode) -> set[Link]:
    return {link for link in node.links if link.direction in ["out", "bidir"]}

# endregion

# region Langchain Document manipulation

def _build_docs_from_texts(
    texts: List[str],
    metadatas: Optional[List[dict]] = None,
    ids: Optional[List[str]] = None,
) -> List[Document]:
    docs: List[Document] = []
    for i, text in enumerate(texts):
        doc = Document(
            page_content=text,
        )
        if metadatas is not None:
            doc.metadata = metadatas[i]
        if ids is not None:
            doc.id = ids[i]
        docs.append(doc)
    return docs

def _add_ids_to_docs(
    docs: List[Document],
    ids: Optional[List[str]] = None,
) -> List[Document]:
    if ids is not None:
        for doc, doc_id in zip(docs, ids):
            doc.id = doc_id
    return docs

# endregion


@beta()
class OpenSearchGraphVectorStore(GraphVectorStore):
    """OpenSearchGraphVectorStore is a class that extends GraphVectorStore to provide
    integration with OpenSearch for storing and searching vectorized documents.

    Args:
        embedding (Embeddings): The embedding function to use for vectorizing documents.
        index_name (str): The name of the OpenSearch index.
        opensearch_url (str): The URL of the OpenSearch instance.
        auth (tuple): HTTP authentication credentials for OpenSearch.
        use_ssl (bool): Whether to use SSL for the OpenSearch connection.
        verify_certs (bool): Whether to verify SSL certificates.
        ssl_show_warn (bool): Whether to show SSL warnings.
        reset_index (bool): Whether to reset the OpenSearch index on initialization.
        os_vector_store (OpenSearchVectorSearch): The OpenSearch vector store instance.
    """

    def __init__(
        self,
        embedding: Embeddings,
        index_name: str = "myindex",
        opensearch_url: str = "http://localhost:9200",
        http_auth: tuple = {"admin, admin"},
        use_ssl: bool = True,
        verify_certs: bool = False,
        ssl_show_warn: bool = False,
        reset_index: bool = False,
    ) -> None:
        """Initialize the OpenSearchVectorStore.

        Args
            embedding (Embeddings): The embedding function to use.
            index_name (str, optional): The name of the OpenSearch index. Defaults to "myindex".
            opensearch_url (str, optional): The URL of the OpenSearch instance. Defaults to "http://localhost:9200".
            http_auth (tuple, optional): The HTTP authentication credentials. Defaults to {"admin, admin"}.
            use_ssl (bool, optional): Whether to use SSL. Defaults to True.
            verify_certs (bool, optional): Whether to verify SSL certificates. Defaults to False.
            ssl_show_warn (bool, optional): Whether to show SSL warnings. Defaults to False.
            reset_index (bool, optional): Whether to reset the index if it exists. Defaults to False.

        Returns
            None

        """
        self.embedding = embedding
        self.index_name = index_name
        self.opensearch_url = opensearch_url
        self.auth = http_auth
        self.use_ssl = use_ssl
        self.verify_certs = verify_certs
        self.ssl_show_warn = ssl_show_warn
        self.reset_index = reset_index

        self.os_vector_store = OpenSearchVectorSearch(
            index_name=self.index_name,
            embedding_function=self.embedding,
            opensearch_url=self.opensearch_url,
            http_auth=self.auth,
            use_ssl=self.use_ssl,
            verify_certs=self.verify_certs,
            ssl_show_warn=self.ssl_show_warn,
        )

        if self.reset_index and self.os_vector_store.index_exists(
            index_name=index_name
        ):
            self._truncate_index()
  
    # region Injestion Methods
    @override
    def add_documents(self, documents: Sequence[Document], **kwargs: Any) -> None:
        """Add a sequence of documents to the OpenSearch vector store.

        This method processes each document to serialize its metadata links if present,
        and then adds the documents to the OpenSearch vector store.

        Args
            documents (Sequence[Document]): A sequence of Document objects to be added.
            **kwargs (Any): Additional keyword arguments.

        Raises
            Any exceptions raised by the underlying OpenSearch vector store's add_documents method.

        """
        for doc in documents:
            if METADATA_LINKS_KEY in doc.metadata:
                doc.metadata[METADATA_LINKS_KEY] = _serialize_links(
                    doc.metadata[METADATA_LINKS_KEY]
                )
        self.os_vector_store.add_documents(documents, ids=[doc.id for doc in documents])

    async def aadd_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> None:
        """Asynchronously add documents to the vector store.

        Args:
            documents: A sequence of Document objects to add.
            **kwargs: Additional keyword arguments.

        """
        ids = [doc.id for doc in documents]
        page_contents = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        self.os_vector_store.aadd_texts(
            texts=page_contents, ids=ids, metadata=metadatas
        )

    def add_content_graph(self, graph: ContentGraph) -> None:
        """Add the content of a ContentGraph to the vector store.

        Args:
            graph (ContentGraph): The content graph to be added.

        """
        self.add_documents(graph.graph)

    async def aadd_content_graph(self, graph: ContentGraph) -> None:
        """Asynchronously adds the content of a given ContentGraph to the graph vector store.

        Args:
            graph (ContentGraph): The ContentGraph object containing the content to be added.

        """
        await self.aadd_documents(graph.graph)

    def from_texts(self, texts: List[str]) -> None:
        """Create nodes from texts and add them to the vector store."""
        # Convert all texts to Document objects at once
        documents = [Document(page_content=text) for text in texts]

        # Add all documents in a single call to add_documents
        self.add_documents(documents)
    # endregion

    # region Basic Search Methods

    @override
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search on the OpenSearch vector store.

        Args:
            query (str): The query string to search for similar documents.
            k (int, optional): The number of top similar documents to return. Defaults to 4.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[Document]: A list of documents that are most similar to the query.

        """
        search_type = kwargs.get("search_type", "knn")
        vector_field = kwargs.get("vector_field", "vector_field")
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")
        query_embedding = self.embedding.embed_query(query)
        query = {
            "size": k,
            "query": {search_type: {vector_field: {"vector": query_embedding, "k": k}}},
        }         

        # Execute the synchronous search
        response = self.os_vector_store.client.search(
            index=self.index_name,  body=query, size=k
        )
        hits = response["hits"]["hits"]

        # Generate and return Document objects from the search results
        return [
            _restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )
            for hit in hits
        ]
        
    @override
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from this graph store.

        Args:
            query: The query string.
            k: The number of Documents to return. Defaults to 4.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.

        """
        search_type = kwargs.get("search_type", "knn")
        vector_field = kwargs.get("vector_field", "vector_field")
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        query_embedding = self.embedding.embed_query(query)

        # Define the search query
        query = {
            "size": k,
            "query": {search_type: {vector_field: {"vector": query_embedding, "k": k}}},
        }

        # Perform the asynchronous search
        response = await self.os_vector_store.async_client.search(
            index=index_name, body=query
        )

        hits = response["hits"]["hits"]

        for hit in hits:
            yield _restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )

    @override
    def similarity_search_by_vector(
        self, query_vector: list[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Perform a similarity search on the OpenSearch vector store.

        Args:
            query (str): The query string to search for similar documents.
            k (int, optional): The number of top similar documents to return. Defaults to 4.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[Document]: A list of documents that are most similar to the query.

        """
        search_type = kwargs.get("search_type", "knn")
        vector_field = kwargs.get("vector_field", "vector_field")
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")
        query = {
            "size": k,
            "query": {search_type: {vector_field: {"vector": query_vector, "k": k}}},
        }         

        # Execute the synchronous search
        response = self.os_vector_store.client.search(
            index=self.index_name,  body=query, size=k
        )
        hits = response["hits"]["hits"]

        # Generate and return Document objects from the search results
        return [
            _restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )
            for hit in hits
        ]

    @override
    async def asimilarity_search_by_vector(
        self,
        query_vector: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Retrieve documents from this graph store.

        Args:
            query: The query string.
            k: The number of Documents to return. Defaults to 4.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
            Collection of retrieved documents.

        """
        search_type = kwargs.get("search_type", "knn")
        vector_field = kwargs.get("vector_field", "vector_field")
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        # Define the search query
        query = {
            "size": k,
            "query": {search_type: {vector_field: {"vector": query_vector, "k": k}}},
        }

        # Perform the asynchronous search
        response = await self.os_vector_store.async_client.search(
            index=index_name, body=query
        )

        hits = response["hits"]["hits"]

        for hit in hits:
            yield _restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )

    def search_by_id(self, document_id: str, **kwargs: Any) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.

        """
        try:
            text_field = kwargs.get("text_field", "text")
            metadata_field = kwargs.get("metadata_field", "metadata")
            response = self.os_vector_store.client.get(
                index=self.index_name, id=document_id
            )
            hit = response["_source"]

            return _restore_links(
                Document(
                    id=document_id,
                    page_content=hit[text_field],
                    metadata=(
                        hit
                        if metadata_field == "*" or metadata_field not in hit
                        else hit[metadata_field]
                    ),
                )
            )

        except Exception as e:
            logger.error("Error retrieving document: %s", e)

    async def asearch_by_id(self, document_id: str, **kwargs: Any) -> Document | None:
        """Retrieve a single document from the store, given its document ID.

        Args:
            document_id: The document ID

        Returns:
            The the document if it exists. Otherwise None.

        """
        try:
            text_field = kwargs.get("text_field", "text")
            metadata_field = kwargs.get("metadata_field", "metadata")
            response = await self.os_vector_store.async_client.get(
                index=self.index_name, id=document_id
            )
            hit = response["_source"]

            return _restore_links(
                Document(
                    id=document_id,
                    page_content=hit[text_field],
                    metadata=(
                        hit
                        if metadata_field == "*" or metadata_field not in hit
                        else hit[metadata_field]
                    ),
                )
            )

        except Exception as e:
            logger.error("Error retrieving document: %s", e)

    def search_by_metadata(
        self, metadata: Dict[str, Any] | None = None, k: int = 10, **kwargs: Any
    ) -> Iterable[Document]:
        """Search for documents in the OpenSearch vector store based on metadata.

        Args:
            metadata (Dict[str, Any] | None): A dictionary of metadata key-value pairs to search for.
            k (int): The number of top results to return. Defaults to 10.
            **kwargs (Any): Additional keyword arguments.
                - text_field (str): The field name in the document that contains the text content. Defaults to "text".
                - metadata_field (str): The field name in the document that contains the metadata. Defaults to "metadata".

        Returns:
            Iterable[Document]: An iterable of Document objects that match the search criteria.

        """
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        # Build a list of match queries for each metadata field
        query = {
            "bool": {
                "must": [
                    {"match": {f"metadata.{key}": value}}
                    for key, value in metadata.items()
                ]
            }
        }

        # Execute the synchronous search
        response = self.os_vector_store.client.search(
            index=self.index_name, body={"query": query}, size=k
        )
        hits = response["hits"]["hits"]

        # Generate and return Document objects from the search results
        return [
            _restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )
            for hit in hits
        ]

    async def asearch_by_metadata(
        self, metadata: dict[str, Any] | None = None, k: int = 10, **kwargs: Any
    ) -> AsyncIterable[Document]:
        """Async Search for documents in the OpenSearch vector store based on metadata.

        Args:
            metadata (Dict[str, Any] | None): A dictionary of metadata key-value pairs to search for.
            k (int): The number of top results to return. Defaults to 10.
            **kwargs (Any): Additional keyword arguments.
                - text_field (str): The field name in the document that contains the text content. Defaults to "text".
                - metadata_field (str): The field name in the document that contains the metadata. Defaults to "metadata".

        Returns:
            Iterable[Document]: An iterable of Document objects that match the search criteria.

        """
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")

        # Build a list of match queries for each metadata field
        if metadata:
            query = {
                "bool": {
                    "must": [
                        {"match": {f"metadata.{key}": value}}
                        for key, value in metadata.items()
                    ]
                }
            }
        else:
            query = {"match_all": {}}

        # Execute the search, Notice that we are using the async client
        response = await self.os_vector_store.async_client.search(
            index=self.index_name, body={"query": query}, size=k
        )
        hits = response["hits"]["hits"]

        for hit in hits:
            yield _restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )
                                
    
    def similarity_search_by_vector_and_metadata(self, query, metadata: Dict[str, Any] | None = None, k: int = 10, **kwargs: Any) -> Iterable[Document]: 
        
        search_type = kwargs.get("search_type", "knn")
        vector_field = kwargs.get("vector_field", "vector_field")
        index_name = kwargs.get("index_name", self.index_name)
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")
        query_vector = self.embedding.embed_query(query)

        query = {
            "size": k,
            "query": {
                "bool": {
                    "filter": [
                        # Metadata match conditions
                        *[
                            {"match": {f"metadata.{key}": value}}
                            for key, value in metadata.items()
                        ],
                    ],
                    "must": [
                        # Vector similarity using script_score
                        {
                           search_type: {vector_field: {"vector": query_vector, "k": k}},
                        }
                    ]
                }
            }
        }
        
        # Execute the synchronous search
        response = self.os_vector_store.client.search(
            index=self.index_name,  body=query, size=k
        )
        hits = response["hits"]["hits"]

        # Generate and return Document objects from the search results
        return [
            _restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )
            for hit in hits
        ]
        
    def get_documents(
        self, k: int = 10, **kwargs: Any
    ) -> Iterable[Document]:
        """Simple search to retrieve documents from the OpenSearch vector store.No search criteria
        Args:
            k: Number of documents to retrieve

        Returns:
            Iterable[Document]: An iterable of Document objects that match the search criteria.

        """
        
        text_field = kwargs.get("text_field", "text")
        metadata_field = kwargs.get("metadata_field", "metadata")
        
        # Execute the synchronous search
        response = self.os_vector_store.client.search(
            index=self.index_name, 
            body={"query": {"match_all": {}}},
            size=k
        )
            
        hits = response["hits"]["hits"]

        # Generate and return Document objects from the search results
        return [
            _restore_links(
                Document(
                    id=hit["_id"],
                    page_content=hit["_source"][text_field],
                    metadata=hit["_source"][metadata_field],
                )
            )
            for hit in hits
        ]        
    # endregion

    # region Traversal Search Methods
    @override
    def traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Retrieve documents from this graph store using MMR-traversal.

                This strategy first retrieves the top `fetch_k` results by similarity to
                the question. It then selects the top `k` results based on
                maximum-marginal relevance using the given `lambda_mult`.

                At each step, it considers the (remaining) documents from `fetch_k` as
                well as any documents connected by edges to a selected document
                retrieved based on similarity (a "root").

        Args:
        ----
            query (str): _description_
            k (int, optional): _description_. Defaults to 4.
            depth (int, optional): _description_. Defaults to 1.
            filter (dict[str, Any] | None, optional): _description_. Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Raises:
        ------
            RuntimeError: _description_

        Returns:
        -------
            Iterable[Document]: _description_

        """
        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited link to depth
        visited_links: dict[Link, int] = {}

        # Map from id to Document
        retrieved_docs: dict[str, Document] = {}

        def visit_nodes(d: int, docs: Iterable[Document]) -> None:
            """Recursively visit nodes and their outgoing links."""
            nonlocal visited_ids, visited_links, retrieved_docs

            # Iterate over nodes, tracking the *new* outgoing links for this
            # depth. These are links that are either new, or newly discovered at a
            # lower depth.
            outgoing_links: set[Link] = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    # If this node is at a closer depth, update visited_ids
                    if d <= visited_ids.get(doc.id, depth):
                        visited_ids[doc.id] = d

                        # If we can continue traversing from this node,
                        if d < depth:
                            node = _doc_to_node(doc=doc)
                            # Record any new (or newly discovered at a lower depth)
                            # links to the set to traverse.
                            for link in _outgoing_links(node=node):
                                if d <= visited_links.get(link, depth):
                                    # Record that we'll query this link at the
                                    # given depth, so we don't fetch it again
                                    # (unless we find it an earlier depth)
                                    visited_links[link] = d
                                    outgoing_links.add(link)

            if outgoing_links:
                for outgoing_link in outgoing_links:
                    metadata_filter = self._get_metadata_filter(
                        metadata=filter,
                        outgoing_link=outgoing_link,
                    )

                    docs = self.search_by_metadata(
                        metadata=metadata_filter, k=1000
                    )

                    visit_targets(d=d + 1, docs=docs)

        def visit_targets(d: int, docs: Iterable[Document]) -> None:
            """Visit target nodes retrieved from outgoing links."""
            nonlocal visited_ids, retrieved_docs

            new_ids_at_next_depth = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    if d <= visited_ids.get(doc.id, depth):
                        new_ids_at_next_depth.add(doc.id)

            if new_ids_at_next_depth:
                for doc_id in new_ids_at_next_depth:
                    if doc_id in retrieved_docs:
                        visit_nodes(d=d, docs=[retrieved_docs[doc_id]])
                    else:
                        new_doc = self.vector_store.get_by_document_id(
                            document_id=doc_id
                        )
                        if new_doc is not None:
                            visit_nodes(d=d, docs=[new_doc])

        # Start the traversal
        initial_docs = self.similarity_search(
            query=query,
            k=k
        )

        visit_nodes(d=0, docs=initial_docs)

        result_docs = []
        for doc_id in visited_ids:
            if doc_id in retrieved_docs:
                #result_docs.append(_restore_links(retrieved_docs[doc_id]))
                result_docs.append(retrieved_docs[doc_id])
            else:
                msg = f"retrieved_docs should contain id: {doc_id}"
                raise RuntimeError(msg)
        return result_docs

    @override
    async def atraversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[Document]:
        """Retrieve documents from this knowledge store.

        First, `k` nodes are retrieved using a vector search for the `query` string.
        Then, additional nodes are discovered up to the given `depth` from those
        starting nodes.

        Args:
        ----
            query: The query string.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4.
            depth: The maximum depth of edges to traverse. Defaults to 1.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            Collection of retrieved documents.

        """
        # Map from visited ID to depth
        visited_ids: dict[str, int] = {}

        # Map from visited link to depth
        visited_links: dict[Link, int] = {}

        # Map from id to Document
        retrieved_docs: dict[str, Document] = {}

        async def visit_nodes(d: int, docs: Iterable[Document]) -> None:
            """Recursively visit nodes and their outgoing links."""
            nonlocal visited_ids, visited_links, retrieved_docs

            # Iterate over nodes, tracking the *new* outgoing links for this
            # depth. These are links that are either new, or newly discovered at a
            # lower depth.
            outgoing_links: set[Link] = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    # If this node is at a closer depth, update visited_ids
                    if d <= visited_ids.get(doc.id, depth):
                        visited_ids[doc.id] = d

                        # If we can continue traversing from this node,
                        if d < depth:
                            node = _doc_to_node(doc=doc)
                            # Record any new (or newly discovered at a lower depth)
                            # links to the set to traverse.
                            for link in _outgoing_links(node=node):
                                if d <= visited_links.get(link, depth):
                                    # Record that we'll query this link at the
                                    # given depth, so we don't fetch it again
                                    # (unless we find it an earlier depth)
                                    visited_links[link] = d
                                    outgoing_links.add(link)

            if outgoing_links:
                metadata_search_tasks = []
                for outgoing_link in outgoing_links:
                    metadata_filter = self._get_metadata_filter(
                        metadata=filter,
                        outgoing_link=outgoing_link,
                    )
                    metadata_search_tasks.append(
                        asyncio.create_task(
                            self.asearch_by_metadata(
                                metadata=metadata_filter, k=1000
                            )
                        )
                    )
                results = await asyncio.gather(*metadata_search_tasks)

                # Visit targets concurrently
                visit_target_tasks = [
                    visit_targets(d=d + 1, docs=docs) for docs in results
                ]
                await asyncio.gather(*visit_target_tasks)

        async def visit_targets(d: int, docs: Iterable[Document]) -> None:
            """Visit target nodes retrieved from outgoing links."""
            nonlocal visited_ids, retrieved_docs

            new_ids_at_next_depth = set()
            for doc in docs:
                if doc.id is not None:
                    if doc.id not in retrieved_docs:
                        retrieved_docs[doc.id] = doc

                    if d <= visited_ids.get(doc.id, depth):
                        new_ids_at_next_depth.add(doc.id)

            if new_ids_at_next_depth:
                visit_node_tasks = [
                    visit_nodes(d=d, docs=[retrieved_docs[doc_id]])
                    for doc_id in new_ids_at_next_depth
                    if doc_id in retrieved_docs
                ]

                fetch_tasks = [
                    asyncio.create_task(
                        self.asearch_by_id(document_id=doc_id)
                    )
                    for doc_id in new_ids_at_next_depth
                    if doc_id not in retrieved_docs
                ]

                new_docs: list[Document | None] = await asyncio.gather(*fetch_tasks)

                visit_node_tasks.extend(
                    visit_nodes(d=d, docs=[new_doc])
                    for new_doc in new_docs
                    if new_doc is not None
                )

                await asyncio.gather(*visit_node_tasks)
      
        # Start the traversal
        initial_docs = self.asimilarity_search(query=query, k=k)

        visit_nodes(d=0, docs=initial_docs)

        result_docs = []
        for doc_id in visited_ids:
            if doc_id in retrieved_docs:
                # result_docs.append(_restore_links(retrieved_docs[doc_id]))
                result_docs.append(retrieved_docs[doc_id])
            else:
                msg = f"retrieved_docs should contain id: {doc_id}"
                raise RuntimeError(msg)
        return result_docs

    # endregion

    # region Graph MMR Search Methods
    @override
    def mmr_traversal_search(
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Retrieve documents from this graph store using MMR-traversal.

        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.

        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """

        async def collect_docs() -> Iterable[Document]:
            async_iter = self.ammr_traversal_search(
                query=query,
                initial_roots=initial_roots,
                k=k,
                depth=depth,
                fetch_k=fetch_k,
                adjacent_k=adjacent_k,
                lambda_mult=lambda_mult,
                score_threshold=score_threshold,
                filter=filter,
                **kwargs,
            )
            return [doc async for doc in async_iter]

        return asyncio.run(collect_docs())
    
    @override
    async def ammr_traversal_search(
        self,
        query: str,
        *,
        initial_roots: Sequence[str] = (),
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        filter: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[Document]:
        """Retrieve documents from this graph store using MMR-traversal.

        This strategy first retrieves the top `fetch_k` results by similarity to
        the question. It then selects the top `k` results based on
        maximum-marginal relevance using the given `lambda_mult`.

        At each step, it considers the (remaining) documents from `fetch_k` as
        well as any documents connected by edges to a selected document
        retrieved based on similarity (a "root").

        Args:
            query: The query string to search for.
            initial_roots: Optional list of document IDs to use for initializing search.
                The top `adjacent_k` nodes adjacent to each initial root will be
                included in the set of initial candidates. To fetch only in the
                neighborhood of these nodes, set `fetch_k = 0`.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of initial Documents to fetch via similarity.
                Will be added to the nodes adjacent to `initial_roots`.
                Defaults to 100.
            adjacent_k: Number of adjacent Documents to fetch.
                Defaults to 10.
            depth: Maximum depth of a node (number of edges) from a node
                retrieved via similarity. Defaults to 2.
            lambda_mult: Number between 0 and 1 that determines the degree
                of diversity among the results with 0 corresponding to maximum
                diversity and 1 to minimum diversity. Defaults to 0.5.
            score_threshold: Only documents with a score greater than or equal
                this threshold will be chosen. Defaults to -infinity.
            filter: Optional metadata to filter the results.
            **kwargs: Additional keyword arguments.
        """
        query_embedding = self.embedding.embed_query(query)
        helper = MmrHelper(
            k=k,
            query_embedding=query_embedding,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )

        # For each unselected node, stores the outgoing links.
        outgoing_links_map: dict[str, set[Link]] = {}
        visited_links: set[Link] = set()
        # Map from id to Document
        retrieved_docs: dict[str, Document] = {}

        async def fetch_neighborhood(neighborhood: Sequence[str]) -> None:
            nonlocal outgoing_links_map, visited_links, retrieved_docs

            # Put the neighborhood into the outgoing links, to avoid adding it
            # to the candidate set in the future.
            outgoing_links_map.update(
                {content_id: set() for content_id in neighborhood}
            )

            # Initialize the visited_links with the set of outgoing links from the
            # neighborhood. This prevents re-visiting them.
            visited_links = await self._get_outgoing_links(neighborhood)

            # Call `self._get_adjacent` to fetch the candidates.
            adjacent_nodes = await self._get_adjacent(
                links=visited_links,
                query_embedding=query_embedding,
                k_per_link=adjacent_k,
                filter=filter,
                retrieved_docs=retrieved_docs,
            )

            new_candidates: dict[str, list[float]] = {}
            for adjacent_node in adjacent_nodes:
                if adjacent_node.id not in outgoing_links_map:
                    outgoing_links_map[adjacent_node.id] = _outgoing_links(
                        node=adjacent_node
                    )
                    new_candidates[adjacent_node.id] = adjacent_node.embedding
            helper.add_candidates(new_candidates)

        async def fetch_initial_candidates() -> None:
            nonlocal outgoing_links_map, visited_links, retrieved_docs

            results = (
                await self.vector_store.asimilarity_search_with_embedding_id_by_vector(
                    embedding=query_embedding,
                    k=fetch_k,
                    filter=filter,
                )
            )

            candidates: dict[str, list[float]] = {}
            for doc, embedding, doc_id in results:
                if doc_id not in retrieved_docs:
                    retrieved_docs[doc_id] = doc

                if doc_id not in outgoing_links_map:
                    node = _doc_to_node(doc)
                    outgoing_links_map[doc_id] = _outgoing_links(node=node)
                    candidates[doc_id] = embedding
            helper.add_candidates(candidates)

        if initial_roots:
            await fetch_neighborhood(initial_roots)
        if fetch_k > 0:
            await fetch_initial_candidates()

        # Tracks the depth of each candidate.
        depths = {candidate_id: 0 for candidate_id in helper.candidate_ids()}

        # Select the best item, K times.
        for _ in range(k):
            selected_id = helper.pop_best()

            if selected_id is None:
                break

            next_depth = depths[selected_id] + 1
            if next_depth < depth:
                # If the next nodes would not exceed the depth limit, find the
                # adjacent nodes.

                # Find the links linked to from the selected ID.
                selected_outgoing_links = outgoing_links_map.pop(selected_id)

                # Don't re-visit already visited links.
                selected_outgoing_links.difference_update(visited_links)

                # Find the nodes with incoming links from those links.
                adjacent_nodes = await self._get_adjacent(
                    links=selected_outgoing_links,
                    query_embedding=query_embedding,
                    k_per_link=adjacent_k,
                    filter=filter,
                    retrieved_docs=retrieved_docs,
                )

                # Record the selected_outgoing_links as visited.
                visited_links.update(selected_outgoing_links)

                new_candidates = {}
                for adjacent_node in adjacent_nodes:
                    if adjacent_node.id not in outgoing_links_map:
                        outgoing_links_map[adjacent_node.id] = _outgoing_links(
                            node=adjacent_node
                        )
                        new_candidates[adjacent_node.id] = adjacent_node.embedding
                        if next_depth < depths.get(adjacent_node.id, depth + 1):
                            # If this is a new shortest depth, or there was no
                            # previous depth, update the depths. This ensures that
                            # when we discover a node we will have the shortest
                            # depth available.
                            #
                            # NOTE: No effort is made to traverse from nodes that
                            # were previously selected if they become reachable via
                            # a shorter path via nodes selected later. This is
                            # currently "intended", but may be worth experimenting
                            # with.
                            depths[adjacent_node.id] = next_depth
                helper.add_candidates(new_candidates)

        for doc_id, similarity_score, mmr_score in zip(
            helper.selected_ids,
            helper.selected_similarity_scores,
            helper.selected_mmr_scores,
        ):
            if doc_id in retrieved_docs:
                doc = self._restore_links(retrieved_docs[doc_id])
                doc.metadata["similarity_score"] = similarity_score
                doc.metadata["mmr_score"] = mmr_score
                yield doc
            else:
                msg = f"retrieved_docs should contain id: {doc_id}"
                raise RuntimeError(msg)

    # region Helper Methods  
      
    def _get_metadata_filter(
        self,
        metadata: dict[str, Any] | None = None,
        outgoing_link: Link | None = None,
    ) -> dict[str, Any]:
        if outgoing_link is None:
            return metadata or {}

        metadata_filter = {} if metadata is None else metadata.copy()
        metadata_filter[_metadata_link_key(link=outgoing_link)] = _metadata_link_value()
        return metadata_filter

    def _truncate_index(self) -> None:
        """Delete all documents in the index."""
        self.os_vector_store.client.delete_by_query(
            index=self.index_name, body={"query": {"match_all": {}}}
        )
        return None

    @override
    def add_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> Iterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    def delete(self, ids: List[str], **kwargs: Any) -> None:
        pass

    def get_node(self, node_id: str) -> Optional[Node]:
        pass

    def get_links(self, node_id: str) -> List[Link]:
        pass

    def add_link(self, link: Link) -> None:
        pass

    def delete_link(self, link: Link) -> None:
        pass

    def get_all_nodes(self) -> List[Node]:
        pass

    def get_all_links(self) -> List[Link]:
        pass

    def get_metadata(self, node_id: str) -> dict:
        pass

    def set_metadata(self, node_id: str, metadata: dict) -> None:
        pass

    @override
    def add_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> Iterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    @override
    async def aadd_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> AsyncIterable[str]:
        """Add nodes to the graph store.

        Args:
            nodes: the nodes to add.
            **kwargs: Additional keyword arguments.
        """
        pass

    async def _get_adjacent(
        self,
        links: set[Link],
        query_embedding: list[float],
        retrieved_docs: dict[str, Document],
        k_per_link: int | None = None,
        filter: dict[str, Any] | None = None,  # noqa: A002
    ) -> Iterable[AdjacentNode]:
        """Return the target nodes with incoming links from any of the given links.

        Args:
            links: The links to look for.
            query_embedding: The query embedding. Used to rank target nodes.
            retrieved_docs: A cache of retrieved docs. This will be added to.
            k_per_link: The number of target nodes to fetch for each link.
            filter: Optional metadata to filter the results.

        Returns:
            Iterable of adjacent edges.
        """
        targets: dict[str, AdjacentNode] = {}

        tasks = []
        for link in links:
            metadata_filter = self._get_metadata_filter(
                metadata=filter,
                outgoing_link=link,
            )

            tasks.append(
                self.asimilarity_search_by_vector(
                    query_vector=query_embedding,
                    k=k_per_link or 10,
                    filter=metadata_filter,
                )
            )

        results = await asyncio.gather(*tasks)

        for result in results:
            for doc, embedding, doc_id in result:
                if doc_id not in retrieved_docs:
                    retrieved_docs[doc_id] = doc
                if doc_id not in targets:
                    node = _doc_to_node(doc=doc)
                    targets[doc_id] = AdjacentNode(node=node, embedding=embedding)

        # TODO: Consider a combined limit based on the similarity and/or
        # predicated MMR score?
        return targets.values()
    
    # endregion
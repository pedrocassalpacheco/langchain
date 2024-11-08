"""Helper class to generate Document based graph for graph vector store."""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

from langchain.schema import Document
from pyvis.network import Network
from unstructured.documents.elements import Element
from unstructured.partition.pdf import partition_pdf

from langchain_community.graph_vectorstores.links import Link, add_links
from langchain_community.graph_vectorstores.networkx import documents_to_networkx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ContentGraph:
    """A class used to represent a Content Graph.

    Attributes:
    name : str
        The name of the content graph.
    metadata : Optional[Dict[str, Union[str, int]]]
    """
    
    def __init__(
        self, name: str, metadata: Optional[Dict[str, Union[str, int]]] = None
    ) -> None:

        self.name: str = name
        self.graph: List[Document] = []  # Store nodes as LangChain Documents
        self.root: Document = None
        self.infered_parent: Document = None
        self.infer_hierarchy: bool = True
        self.element_handlers = {
            "Formula": self.handle_formula,
            "FigureCaption": self.handle_figure_caption,
            "NarrativeText": self.handle_narrative_text,
            "ListItem": self.handle_list_item,
            "Title": self.handle_title,
            "Address": self.handle_address,
            "EmailAddress": self.handle_email_address,
            "Image": self.handle_image,
            "PageBreak": self.handle_page_break,
            "Table": self.handle_table,
            "Header": self.handle_header,
            "Footer": self.handle_footer,
            "CodeSnippet": self.handle_code_snippet,
            "PageNumber": self.handle_page_number,
            "Text": self.handle_text,
            "UncategorizedText": self.handle_uncategorized_text,
        }

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    def element_to_document(self, element: Element) -> Document:
        """Converts an Element object to a Document object.

        Args:
            element (Element): The element to be converted.

        Returns:
            Document: A document object containing the id, page content, and metadata
                      of the given element. The metadata includes the type of the element,
                      an empty list of links, and any additional metadata from the element.
        """
        return Document(
            id=element.id,
            page_content=element.text,
            metadata={
                "type": type(element).__name__,
                "links": [],
                **element.metadata.to_dict(),
            },
        )

    def fromPDFDocument(
        self,
        file_path: Path,
        output_image_path: Path,
        reset_graph: bool = False,
        infer_hierarchy: bool = True,
    ) -> None:
        """
        Synchronously processes a PDF document.

        :param pdf_path: The path to the PDF file.
        :param pdf_path: The path to were images are stored.
        """
        logger.info(f"Synchronously processing PDF document from '{file_path}'...")
        self.infer_hierarchy = infer_hierarchy
        self.graph.clear() if reset_graph else None

        self.infered_parent = self.root = Document(
            id="root",
            page_content=str(file_path),
            metadata={"file_date": datetime.fromtimestamp(file_path.stat().st_ctime)},
        )
        self.graph.append(self.root)

        if not file_path.exists() or not file_path.is_file():
            logger.error(f"File at {file_path} does not exist or is not a valid file.")
            return None

        elements = partition_pdf(
            filename=file_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            max_characters=2000,
            new_after_n_chars=1700,
            extract_image_block_output_dir="images/",
        )

        for element in elements:
            try:
                # Extract the element type from the class name
                element_type = type(element).__name__
                # handler = self.element_handlers.get(element_type)
                doc = self.element_to_document(element)
                self.graph.append(self.handle_hierarchy(element_type, doc))

                # if handler:
                # handler(element)
                # else:
                # print(f"Unknown element type {element_type}")

            except Exception as e:
                logger.error(
                    f"An error occurred while processing element {element.id}: {e}"
                )
                logger.error(e, exc_info=True)
                break

        return None
    def fromLangChainDocuments(
        self,
        documents: List[Document],
        output_image_path: Path,
        reset_graph: bool = False,
        infer_hierarchy: bool = True,
        deserialize_links: bool = False,
    ) -> None:
        """
        Synchronously processes langchain Documents into a content graph.

        :param pdf_path: The path to the PDF file.
        :param pdf_path: The path to were images are stored.
        """
        logger.info("Creating content graph from existing langchain documents ...")
        self.infer_hierarchy = infer_hierarchy
        self.graph.clear() if reset_graph else None

        self.infered_parent = self.root = Document(
            id="root",
            page_content=self.name,
            metadata={"file_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        )

        self.graph.append(self.root)

        for doc in documents:
            try:
                # Extract the element type from the class name
                element_type = doc.metadata["type"]
                doc = self.handle_hierarchy(element_type, doc)
                self.graph.append(doc)

            except Exception as e:
                logger.error(
                    f"An error occurred while processing element {doc.id}: {e}"
                )
                logger.error(e, exc_info=True)
                break

        return None
    def handle_formula(self, element: Element) -> None:
        doc = self.element_to_document(element)
        self.graph.append(self.add_hierarchy(doc))
        return

    def handle_figure_caption(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_narrative_text(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_text(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_list_item(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_title(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_address(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_email_address(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_image(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_page_break(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_table(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_header(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_footer(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_code_snippet(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_page_number(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    def handle_uncategorized_text(self, element: Element) -> None:
        self.graph.append(self.element_to_document(element))
        self.handle_hierarchy(element)
        return

    # Handles element hierarchy - TODO: Refactor
    def handle_hierarchy(self, element_type: str, doc: Document) -> Document:
        """Figure out the hierarchy of the document based on the element type. Please see langchain_community/graph_vectorstores/link.py for details."""
        if element_type == "Title":
            # From root to title
            add_links(self.root, Link.outgoing(kind=element_type, tag=doc.id))

            # From title to root
            add_links(doc, Link.incoming(kind="root", tag=self.root.id))

            # Record last title so it can be a parent to the next element
            self.infered_parent = doc

        elif self.infer_hierarchy is False and doc.metadata["parent_id"] is not None:
            # From parent to whatever this is
            parent_doc = next(
                (d for d in self.graph if doc.id == doc.metadata["parent_id"]), None
            )
            add_links(parent_doc, Link.outgoing(kind=element_type, tag=doc.id))

            # From whatever this is to parent
            add_links(doc, Link.incoming(kind=element_type, tag=parent_doc.id))

        else:
            # If there is no parent, link to the last saved possible parent
            add_links(
                self.infered_parent, Link.outgoing(kind=element_type, tag=doc.id)
            )

            # From whatever this is to parent
            add_links(doc, Link.incoming(kind=element_type, tag=self.infered_parent.id))

        return doc

    def documents_to_nx(self):
        documents_to_networkx(self.graph)

    def plot_graph(self, file_name):
        net = Network(
            notebook=True,
            cdn_resources="in_line",
            bgcolor="#222222",
            font_color="white",
            height="750px",
            width="100%",
        )
        # Convert the NetworkX graph to a PyVis graph
        G = documents_to_networkx(self.graph, tag_nodes=False)

        net.from_nx(G)

        # Render the graph in the Jupyter notebook
        net.show_buttons()
        net.show(file_name + ".html")

        return G

"""WebShop's narrow string-search path for the pinned Pyserini 0.17 runtime.

Use the same Anserini SimpleSearcher and default Java search/document calls as
Pyserini's Apache-2.0 wrapper, without importing its dense-encoder package:
https://github.com/castorini/pyserini/blob/pyserini-0.17.0/pyserini/search/lucene/_searcher.py
https://github.com/castorini/pyserini/blob/pyserini-0.17.0/pyserini/index/lucene/_base.py
This is not a general replacement for Pyserini's query/fusion/encoder APIs.
"""
from importlib.metadata import version


class RawDocument:
    def __init__(self, document):
        self.document = document

    def raw(self):
        return self.document.get('raw')


class LuceneTextSearcher:
    def __init__(self, index_path):
        if version('pyserini') != '0.17.0':
            raise RuntimeError('The lightweight WebShop search path requires the prepared Pyserini 0.17.0')
        # pyclass configures the installed Anserini jar via the existing setup.
        # Importing pyserini.search instead also imports FAISS/Transformer APIs.
        from pyserini.pyclass import autoclass
        self.object = autoclass('io.anserini.search.SimpleSearcher')(str(index_path))
        self.num_docs = self.object.getTotalNumDocuments()

    def search(self, query, k=10):
        if not isinstance(query, str):
            raise TypeError('WebShop searches require a string query')
        return list(self.object.search(query, k))

    def doc(self, docid):
        document = self.object.document(docid)
        return None if document is None else RawDocument(document)

    def close(self):
        self.object.close()

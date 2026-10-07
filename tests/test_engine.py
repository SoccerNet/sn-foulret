import faiss
import numpy as np
import pandas as pd

from retrieval.engine import RetrievalEngine


def test_search_clamps_top_k_to_index_size():
    engine = RetrievalEngine.__new__(RetrievalEngine)
    engine.index = faiss.IndexFlatIP(2)
    engine.index.add(np.array([[1.0, 0.0]], dtype="float32"))
    engine.df = pd.DataFrame([{"action_id": "Train_action_0"}])

    results = engine.search(np.array([[1.0, 0.0]], dtype="float32"), top_k=5)

    assert results == [{"action_id": "Train_action_0", "score": 1.0}]

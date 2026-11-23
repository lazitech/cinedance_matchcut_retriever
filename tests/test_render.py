from pathlib import Path

import numpy as np
from PIL import Image

from matchcut_retriever.render import render_results
from matchcut_retriever.retrieval import Library, QueryObject, rank
from matchcut_retriever.retrieval_config import RetrievalConfig
from test_retrieval import make_library


def test_render_results_writes_grid(tmp_path):
    root = make_library(tmp_path / "library", [("clip", 0, "disc")])
    library = Library.load(root)
    image = np.zeros((80, 120, 3), np.uint8)
    query = QueryObject.from_mask(image, library.mask(0))
    results = rank(library, query, "soft", params=RetrievalConfig(workers=1), k=1)
    output = tmp_path / "grid.jpg"
    render_results(image, query.mask, results, library, output)
    assert output.is_file()
    rendered = Image.open(output)
    assert rendered.width >= 900 and rendered.height >= 300

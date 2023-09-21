import pandas as pd
from typing_extensions import Annotated
from zenml import step
from zenml.integrations.facets.materializers.facets_materializer import FacetsComparison

from data_structures.schemas import StructuredData


@step
def data_facets(
    data_train: StructuredData, data_val: StructuredData, data_test: StructuredData
) -> tuple[
    Annotated[FacetsComparison, "facets_data"],
    Annotated[FacetsComparison, "facets_meta"],
    Annotated[FacetsComparison, "facets_target"],
]:
    facets_data = FacetsComparison(
        datasets=[
            {"name": "data_train", "table": data_train.data},
            {"name": "data_val", "table": data_val.data},
            {"name": "data_test", "table": data_test.data},
        ]
    )
    facets_meta = FacetsComparison(
        datasets=[
            {"name": "data_train", "table": data_train.meta},
            {"name": "data_val", "table": data_val.meta},
            {"name": "data_test", "table": data_test.meta},
        ]
    )
    facets_target = FacetsComparison(
        datasets=[
            {"name": "data_train", "table": pd.DataFrame(data_train.target.value)},
            {"name": "data_val", "table": pd.DataFrame(data_val.target.value)},
            {"name": "data_test", "table": pd.DataFrame(data_test.target.value)},
        ]
    )
    return facets_data, facets_meta, facets_target

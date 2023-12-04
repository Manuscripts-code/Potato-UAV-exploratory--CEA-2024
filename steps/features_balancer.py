import numpy as np
import pandas as pd
from rich import print
from typing_extensions import Annotated
from zenml import step

from configs import configs
from configs.parser import BalancerConfig
from data_manager.loaders import StructuredData


@step(enable_cache=False)
def features_balancer(
    data_train_feat: StructuredData,
    balancer_cfg: BalancerConfig,
) -> Annotated[StructuredData, "data_train_feat_aug"]:
    if not balancer_cfg.use:
        print("Skipping features balancer.")
        return data_train_feat

    print("Features will be balanced - the data imbalance will be corrected.")

    # !TODO: balancer is hardcoded here, in the future move outside
    # Note: now only appropriate for classification
    import imblearn.over_sampling as over_sampling

    from data_structures.schemas import ClassificationTarget, StructuredData

    label = data_train_feat.meta[[configs.VARIETY_ENG]].apply(tuple, axis=1)
    stratify_indices, _ = pd.factorize(label)

    data = []
    meta = []
    target = []
    for idx in np.unique(stratify_indices):
        indices = np.where(stratify_indices == idx)[0]
        data_part = data_train_feat[indices].reset_index()

        classes, counts = np.unique(data_part.target.value, return_counts=True)
        # Oversample both classes to the sum of counts
        balancer = over_sampling.SMOTE(
            sampling_strategy={class_: int(np.sum(counts)) for class_ in classes},
            random_state=configs.RANDOM_SEED,
        )
        data_res, target_res = balancer.fit_resample(data_part.data, data_part.target.value)
        meta_res = pd.DataFrame(
            {
                configs.VARIETY_ENG: data_part.meta[configs.VARIETY_ENG][0],
                configs.BLOCK_ENG: np.nan,
                configs.PLANT_ENG: np.nan,
                configs.TREATMENT_ENG: np.nan,
                configs.DATE_ENG: np.nan,
            },
            index=range(0, len(data_res)),
        )
        data.append(data_res)
        meta.append(meta_res)
        target.append(target_res)

    data = pd.concat(data).reset_index(drop=True)
    meta = pd.concat(meta).reset_index(drop=True)
    target_value = pd.concat(target).reset_index(drop=True)

    encoding = data_train_feat.target.encoding
    target_label = target_value.map(dict(encoding))

    target = ClassificationTarget(
        label=target_label,
        value=target_value,
        encoding=encoding,
    )

    return StructuredData(data=data, meta=meta, target=target)

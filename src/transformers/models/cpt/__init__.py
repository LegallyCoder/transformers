

from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_tokenizers_available, is_torch_available
from ...utils import OptionalDependencyNotAvailable


_import_structure = {"configuration_cpt": ["CPT_PRETRAINED_CONFIG_ARCHIVE_MAP", "CPTConfig"]}

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_cpt_fast"] = ["CPTTokenizerFast"]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["modeling_cpt"] = [
        "CPT_PRETRAINED_MODEL_ARCHIVE_LIST",
        "CPTForCausalLM",
        "CPTForQuestionAnswering",
        "CPTForSequenceClassification",
        "CPTForTokenClassification",
        "CPTLayer",
        "CPTModel",
        "CPTPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_cpt import CPT_PRETRAINED_CONFIG_ARCHIVE_MAP, CPTConfig

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_cpt_fast import CPTTokenizerFast

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .modeling_cpt import (
            CPT_PRETRAINED_MODEL_ARCHIVE_LIST,
            CPTForCausalLM,
            CPTForQuestionAnswering,
            CPTForSequenceClassification,
            CPTForTokenClassification,
            CPTLayer,
            CPTModel,
            CPTPreTrainedModel,
        )


else:   
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)

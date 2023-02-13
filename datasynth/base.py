from typing import ClassVar
from langchain.chains.base import Chain
from typing import Any
import os
class BaseChain(Chain):

    datatype: ClassVar[str]
    chain_type: ClassVar[str]
    registry: ClassVar[dict[Any, str]] = {}
    

    def __init_subclass__(cls, **kwargs:Any):
        super().__init_subclass__(**kwargs)
        cls.register(cls)

    @classmethod
    def register(cls, sub_cls:Any):
        if hasattr(sub_cls, 'datatype'):
            cls.registry[(sub_cls.chain_type, sub_cls.datatype)] = sub_cls

    @classmethod
    def from_name(cls, datatype:str) -> Chain:
        return cls.registry[(cls.chain_type, datatype)]()
    


def template_names(path: str) -> list[str]:
    """

    Args:
        path (str): templates directory

    Returns:
        list[str]: a list of strings 
    """
    result: list[str] = []
    items = os.listdir(path)
    for file in items:
        result.append(os.path.splitext(file)[0])
    return result
    

def auto_class(path, base_cls:type, class_suffix: str, generator_chain = None, normalizer_chain = None):
    """Dynamically creating classes with type

    Args:
        path (str): _description
        base_cls (type): _description_
        class_suffix (str): _description_
    """
    items = template_names(path)
    for item in items:
        generated_type: type = type(item.capitalize()+class_suffix, (base_cls,), {
            "datatype": item
        })

        if class_suffix == "TestPipeline":
            generated_type: type = type(item.capitalize()+class_suffix, (base_cls,), {
                "datatype": item,
                "generator": generator_chain.from_name(item),
                "discriminator": normalizer_chain.from_name(item)               
            })
 
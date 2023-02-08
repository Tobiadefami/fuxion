from typing import ClassVar
from langchain.chains.base import Chain

class BaseChain(Chain):

    datatype: ClassVar[str]
    chain_type: ClassVar[str]
    registry: ClassVar[dict] = {}
    

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.register(cls)

    @classmethod
    def register(cls, sub_cls):
        if hasattr(sub_cls, 'datatype'):
            cls.registry[(sub_cls.chain_type, sub_cls.datatype)] = sub_cls
    
    @classmethod
    def from_name(cls, datatype):
        return cls.registry[(cls.chain_type, datatype)]()

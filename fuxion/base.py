from typing import ClassVar
from langchain.chains.base import Chain
from typing import Any, Type
from langchain.cache import SQLiteCache
import langchain

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

class BaseChain(Chain):
    template_file: ClassVar[str]
    generator_template: ClassVar[str]
    normalizer_template: ClassVar[str]
    chain_type: ClassVar[str]
    registry: ClassVar[dict[Any, str]] = {}

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        cls.register(cls)

    @classmethod
    def register(cls, sub_cls: Any):
        if hasattr(sub_cls, "template_file"):
            cls.registry[(sub_cls.chain_type, sub_cls.template_file)] = sub_cls

    @classmethod
    def from_name(
        cls,
        template_file: str,
        class_suffix: str,
        base_cls: Type[Chain],
        *args,
        **kwargs
    ) -> Chain:

        template_name = template_file.split("/")[-1].split(".")[0]
        generated_type: type = type(
            template_name.capitalize() + class_suffix,
            (base_cls,),
            {"template_file": template_file},
        )
        return generated_type(*args, **kwargs)

    @classmethod
    def _from_name(
        cls,
        generator_template: str,
        normalizer_template: str,
        generator_chain: Chain, 
        normalizer_chain: Chain,
        base_cls: Type[Chain],
        class_suffix: str,
        *args,
        **kwargs
    ) -> Chain:
        """ Used to generate dynamic classes for base class == DatasetPipeline

        Args:
            generator_template (str): _description_
            normalizer_template (str): _description_
            generator_chain (Chain): _description_
            normalizer_chain (Chain): _description_
            base_cls (Type[Chain]): _description_
            class_suffix (str): _description_

        Returns:
            Chain: _description_
        """
        template_name: str = generator_template.split("/")[-1].split(".")[0]

        if cls.chain_type != "DatasetPipeline":
            return
        else:
            generated_type: Type[Chain] = type(
                template_name.capitalize() + class_suffix,
                (base_cls,),
                {
                    "generator_template": generator_template,
                    "normalizer_template": normalizer_template,
                    "generator": generator_chain.from_name(
                        generator_template,
                        *args,
                        base_cls=generator_chain,
                        class_suffix="Generator",
                        **kwargs
                    ),
                    "normalizer": normalizer_chain.from_name(
                        normalizer_template,
                        *args,
                        base_cls=normalizer_chain,
                        class_suffix="Normalizer",
                        **kwargs
                    ),
                },
            )
            return generated_type(*args, **kwargs)

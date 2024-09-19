from pydantic import create_model
from typing import List, Optional, Dict, Any, Type, Union, get_origin

def create_dynamic_model(structure: Any, model_name: str = "GeneratedItem") -> Type:
    if isinstance(structure, dict):
        fields = {}
        for key, value in structure.items():
            if isinstance(value, (dict, list, type)):
                fields[key] = (create_dynamic_model(value, f"{model_name}_{key.capitalize()}"), ...)
            else:
                raise ValueError(f"Invalid type for key '{key}'. Expected dict, list, or type, got {type(value)}")
        return create_model(model_name, **fields)

    elif isinstance(structure, list):
        if len(structure) != 1:
            raise ValueError("Lists must contain exactly one type specification")
        return List[create_dynamic_model(structure[0], f"{model_name}Item")]

    elif get_origin(structure) is Union:
        args = get_args(structure)
        return Union[tuple(create_dynamic_model(arg, f"{model_name}Union{i}") for i, arg in enumerate(args))]
    elif isinstance(structure, type):
        return structure
    else:
        raise ValueError(f"Invalid structure type: {type(structure)}. Expected dict, list, Union, or type")

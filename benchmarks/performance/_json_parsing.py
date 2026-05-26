# ruff: noqa: D103
"""Creation and search methods for brute force"""

import json

from unyt import unyt_array, unyt_quantity


class UnytEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, unyt_quantity):
            return {"__unyt_quantity__": True, "value": obj.v, "units": str(obj.units)}
        if isinstance(obj, unyt_array):
            return {
                "__unyt_array__": True,
                "values": obj.v.tolist(),
                "units": str(obj.units),
            }
        return super().default(obj)


def as_unyt(dct):
    if "__unyt_array__" in dct:
        return unyt_array(dct["values"], dct["units"])
    if "__unyt_quantity__" in dct:
        return unyt_quantity(dct["value"], dct["units"])
    return dct

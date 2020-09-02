from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union

import pytest

from summarize_from_feedback.utils import hyperparams


@dataclass
class Simple(hyperparams.HParams):
    mandatory_nodefault: int = None
    mandatory_withdefault: str = "foo"
    optional_nodefault: Optional[int] = None
    fun: bool = True


def test_simple_works():
    hp = Simple()
    with pytest.raises(TypeError):
        # mandatory_nodefault not set
        hp.validate()
    with pytest.raises(TypeError):
        hp.clone()
    assert hp.fun == True
    assert hp.mandatory_withdefault == "foo"
    hp.override_from_pairs(
        [("mandatory_nodefault", 3), ("optional_nodefault", None), ("fun", False)]
    )
    assert hp == Simple.from_json(dict(mandatory_nodefault=3, optional_nodefault=None, fun=False))
    hp.validate()
    assert hp == hp.clone()
    assert hp.mandatory_nodefault == 3
    assert hp.mandatory_withdefault == "foo"
    assert hp.optional_nodefault is None
    assert not hp.fun
    with pytest.raises(TypeError):
        hp.override_from_json(dict(fun=2))


def test_simple_failures():
    hp = Simple()
    with pytest.raises(TypeError):
        hp.validate()  # mandatory_nodefault unset
    with pytest.raises(TypeError):
        hp.override_from_pair("mandatory_nodefault", "abc")
    with pytest.raises(AttributeError):
        hp.override_from_pair("nonexistent_field", 7.0)
    with pytest.raises(TypeError):
        hp.override_from_pair("fun", "?")
    assert hp == Simple.from_json(dict())


@dataclass
class Nested(hyperparams.HParams):
    first: bool = False
    simple_1: Simple = field(default_factory=Simple)
    simple_2: Optional[Simple] = None


def test_nested():
    hp = Nested()
    hp.override_from_pairs(
        [
            ("simple_1.mandatory_nodefault", 8),
            ("simple_2", "on"),
            ("simple_2.mandatory_withdefault", "HELLO"),
        ]
    )
    assert hp == Nested.from_json(
        dict(simple_1=dict(mandatory_nodefault=8), simple_2=dict(mandatory_withdefault="HELLO"))
    )
    with pytest.raises(TypeError):
        hp.validate()  # simple_2.mandatory_nodefault unset
    hp.override_from_pairs(
        [("simple_2/mandatory_nodefault", 7), ("simple_1/optional_nodefault", 55)], separator="/"
    )
    hp.validate()
    assert hp == hp.clone()
    assert hp.simple_1.mandatory_nodefault == 8
    assert hp.simple_1.mandatory_withdefault == "foo"
    assert hp.simple_1.optional_nodefault == 55
    assert hp.simple_2.mandatory_nodefault == 7
    assert hp.simple_2.mandatory_withdefault == "HELLO"
    assert hp.simple_2.optional_nodefault is None

    hp.override_from_pair("simple_2", "off")
    hp.validate()
    assert hp == hp.clone()
    assert hp.simple_2 is None

    with pytest.raises((TypeError, AttributeError)):
        hp.override_from_pair("simple_2.fun", True)
    with pytest.raises(TypeError):
        hp.override_from_pair("simple_2", "BADVAL")
    assert hp == hp.clone()


def test_has_param():
    assert Nested.has_param("first")
    assert Nested.has_param("simple_1")
    assert Nested.has_param("simple_1.optional_nodefault")
    assert Nested.has_param("simple_2.mandatory_withdefault")
    assert not Nested.has_param("optional_nodefault")
    assert not Nested.has_param("xyz")
    assert not Nested.has_param("simple_1.simple_1")


def test_nested_dict():
    hp = Nested()
    hp.override_from_json(
        {"simple_1": {"mandatory_nodefault": 8}, "simple_2": {"mandatory_withdefault": "HELLO"}}
    )
    assert hp == Nested.from_json(
        dict(simple_1=dict(mandatory_nodefault=8), simple_2=dict(mandatory_withdefault="HELLO"))
    )
    with pytest.raises(TypeError):
        hp.validate()  # simple_2.mandatory_nodefault unset
    hp.override_from_json(
        {
            "simple_2": {"mandatory_nodefault": 7},
            "simple_1": {"optional_nodefault": 55},
            "first": True,
        }
    )
    hp.validate()
    assert hp.to_json() == {
        "first": True,
        "simple_1": {
            "mandatory_nodefault": 8,
            "mandatory_withdefault": "foo",
            "optional_nodefault": 55,
            "fun": True,
        },
        "simple_2": {
            "mandatory_nodefault": 7,
            "mandatory_withdefault": "HELLO",
            "optional_nodefault": None,
            "fun": True,
        },
    }
    assert hp == hp.clone()


def test_nested_order():
    with pytest.raises(AttributeError):
        # Have to set simple_2 on first
        hp = Nested()
        hp.override_from_pairs([("simple_2.fun", True), ("simple_2", "on")])
    # This works
    hp = Nested()
    hp.override_from_pairs([("simple_2", "on"), ("simple_2.fun", True)])
    assert hp == Nested.from_json(dict(simple_2=dict(fun=True)))
    # Turning on again doesn't override the values
    hp = Nested()
    hp.override_from_pairs([("simple_1.fun", False), ("simple_1", "on")])
    assert hp == Nested.from_json(dict(simple_1=dict(fun=False)))
    assert not hp.simple_1.fun
    with pytest.raises(TypeError):
        # simple_1.mandatory_nodefault not set
        hp.validate()
    with pytest.raises(TypeError):
        hp.clone()


@dataclass
class Deeply(hyperparams.HParams):
    nested: Nested = None


def test_deeply_nested():
    hp = Deeply()
    with pytest.raises(TypeError):
        # nested not set
        hp.validate()
    hp.override_from_pairs([("nested", "on"), ("nested.simple_2", "on")])
    assert hp != Deeply.from_json(dict())
    assert hp == Deeply.from_json(dict(nested=dict(simple_2=dict())))
    assert hp.nested is not None
    assert hp.nested.simple_2 is not None
    with pytest.raises(TypeError):
        # don't allow unsetting
        hp.override_from_json(dict(nested=None))
    with pytest.raises(TypeError):
        hp.override_from_pairs([("nested", "off")])
    hp.override_from_pairs([("nested.simple_2", "off")])
    assert hp.nested.simple_2 is None
    with pytest.raises(TypeError):
        hp.override_from_json(dict(nested=2))
    hp.override_from_json(dict(nested=Nested(simple_2=Simple())))
    assert hp.nested.simple_2 is not None
    hp.override_from_json(dict(nested=dict(simple_2=None)))
    assert hp.nested is not None
    assert hp.nested.simple_2 is None
    with pytest.raises(TypeError):
        # nested not set
        hp.validate()
    with pytest.raises(TypeError):
        hp.clone()


def test_flat_to_nested():
    def test_pair(flt, nst):
        assert hyperparams.flat_to_nested(flt) == nst
        assert hyperparams.nested_to_flat(nst) == flt

    test_pair({"a.b.c": 2, "a.c": 4, "c": 3}, dict(a=dict(b=dict(c=2), c=4), c=3))

    with pytest.raises(ValueError):
        hyperparams.flat_to_nested({"a.b.c": 2, "a.c": 4, "a": 3})

    with pytest.raises(ValueError):
        hyperparams.flat_to_nested({"a.b.c": 2, "a.b": 4, "c": 3})

    test_pair(
        {"a.b.c": 2, "a.c": [dict(a=4)], "c": (3, 2)},
        dict(a=dict(b=dict(c=2), c=[dict(a=4)]), c=(3, 2)),
    )


@dataclass
class Trickier(hyperparams.HParams):
    dict_field: Dict[str, int] = None
    tuple_list_field: Optional[Tuple[str, List[int]]] = None
    list_tuple_field: Optional[List[Tuple[str, int]]] = None
    nested_dict_field: Dict[str, List[Simple]] = None


def test_dict_field():
    hp = Trickier()
    hp.override_from_pairs([("dict_field", dict())])
    assert hp == Trickier.from_json(dict(dict_field=dict()))
    assert hp.dict_field == dict()
    hp.override_from_json(dict(dict_field=dict(a=3, b=4)))
    assert hp == Trickier.from_json(dict(dict_field=dict(a=3, b=4)))
    assert hp.to_json() == dict(
        dict_field=dict(a=3, b=4),
        tuple_list_field=None,
        list_tuple_field=None,
        nested_dict_field=None,
    )
    assert hp.dict_field == dict(a=3, b=4)
    with pytest.raises(TypeError):
        hp.override_from_json(dict(dict_field=dict(a="hmm", b=4)))
    assert hp.dict_field == dict(a=3, b=4)
    # overrides entire dict
    hp.override_from_json(dict(dict_field=dict(c=5)))
    assert hp.dict_field == dict(c=5)
    with pytest.raises(TypeError):
        hp.override_from_json(dict(dict_field=None))
    with pytest.raises(TypeError):
        hp.override_from_pair("dict_field", None)

    with pytest.raises(TypeError):
        hp.validate()  # nested_dict_field unset
    hp.override_from_pair("nested_dict_field", dict())
    hp.validate()
    assert hp == hp.clone()


def test_nested_type_field():
    hp = Trickier()
    with pytest.raises(TypeError):
        # first field should be str
        hp.override_from_json(dict(tuple_list_field=(2, [])))
    with pytest.raises(TypeError):
        # second field should be list
        hp.override_from_json(dict(tuple_list_field=("2", 3)))
    hp.override_from_json(dict(tuple_list_field=("2", [3])))
    assert hp == Trickier.from_json(dict(tuple_list_field=("2", [3])))
    with pytest.raises(TypeError):
        # should be a list
        hp.override_from_json(dict(list_tuple_field=("2", 3)))
    with pytest.raises(TypeError):
        # should be list of tuples
        hp.override_from_json(dict(list_tuple_field=["2"]))
    hp.override_from_json(dict(list_tuple_field=[]))
    hp.override_from_json(dict(list_tuple_field=[("2", 3)]))

    with pytest.raises(TypeError):
        # mandatory_nodefault not set
        hp.validate()
    with pytest.raises(TypeError):
        hp.clone()
    hp.dict_field = dict()
    hp.nested_dict_field = dict()
    hp.validate()
    assert hp == hp.clone()


def test_nested_dict_field():
    hp = Trickier()
    hp.override_from_json(dict(nested_dict_field=dict(a=[])))
    assert hp == Trickier.from_json(dict(nested_dict_field=dict(a=[])))
    assert hp.nested_dict_field == dict(a=[])
    with pytest.raises(TypeError):
        # b should be list of Simple, not integer
        hp.override_from_json(dict(nested_dict_field=dict(a=[], b=[2])))
    hp.override_from_json(dict(nested_dict_field=dict(a=[], b=[Simple()])))
    assert hp.to_json() == dict(
        dict_field=None,
        tuple_list_field=None,
        list_tuple_field=None,
        nested_dict_field=dict(a=[], b=[Simple().to_json()]),
    )


@dataclass
class Unions(hyperparams.HParams):
    union_field: Union[str, int] = "3"
    union_field_2: Union[Simple, List[str]] = None


def test_union_field():
    hp = Unions()
    assert hp.union_field == "3"
    hp.override_from_pairs([("union_field", 3)])
    assert hp == Unions.from_json(dict(union_field=3))
    assert hp.union_field == 3

    hp.override_from_pairs([("union_field_2", ["3"])])
    assert hp == Unions.from_json(dict(union_field=3, union_field_2=["3"]))
    assert hp.union_field_2 == ["3"]
    with pytest.raises(TypeError):
        hp.override_from_pairs([("union_field_2", dict())])
    hp.override_from_pairs([("union_field_2", Simple())])
    assert hp == Unions.from_json(dict(union_field=3, union_field_2=dict()))
    assert hp.union_field_2 == Simple()

    hp.override_from_json(dict(union_field_2=["3"]))
    assert hp.union_field_2 == ["3"]
    with pytest.raises(TypeError):
        hp.override_from_json(dict(union_field_2=2))
    hp.override_from_json(dict(union_field_2=dict()))
    assert hp.union_field_2 == Simple()
    with pytest.raises(TypeError):
        # mandatory_nodefault not set
        hp.validate()
    with pytest.raises(TypeError):
        hp.clone()
    hp.union_field_2.mandatory_nodefault = 3
    hp.validate()
    assert hp == hp.clone()


@dataclass
class InvalidUnion(hyperparams.HParams):
    union_field: Union[int, Simple] = 3
    union_field_2: Optional[Union[Simple, str]] = None
    union_field_3: Optional[Union[Optional[dict], Optional[Simple]]] = None


def test_invalid_union_field():
    hp = InvalidUnion()
    hp.override_from_pairs([("union_field", 3)])
    assert hp == InvalidUnion.from_json(dict(union_field=3))
    assert hp.union_field == 3

    with pytest.raises(TypeError):
        hp.override_from_pairs([("union_field_2", ["3"])])
    with pytest.raises(TypeError):
        hp.override_from_pairs([("union_field_3", dict())])
    with pytest.raises(TypeError):
        hp.clone()

import typing
from dataclasses import fields, is_dataclass
from functools import lru_cache

from typeguard import check_type


class HParams:
    """Used as a base class for hyperparameter structs. They also need to be annotated with @dataclass."""

    def override_from_pair(self, flat_k, v, separator="."):
        """Overrides values from a key-value pair (flat_k, v) = ('x.y', 1) or ('name', "foobar").

        Treats keys with separators as paths into nested HParams.
        """

        typemap = _flat_type_map(type(self), separator=separator)

        *ks, f = flat_k.split(separator)
        # Traverse down to the nested hparam value which the field will be set on
        hp = self
        for i, k in enumerate(ks):
            try:
                hp = getattr(hp, k)
            except AttributeError:
                raise AttributeError(
                    f"{separator.join(ks[:i]) if i else 'hparams'} is {hp} which has no field '{k}'"
                )

        try:
            old_v = getattr(hp, f)
        except AttributeError:
            raise AttributeError(f"{separator.join(ks)} is {hp} which has no field '{f}'")
        # Figure out what to set; handle the special 'on' and 'off' values for nested hparams
        hps_cls = _hparam_constructible_class(typemap[flat_k])
        if hps_cls is not None:
            if v == "on":
                if old_v is None:
                    # Set the nested hparam class to its default values
                    v = hps_cls()
                else:
                    # The field is already set; skip it so we don't override any of its values
                    # with the defaults
                    return
            elif v == "off":
                v = None

        # Set it!
        check_type(flat_k, v, typemap[flat_k])
        setattr(hp, f, v)

    @classmethod
    def has_param(cls, flat_k, separator="."):
        return flat_k in _flat_type_map(cls, separator=separator)

    def override_from_pairs(self, pairs, separator="."):
        """Overrides values from a list of key-value pairs like [('x.y', 1), ('name', "foobar")].

        Treats keys with separators as paths into nested HParams.

        Uses a list rather than a dict because order can matter for nested hyperparameters.
        In particular, you have to set x on before you set x.y; you also want to allow x to later be
        overridden to None even if x.y has already been set.
        """

        for flat_k, v in pairs:
            self.override_from_pair(flat_k, v, separator=separator)

    def clone(self):
        return type(self).from_json(self.to_json())

    @classmethod
    def from_json(cls, json_val):
        return _construct_from_json(cls, json_val)

    def override_from_json(self, json_val, key=""):
        typemap = _type_map(type(self))

        for k, v in json_val.items():
            if k not in typemap:
                raise AttributeError(f"{self} has no attribute {k}")
            t = typemap[k]
            hps_cls = _hparam_constructible_class(t)
            if hps_cls is not None and isinstance(v, dict):
                old_v = getattr(self, k)
                # initialize constructor, overwriting value if it's not the correct class (can happen with Unions)
                if old_v is None or not isinstance(old_v, hps_cls):
                    setattr(self, k, hps_cls())
                getattr(self, k).override_from_json(v, key + "." + k)
            else:
                new_v = _construct_from_json(t, v, key + "." + k)
                check_type(k, new_v, t)
                setattr(self, k, new_v)

    def to_json(self):
        return to_json(self)

    def validate(self, *, prefix=""):
        assert is_dataclass(self), f"You forgot to annotate {type(self)} with @dataclass"
        for f in fields(self):
            fieldval = getattr(self, f.name)
            check_type(prefix + f.name, fieldval, f.type)
            if isinstance(fieldval, HParams):
                fieldval.validate(prefix=prefix + f.name + ".")


def is_hparam_type(ty):
    if isinstance(ty, type) and issubclass(ty, HParams):
        assert is_dataclass(ty)
        return True
    else:
        return False


def is_hparam(x):
    return is_hparam_type(type(x))


def to_json(x):
    if is_hparam(x):
        return {f.name: to_json(getattr(x, f.name)) for f in fields(x)}
    if isinstance(x, list):
        return [to_json(y) for y in x]
    if isinstance(x, dict):
        return {k: to_json(v) for k, v in x.items()}
    return x


def _construct_from_json(ty, json_val, key=""):
    """
    Construct a value of type `ty` based on the json value `json_val`.
    """
    if json_val is None:
        return json_val
    if is_hparam_type(ty):
        if isinstance(json_val, ty):
            return json_val
        if not isinstance(json_val, dict):
            raise TypeError(
                f"Tried to construct attribute {key} of type {ty} with value {json_val}"
            )
        x = ty()
        x.override_from_json(json_val, key=key)
        return x
    if _is_list_type(ty):
        subtype = ty.__args__[0]
        return [_construct_from_json(subtype, y, key + ".listitem") for y in json_val]
    if _is_dict_type(ty):
        ktype = ty.__args__[0]
        vtype = ty.__args__[1]
        return {
            _construct_from_json(ktype, k, key + ".dictkey"): _construct_from_json(
                vtype, v, key + ".dictitem"
            )
            for k, v in json_val.items()
        }
    check_type(key, json_val, ty)
    return json_val


def _is_union_type(ty):
    return getattr(ty, "__origin__", None) is typing.Union


def _is_list_type(ty):
    return getattr(ty, "__origin__", None) is list


def _is_dict_type(ty):
    return getattr(ty, "__origin__", None) is dict


def _union_subtypes(ty):
    if _is_union_type(ty):
        results = []
        for x in ty.__args__:
            results.extend(_union_subtypes(x))
        return results
    return [ty]


def _hparam_constructible_class(ty):
    """
    Given a type, returns:
    - an unambiguous HParam subtype, if one exists,
    - None, if there are no HParam subtypes
    If there is ambiguity, throws a TypeError.
    """
    subtypes = _union_subtypes(ty)

    hparam_subtypes = [ty for ty in subtypes if is_hparam_type(ty)]

    if len(hparam_subtypes) > 1:
        raise TypeError(f"Unions with multiple HParam subtypes unsupported")
    if len(hparam_subtypes) == 0:
        return None
    assert len(hparam_subtypes) == 1
    hparam_ty = hparam_subtypes[0]
    if dict in subtypes:
        # avoid ambiguity for nested dict construction
        raise TypeError(f"Unions with both HParam and dict subtypes unsupported")
    if str in subtypes:
        # avoid ambiguity for "on"/"off"
        raise TypeError(f"Unions with both HParam and str subtypes unsupported")
    return hparam_ty


def dump(hparams, *, name="hparams"):
    print("%s:" % name)

    def dump_nested(hp, indent):
        for f in sorted(fields(hp), key=lambda f: f.name):
            v = getattr(hp, f.name)
            if isinstance(v, HParams):
                print("%s%s:" % (indent, f.name))
                dump_nested(v, indent=indent + "  ")
            else:
                print("%s%s: %s" % (indent, f.name, v))

    dump_nested(hparams, indent="  ")


@lru_cache()
def _type_map(ty):
    return {f.name: f.type for f in fields(ty)}


def flat_to_nested(flat_dict, separator="."):
    nested_dict = dict()
    for k, v in flat_dict.items():
        parts = k.split(separator)
        d = nested_dict
        subkey = None
        for part in parts[:-1]:
            subkey = part if subkey is None else subkey + separator + part
            if part not in d:
                d[part] = dict()
            if not isinstance(d[part], dict):
                raise ValueError(f"Set conflicting values for {subkey}")
            d = d[part]
        if parts[-1] in d:
            raise ValueError(f"Set conflicting values for {k}")
        d[parts[-1]] = v
    return nested_dict


def nested_to_flat(nested_dict, separator="."):
    flat_dict = dict()

    def helper(val, subkey_prefix=""):
        for k, v in val.items():
            if isinstance(v, dict):
                helper(v, subkey_prefix=subkey_prefix + k + separator)
            else:
                flat_dict[subkey_prefix + k] = v

    helper(nested_dict)
    return flat_dict


def _update_disjoint(dst: dict, src: dict):
    for k, v in src.items():
        assert k not in dst
        dst[k] = v


@lru_cache()
def _flat_type_map(ty, separator):
    typemap = {}
    for f in fields(ty):
        typemap[f.name] = f.type
        if is_hparam_type(f.type):
            nested = _flat_type_map(f.type, separator=separator)
        elif _is_union_type(f.type):
            nested = {}
            for ty_option in f.type.__args__:
                if is_hparam_type(ty_option):
                    _update_disjoint(nested, _flat_type_map(ty_option, separator=separator))
        else:
            nested = {}
        _update_disjoint(typemap, {f"{f.name}{separator}{k}": t for k, t in nested.items()})
    return typemap

from typing import Callable, Dict, Optional, Sequence, Tuple, Union

BindingVar = str
BindingVal = object
Descriptor = Union[int, str]
BindingMetadata = Dict[str, Union[Descriptor, bool]]

Binding = Tuple[Optional[BindingVar], Optional[BindingVal], BindingMetadata]
Experiment = Tuple[Binding, ...]
ExperimentGroup = Sequence[Experiment]


def combos(*xs: ExperimentGroup) -> ExperimentGroup:
    if xs:
        return [x + combo for x in xs[0] for combo in combos(*xs[1:])]
    else:
        return [()]


def each(*xs: ExperimentGroup) -> ExperimentGroup:
    return [y for x in xs for y in x]


def bind(
    var: Optional[BindingVar], val: Optional[BindingVal], descriptor: Descriptor = ""
) -> ExperimentGroup:
    extra: BindingMetadata = {}
    if descriptor:
        extra["descriptor"] = descriptor
    return [((var, val, extra),)]


def label(descriptor: Descriptor) -> ExperimentGroup:
    return bind(None, None, descriptor)


def labels(*descriptors: Descriptor) -> ExperimentGroup:
    return each(*[label(d) for d in descriptors])


def options(
    var: BindingVar, opts_with_descs: Sequence[Tuple[BindingVal, Descriptor]]
) -> ExperimentGroup:
    return each(*[bind(var, val, descriptor) for val, descriptor in opts_with_descs])


def _shortstr(v: object) -> str:
    if isinstance(v, float):
        s = f"{v:.03}"
        if "." in s:
            s = s.lstrip("0").replace(".", "x")
    else:
        s = str(v)
    return s


def options_shortdesc(var: BindingVar, desc: str, opts: Sequence[BindingVal]) -> ExperimentGroup:
    return each(*[bind(var, val, desc + _shortstr(val)) for val in opts])


def options_vardesc(var: BindingVar, opts: Sequence[BindingVal]) -> ExperimentGroup:
    return each(*[bind(var, val, var + _shortstr(val)) for val in opts])


def repeat(n: int) -> ExperimentGroup:
    return each(*[label(i) for i in range(n)])


# list monad bind; passes descriptors to body
def foreach(inputs: ExperimentGroup, body: Callable[..., ExperimentGroup]) -> ExperimentGroup:
    return [
        inp + y for inp in inputs for y in body(*[extra["descriptor"] for var, val, extra in inp])
    ]


def bind_nested(
    prefix: str, binds: ExperimentGroup, descriptor: Optional[Descriptor] = None
) -> ExperimentGroup:
    ret: ExperimentGroup = [
        tuple([(var if var is None else prefix + "." + var, val, extra) for (var, val, extra) in x])
        for x in binds
    ]
    if descriptor is not None:
        ret = combos(ret, label(descriptor))
    return combos(bind(prefix, "on"), ret)

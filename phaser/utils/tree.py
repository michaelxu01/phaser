from functools import wraps
import typing as t

from numpy.typing import ArrayLike, DTypeLike

from phaser.utils.num import get_array_module, is_torch, xp_is_jax, xp_is_torch

Tree: t.TypeAlias = t.Any


def map(
    f: t.Callable[..., t.Any],
    tree: Tree,
    *rest: Tree,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> t.Any:
    if is_torch(tree):
        from torch.utils._pytree import tree_map  # type: ignore
        return tree_map(f, tree, *rest, is_leaf=is_leaf)

    import jax.tree  # type: ignore
    return jax.tree.map(f, tree, *rest, is_leaf=is_leaf)


def map_with_path(
    f: t.Callable[..., t.Any],
    tree: Tree,
    *rest: Tree,
    is_leaf: t.Optional[t.Callable[..., t.Any]] = None,
) -> t.Any:
    if is_torch(tree):
        from torch.utils._pytree import tree_map_with_path  # type: ignore
        return tree_map_with_path(f, tree, *rest, is_leaf=is_leaf)

    from jax.tree_util import tree_map_with_path  # type: ignore
    return tree_map_with_path(f, tree, *rest, is_leaf=is_leaf)


def grad(
    f: t.Callable,
    argnums: t.Union[int, t.Tuple[int, ...]] = 0,
    has_aux: bool = False, *, xp: t.Optional[t.Any] = None,
) -> t.Callable[..., Tree]:
    if xp is None or xp_is_jax(xp):
        import jax  # type: ignore
        return jax.grad(f, argnums, has_aux=has_aux)
    if xp_is_torch(xp):
        import torch.func  # type: ignore
        return torch.func.grad(f, argnums, has_aux=has_aux)
    raise ValueError("`grad` is only supported for backends 'jax' and 'torch'")


def value_and_grad(
    f: t.Callable,
    argnums: t.Union[int, t.Tuple[int, ...]] = 0,
    has_aux: bool = False, *, xp: t.Optional[t.Any] = None,
) -> t.Callable[..., t.Tuple[Tree, Tree]]:
    if xp is None or xp_is_jax(xp):
        import jax  # type: ignore
        return jax.value_and_grad(f, argnums, has_aux=has_aux)
    if not xp_is_torch(xp):
        raise ValueError("`grad` is only supported for backends 'jax' and 'torch'")

    import torch.func  # type: ignore
    f = torch.func.grad_and_value(f, argnums, has_aux=has_aux)

    # flip order of return values
    @wraps(f)
    def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Tuple[Tree, Tree]:
        (grad, value) = f(*args, **kwargs)
        return (value, grad)

    return wrapper



def zeros_like(
    tree: Tree, dtype: DTypeLike = None,
) -> Tree:
    xp = get_array_module(tree)
    return map(lambda x: xp.zeros_like(x, dtype=dtype), tree)


def ones_like(
    tree: Tree, dtype: DTypeLike = None,
) -> Tree:
    xp = get_array_module(tree)
    return map(lambda x: xp.ones_like(x, dtype=dtype), tree)


def full_like(
    tree: Tree, fill_value: ArrayLike,
    dtype: DTypeLike = None,
) -> Tree:
    xp = get_array_module(tree)
    return map(lambda x: xp.full_like(x, fill_value, dtype=dtype), tree)


def cast(
    tree: Tree, dtype: t.Optional[DTypeLike],
) -> Tree:
    if dtype is None:
        return tree
    return map(lambda x: x.astype(dtype), tree)


def clip(
    tree: Tree,
    min_value: t.Optional[ArrayLike] = None,
    max_value: t.Optional[ArrayLike] = None,
) -> Tree:
    xp = get_array_module(tree)
    return map(lambda x: xp.clip(x, min_value, max_value), tree)


def conj(
    tree: Tree
) -> Tree:
    xp = get_array_module(tree)
    return map(xp.conj, tree)
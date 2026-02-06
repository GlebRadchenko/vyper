import pytest

from tests.venom_utils import PrePostChecker
from vyper.venom.passes import BranchThreadingPass

pytestmark = pytest.mark.hevm


_check_pre_post = PrePostChecker([BranchThreadingPass], default_hevm=False)


def test_branch_threading_basic_jump_chain():
    pre = """
    main:
        jmp @a
    a:
        jmp @b
    b:
        sink 1
    """

    post = """
    main:
        jmp @b
    a:
        jmp @b
    b:
        sink 1
    """

    _check_pre_post(pre, post)


def test_branch_threading_phi_rewrite():
    pre = """
    main:
        %cond = source
        jnz %cond, @left, @right
    left:
        %x = 10
        jmp @mid
    right:
        %y = 20
        jmp @join
    mid:
        jmp @join
    join:
        %z = phi @mid, %x, @right, %y
        sink %z
    """

    post = """
    main:
        %cond = source
        jnz %cond, @left, @right
    left:
        %x = 10
        jmp @join
    right:
        %y = 20
        jmp @join
    mid:
        jmp @join
    join:
        %z = phi @left, %x, @right, %y
        sink %z
    """

    _check_pre_post(pre, post)


def test_branch_threading_rejects_phi_conflict():
    pre = """
    main:
        %cond = source
        %v0 = 1
        jnz %cond, @mid, @join
    mid:
        %v1 = 2
        jmp @join
    join:
        %p = phi @mid, %v1, @main, %v0
        sink %p
    """

    _check_pre_post(pre, pre)


def test_branch_threading_dedups_equal_phi_arms_and_collapses_branch():
    pre = """
    main:
        %cond = source
        %v = 1
        jnz %cond, @mid, @join
    mid:
        jmp @join
    join:
        %p = phi @mid, %v, @main, %v
        sink %p
    """

    post = """
    main:
        %cond = source
        %v = 1
        jmp @join
    mid:
        jmp @join
    join:
        %p = phi @main, %v
        sink %p
    """

    _check_pre_post(pre, post)

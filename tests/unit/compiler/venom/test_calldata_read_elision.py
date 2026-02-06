import pytest

from tests.venom_utils import PrePostChecker
from vyper.venom.passes import CalldataReadElisionPass

pytestmark = pytest.mark.hevm


_check_pre_post = PrePostChecker([CalldataReadElisionPass], default_hevm=False)


def test_basic_calldatacopy_mload_elision():
    pre = """
    main:
        calldatacopy 64, 4, 128
        %x = mload 128
        %y = mload 160
        sink %x, %y
    """

    post = """
    main:
        calldatacopy 64, 4, 128
        %x = calldataload 4
        %y = calldataload 36
        sink %x, %y
    """

    _check_pre_post(pre, post)


def test_clobber_blocks_elision():
    pre = """
    main:
        calldatacopy 64, 4, 128
        mstore 0, 128
        %x = mload 128
        sink %x
    """

    _check_pre_post(pre, pre)


def test_out_of_range_not_elided():
    pre = """
    main:
        calldatacopy 32, 4, 128
        %x = mload 160
        sink %x
    """

    _check_pre_post(pre, pre)


def test_alloca_backed_pointer_elision():
    pre = """
    main:
        %base = alloca 64
        calldatacopy 64, 4, %base
        %p = gep 32, %base
        %x = mload %p
        sink %x
    """

    post = """
    main:
        %base = alloca 64
        calldatacopy 64, 4, %base
        %p = gep 32, %base
        %x = calldataload 36
        sink %x
    """

    _check_pre_post(pre, post)


def test_unknown_size_write_invalidates_copy_tracking():
    pre = """
    main:
        calldatacopy 64, 4, 128
        %sz = source
        mcopy %sz, 0, 128
        %x = mload 128
        sink %x
    """

    _check_pre_post(pre, pre)

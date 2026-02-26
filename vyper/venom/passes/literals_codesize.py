from vyper.utils import evm_not
from vyper.venom.basicblock import IRInstruction, IRLiteral
from vyper.venom.passes.base_pass import IRPass

# not takes 1 byte1, so it makes sense to use it when we can save at least
# 1 byte
NOT_THRESHOLD = 1

# shl takes 3 bytes, so it makes sense to use it when we can save at least
# 3 bytes
SHL_THRESHOLD = 3


class ReduceLiteralsCodesize(IRPass):
    def run_pass(self):
        for bb in self.function.get_basic_blocks():
            self._process_bb(bb)

    def _process_bb(self, bb):
        for inst in bb.instructions:
            if inst.opcode != "assign":
                continue

            (op,) = inst.operands
            if not isinstance(op, IRLiteral):
                continue

            val = op.value % (2**256)

            # Rebuild big low-bit masks as `shr (not 0)`:
            #   (1<<n)-1  =>  (not 0) >> (256-n)
            # This is often much smaller than PUSH20/PUSH32 masks and works
            # particularly well with LiteralAliasPass on repeated address masks.
            if self._rewrite_low_ones_mask(inst, val):
                continue

            # calculate amount of bits saved by not optimization
            not_benefit = ((len(hex(val)) // 2 - len(hex(evm_not(val))) // 2) - NOT_THRESHOLD) * 8

            # calculate amount of bits saved by shl optimization
            binz = bin(val)[2:]
            ix = len(binz) - binz.rfind("1")
            shl_benefit = ix - SHL_THRESHOLD * 8

            if not_benefit <= 0 and shl_benefit <= 0:
                # no optimization can be done here
                continue

            if not_benefit >= shl_benefit:
                assert not_benefit > 0  # implied by previous conditions
                # transform things like 0xffff...01 to (not 0xfe)
                inst.opcode = "not"
                inst.operands = [IRLiteral(evm_not(val))]
                continue
            else:
                assert shl_benefit > 0  # implied by previous conditions
                # transform things like 0x123400....000 to 0x1234 << ...
                ix -= 1
                # sanity check
                assert (val >> ix) << ix == val, val
                assert (val >> ix) & 1 == 1, val

                inst.opcode = "shl"
                inst.operands = [IRLiteral(val >> ix), IRLiteral(ix)]
                continue

    @staticmethod
    def _push_size(value: int) -> int:
        value = value % (2**256)
        if value == 0:
            return 1  # PUSH0
        return 1 + ((value.bit_length() + 7) // 8)

    def _rewrite_low_ones_mask(self, inst: IRInstruction, val: int) -> bool:
        if val == 0 or val == (2**256 - 1):
            return False
        # Low-bit ones mask test: val == 2^n - 1.
        if (val & (val + 1)) != 0:
            return False

        ones = val.bit_length()
        if ones <= 96 or ones >= 256:
            # Keep this optimization focused on large masks (e.g. 160-bit address mask).
            return False

        shift = 256 - ones
        direct_size = self._push_size(val)
        alt_size = self._push_size(0) + 1 + self._push_size(shift) + 1  # not + shr
        if alt_size >= direct_size:
            return False

        fn = inst.parent.parent
        tmp = fn.get_next_variable()
        not_inst = IRInstruction("not", [IRLiteral(0)], outputs=[tmp])
        inst.parent.insert_instruction(not_inst, inst.parent.instructions.index(inst))
        # Internal order for shr is [value, shift].
        inst.opcode = "shr"
        inst.operands = [tmp, IRLiteral(shift)]
        return True

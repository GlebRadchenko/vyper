from vyper.utils import SizeLimits, int_bounds, int_log2, is_power_of_two, wrap256
from vyper.evm.address_space import MEMORY
from vyper.venom.analysis import BasePtrAnalysis, MemoryAliasAnalysis
from vyper.venom.analysis.dfg import DFGAnalysis
from vyper.venom.analysis.liveness import LivenessAnalysis
from vyper.venom.analysis.variable_range import VariableRangeAnalysis
from vyper.venom.basicblock import (
    COMPARATOR_INSTRUCTIONS,
    IRInstruction,
    IRLabel,
    IRLiteral,
    IROperand,
    IRVariable,
    flip_comparison_opcode,
)
from vyper.venom.effects import Effects
from vyper.venom.passes.base_pass import InstUpdater, IRPass

TRUTHY_INSTRUCTIONS = ("iszero", "jnz", "assert", "assert_unreachable")
ADDRESS_MASK_160 = (1 << 160) - 1
ADDRESS_SPACE_160 = 1 << 160


def lit_eq(op: IROperand, val: int) -> bool:
    return isinstance(op, IRLiteral) and wrap256(op.value) == wrap256(val)


class AlgebraicOptimizationPass(IRPass):
    """
    This pass reduces algebraic evaluatable expressions.

    It currently optimizes:
      - iszero chains
      - binops
      - offset adds
      - signextend elimination via range analysis
    """

    dfg: DFGAnalysis
    updater: InstUpdater
    range_analysis: VariableRangeAnalysis
    base_ptr: BasePtrAnalysis
    mem_alias: MemoryAliasAnalysis

    def run_pass(self):
        self.dfg = self.analyses_cache.request_analysis(DFGAnalysis)
        self.range_analysis = self.analyses_cache.force_analysis(VariableRangeAnalysis)
        self.base_ptr = self.analyses_cache.request_analysis(BasePtrAnalysis)
        self.mem_alias = self.analyses_cache.request_analysis(MemoryAliasAnalysis)
        self.updater = InstUpdater(self.dfg)
        self._handle_offset()

        self._algebraic_opt()
        self._optimize_iszero_chains()
        self._algebraic_opt()

        self.analyses_cache.invalidate_analysis(LivenessAnalysis)

    def _optimize_iszero_chains(self) -> None:
        fn = self.function
        for bb in fn.get_basic_blocks():
            for inst in bb.instructions:
                if inst.opcode != "iszero":
                    continue

                iszero_chain = self._get_iszero_chain(inst.operands[0])
                iszero_count = len(iszero_chain)
                if iszero_count == 0:
                    continue

                inst_out = inst.output
                for use_inst in self.dfg.get_uses(inst_out).copy():
                    opcode = use_inst.opcode

                    if opcode == "iszero":
                        # We keep iszero instuctions as is
                        continue
                    if opcode in ("jnz", "assert", "assert_unreachable"):
                        # instructions that accept a truthy value as input:
                        # we can remove up to all the iszero instructions
                        keep_count = 1 - iszero_count % 2
                    else:
                        # all other instructions:
                        # we need to keep at least one or two iszero instructions
                        keep_count = 1 + iszero_count % 2

                    if keep_count >= iszero_count:
                        continue

                    out_var = iszero_chain[keep_count].operands[0]
                    self.updater.update_operands(use_inst, {inst_out: out_var})

    def _get_iszero_chain(self, op: IROperand) -> list[IRInstruction]:
        chain: list[IRInstruction] = []

        while True:
            if not isinstance(op, IRVariable):
                break
            inst = self.dfg.get_producing_instruction(op)
            if inst is None or inst.opcode != "iszero":
                break
            op = inst.operands[0]
            chain.append(inst)

        chain.reverse()
        return chain

    def _handle_offset(self):
        for bb in self.function.get_basic_blocks():
            for inst in bb.instructions:
                if (
                    inst.opcode == "add"
                    and self._is_lit(inst.operands[0])
                    and isinstance(inst.operands[1], IRLabel)
                ):
                    inst.opcode = "offset"

    def _is_lit(self, operand: IROperand) -> bool:
        return isinstance(operand, IRLiteral)

    def _is_address_mask_literal(self, operand: IROperand) -> bool:
        return isinstance(operand, IRLiteral) and wrap256(operand.value) == ADDRESS_MASK_160

    def _try_const_uint(self, operand: IROperand, seen: set) -> int | None:
        """
        Try to fold an operand to an exact uint256 constant.

        This is intentionally conservative and only supports operations needed
        for robust address-mask recognition.
        """
        if isinstance(operand, IRLiteral):
            return wrap256(operand.value)

        if not isinstance(operand, IRVariable):
            return None
        if operand in seen:
            return None
        seen.add(operand)

        producer = self.dfg.get_producing_instruction(operand)
        if producer is None:
            return None

        if producer.opcode == "assign":
            return self._try_const_uint(producer.operands[-1], seen)

        if producer.opcode == "phi":
            values = []
            for _label, phi_operand in producer.phi_operands:
                folded = self._try_const_uint(phi_operand, seen.copy())
                if folded is None:
                    return None
                values.append(folded)
            if not values:
                return None
            first = values[0]
            return first if all(v == first for v in values[1:]) else None

        if len(producer.operands) == 1:
            a = self._try_const_uint(producer.operands[0], seen.copy())
            if a is None:
                return None
            if producer.opcode == "not":
                return wrap256(~a)
            return None

        if len(producer.operands) == 2:
            a = self._try_const_uint(producer.operands[0], seen.copy())
            b = self._try_const_uint(producer.operands[1], seen.copy())
            if a is None or b is None:
                return None

            # Internal operand convention for non-commutative ops:
            # lhs is operands[1], rhs is operands[0].
            if producer.opcode == "add":
                return wrap256(a + b)
            if producer.opcode == "mul":
                return wrap256(a * b)
            if producer.opcode == "and":
                return wrap256(a & b)
            if producer.opcode == "or":
                return wrap256(a | b)
            if producer.opcode == "xor":
                return wrap256(a ^ b)
            if producer.opcode == "sub":
                return wrap256(b - a)
            if producer.opcode == "div":
                return 0 if a == 0 else wrap256(b // a)
            if producer.opcode == "mod":
                return 0 if a == 0 else wrap256(b % a)
            if producer.opcode == "shl":
                return wrap256(a << b)
            if producer.opcode == "shr":
                if b >= 256:
                    return 0
                return wrap256(a >> b)

        return None

    def _is_address_mask_operand(self, operand: IROperand, seen: set) -> bool:
        """
        Return True if `operand` is provably equal to the canonical 160-bit mask.
        """
        if self._is_address_mask_literal(operand):
            return True

        folded = self._try_const_uint(operand, seen.copy())
        if folded is not None and folded == ADDRESS_MASK_160:
            return True

        if not isinstance(operand, IRVariable):
            return False
        if operand in seen:
            return False
        seen.add(operand)

        producer = self.dfg.get_producing_instruction(operand)
        if producer is None:
            return False

        # Exact-value proof from range analysis (if available).
        op_range = self.range_analysis.get_range(operand, producer)
        if (
            not op_range.is_top
            and not op_range.is_empty
            and op_range.lo == ADDRESS_MASK_160
            and op_range.hi == ADDRESS_MASK_160
        ):
            return True

        if producer.opcode == "assign":
            return self._is_address_mask_operand(producer.operands[-1], seen)

        if producer.opcode == "phi":
            has_inputs = False
            for _label, phi_operand in producer.phi_operands:
                has_inputs = True
                if not self._is_address_mask_operand(phi_operand, seen.copy()):
                    return False
            return has_inputs

        return False

    def _is_address_clean(self, operand: IROperand, at_inst: IRInstruction, seen: set) -> bool:
        """
        Return True if `operand` is provably in [0, 2**160 - 1] at `at_inst`.

        This combines range facts with a small provenance walk through DFG so we
        can eliminate redundant `and(mask160, x)` patterns that survive earlier
        simplifications.
        """
        if isinstance(operand, IRLiteral):
            return wrap256(operand.value) <= ADDRESS_MASK_160
        if not isinstance(operand, IRVariable):
            return False
        if operand in seen:
            return False

        seen.add(operand)

        # Flow-sensitive range proof first (most precise).
        op_range = self.range_analysis.get_range(operand, at_inst)
        if not op_range.is_top and not op_range.is_empty and op_range.lo >= 0 and op_range.hi <= ADDRESS_MASK_160:
            return True

        producer = self.dfg.get_producing_instruction(operand)
        if producer is None:
            return False

        if producer.opcode == "assign":
            return self._is_address_clean(producer.operands[-1], producer, seen)

        if producer.opcode == "and":
            # Explicit mask160 guarantees address-clean result.
            if any(self._is_address_mask_operand(op, set()) for op in producer.operands):
                return True
            # Any AND with a <=160-bit literal keeps result <=160-bit.
            for op in producer.operands:
                folded = self._try_const_uint(op, set())
                if folded is not None and folded <= ADDRESS_MASK_160:
                    return True
            # If either operand is already address-clean, result is also address-clean.
            # Bitwise AND cannot introduce bits outside the clean operand.
            return any(self._is_address_clean(op, producer, seen.copy()) for op in producer.operands)

        if producer.opcode == "shr":
            # Logical right shift by >=96 keeps at most 160 low bits.
            shift = producer.operands[-1]
            if isinstance(shift, IRLiteral) and wrap256(shift.value) >= 96:
                return True
            # If input is already address-clean, any logical right shift preserves that.
            if len(producer.operands) >= 2:
                return self._is_address_clean(producer.operands[-2], producer, seen.copy())
            return False

        if producer.opcode == "mload" and self._mload_from_clean_store(producer, seen.copy()):
            return True

        # bool-producing ops are always 0/1 (thus 160-bit clean).
        if producer.opcode in ("iszero", "eq", "lt", "gt", "slt", "sgt"):
            return True

        # byte(...) is always in [0, 255].
        if producer.opcode == "byte":
            return True

        if producer.opcode == "mod":
            # mod(divisor, x): result is in [0, divisor-1] if divisor != 0.
            divisor = self._try_const_uint(producer.operands[0], set())
            if divisor is not None and divisor <= ADDRESS_SPACE_160:
                return True

        if producer.opcode in ("addmod", "mulmod"):
            # addmod/modulus and mulmod/modulus are bounded by modulus.
            modulus = self._try_const_uint(producer.operands[0], set())
            if modulus is not None and modulus <= ADDRESS_SPACE_160:
                return True

        # Address-like opcodes already return 160-bit values.
        if producer.opcode in ("address", "caller", "origin", "coinbase", "create", "create2"):
            return True

        # Conservative phi handling: only clean if all incoming values are clean.
        if producer.opcode == "phi":
            for _label, phi_operand in producer.phi_operands:
                if not self._is_address_clean(phi_operand, producer, seen.copy()):
                    return False
            return True

        return False

    def _mload_from_clean_store(self, mload_inst: IRInstruction, seen: set) -> bool:
        """
        Prove that an mload result is address-clean by finding a dominating
        same-block store to the exact same pointer with no intervening memory writes.
        """
        if mload_inst.opcode != "mload" or len(mload_inst.operands) != 1:
            return False

        bb = mload_inst.parent
        ptr = mload_inst.operands[0]
        read_loc = self.base_ptr.get_read_location(mload_inst, MEMORY)
        if not read_loc.is_fixed:
            return False
        if read_loc.size != 32:
            return False
        try:
            inst_idx = bb.instructions.index(mload_inst)
        except ValueError:  # pragma: nocover
            return False

        for prior in reversed(bb.instructions[:inst_idx]):
            if (prior.get_write_effects() & Effects.MEMORY) == Effects(0):
                continue

            write_loc = self.base_ptr.get_write_location(prior, MEMORY)
            if prior.opcode == "mstore" and len(prior.operands) >= 2 and prior.operands[1] == ptr:
                if write_loc.completely_contains(read_loc):
                    stored_val = prior.operands[0]
                    return self._is_address_clean(stored_val, prior, seen.copy())

            # Intervening writes only block proof when they may alias the read slot.
            if self.mem_alias.may_alias(read_loc, write_loc):
                return False

        return False

    def _algebraic_opt(self):
        self._algebraic_opt_pass()

    def _algebraic_opt_pass(self):
        for bb in self.function.get_basic_blocks():
            for inst in bb.instructions:
                self._handle_inst_peephole(inst)
                self._flip_inst(inst)

    def _flip_inst(self, inst: IRInstruction):
        ops = inst.operands
        # improve code. this seems like it should be properly handled by
        # better heuristics in DFT pass.
        if inst.flippable and self._is_lit(ops[0]) and not self._is_lit(ops[1]):
            inst.flip()

    # "peephole", weakening algebraic optimizations
    def _handle_inst_peephole(self, inst: IRInstruction):
        if inst.num_outputs != 1:
            return
        inst_out = inst.output
        if inst.is_volatile:
            return
        if inst.opcode == "assign":
            return
        if inst.is_pseudo:
            return

        # TODO nice to have rules:
        # -1 * x => 0 - x
        # x // -1 => 0 - x (?)
        # x + (-1) => x - 1  # save codesize, maybe for all negative numbers)
        # 1 // x => x == 1(?)
        # 1 % x => x > 1(?)
        # !!x => x > 0  # saves 1 gas as of shanghai

        operands = inst.operands

        # make logic easier for commutative instructions.
        if inst.flippable and self._is_lit(operands[1]) and not self._is_lit(operands[0]):
            inst.flip()
            operands = inst.operands

        if inst.opcode in {"shl", "shr", "sar"}:
            # (x >> 0) == (x << 0) == x
            if lit_eq(operands[1], 0):
                self.updater.mk_assign(inst, operands[0])
                return
            # no more cases for these instructions
            return

        if inst.opcode == "signextend":
            # text: signextend n, x -> operands[-1]=n (bytes), operands[-2]=x (value)
            n_op = operands[-1]  # byte count
            x_op = operands[-2]  # value

            # signextend(n, x) where n >= 31 is always a no-op
            if self._is_lit(n_op) and n_op.value >= 31:
                self.updater.mk_assign(inst, x_op)
                return

            # Range-based elimination: if x is already in the valid signed range
            # for (n+1) bytes, signextend is a no-op
            if self._is_lit(n_op):
                n = n_op.value
                if 0 <= n < 31:
                    x_range = self.range_analysis.get_range(x_op, inst)
                    if not x_range.is_top:
                        # Compute valid signed range for (n+1) bytes
                        bits = 8 * (n + 1)
                        signed_min = -(1 << (bits - 1))
                        signed_max = (1 << (bits - 1)) - 1
                        # If x is already in valid range, signextend is no-op
                        if x_range.lo >= signed_min and x_range.hi <= signed_max:
                            self.updater.mk_assign(inst, x_op)
                            return

            # signextend(n, signextend(m, x)) where n >= m -> signextend(m, x)
            if isinstance(x_op, IRVariable):
                producer = self.dfg.get_producing_instruction(x_op)
                if producer is not None and producer.opcode == "signextend":
                    inner_n = producer.operands[-1]  # inner byte count
                    if self._is_lit(n_op) and self._is_lit(inner_n):
                        if n_op.value >= inner_n.value:
                            self.updater.mk_assign(inst, x_op)
                            return
            return

        if inst.opcode == "exp":
            # x ** 0 -> 1
            if lit_eq(operands[0], 0):
                self.updater.mk_assign(inst, IRLiteral(1))
                return

            # 1 ** x -> 1
            if lit_eq(operands[1], 1):
                self.updater.mk_assign(inst, IRLiteral(1))
                return

            # 0 ** x -> iszero x
            if lit_eq(operands[1], 0):
                self.updater.update(inst, "iszero", [operands[0]])
                return

            # x ** 1 -> x
            if lit_eq(operands[0], 1):
                self.updater.mk_assign(inst, operands[1])
                return

            # no more cases for this instruction
            return

        if inst.opcode == "gep":
            if lit_eq(inst.operands[1], 0):
                self.updater.mk_assign(inst, inst.operands[0])
            return

        if inst.opcode in {"add", "sub", "xor"}:
            # (x - x) == (x ^ x) == 0
            if inst.opcode in ("xor", "sub") and operands[0] == operands[1]:
                self.updater.mk_assign(inst, IRLiteral(0))
                return

            # (x + 0) == (0 + x)  -> x
            # x - 0 -> x
            # (x ^ 0) == (0 ^ x)  -> x
            if lit_eq(operands[0], 0):
                self.updater.mk_assign(inst, operands[1])
                return

            # x + (-1) -> x - 1 (smaller immediate)
            if inst.opcode == "add" and lit_eq(operands[0], -1):
                self.updater.update(inst, "sub", [IRLiteral(1), operands[1]])
                return

            # (-1) - x -> ~x
            # from two's complement
            if inst.opcode == "sub" and lit_eq(operands[1], -1):
                self.updater.update(inst, "not", [operands[0]])
                return

            # x ^ 0xFFFF..FF -> ~x
            if inst.opcode == "xor" and lit_eq(operands[0], -1):
                self.updater.update(inst, "not", [operands[1]])
                return

            return

        # x & 0xFF..FF -> x
        if inst.opcode == "and" and lit_eq(operands[0], -1):
            self.updater.mk_assign(inst, operands[1])
            return

        # address-mask redundancy:
        # and(0xffffffffffffffffffffffffffffffffffffffff, x) -> x
        # when x is provably already 160-bit clean.
        if inst.opcode == "and":
            if self._is_address_mask_operand(operands[0], set()) and self._is_address_clean(
                operands[1], inst, set()
            ):
                self.updater.mk_assign(inst, operands[1])
                return
            if self._is_address_mask_operand(operands[1], set()) and self._is_address_clean(
                operands[0], inst, set()
            ):
                self.updater.mk_assign(inst, operands[0])
                return

        if inst.opcode in ("mul", "and", "div", "sdiv", "mod", "smod"):
            # (x * 0) == (x & 0) == (x // 0) == (x % 0) -> 0
            if any(lit_eq(op, 0) for op in operands):
                self.updater.mk_assign(inst, IRLiteral(0))
                return

        if inst.opcode in {"mul", "div", "sdiv", "mod", "smod"}:
            # (-1) * x -> 0 - x (smaller immediate)
            if inst.opcode == "mul" and lit_eq(operands[0], -1):
                self.updater.update(inst, "sub", [operands[1], IRLiteral(0)])
                return

            if inst.opcode in ("mod", "smod") and lit_eq(operands[0], 1):
                # x % 1 -> 0
                self.updater.mk_assign(inst, IRLiteral(0))
                return

            # (x * 1) == (1 * x) == (x // 1)  -> x
            if inst.opcode in ("mul", "div", "sdiv") and lit_eq(operands[0], 1):
                self.updater.mk_assign(inst, operands[1])
                return

            if self._is_lit(operands[0]) and is_power_of_two(operands[0].value):
                val = operands[0].value
                # x % (2^n) -> x & (2^n - 1)
                if inst.opcode == "mod":
                    self.updater.update(inst, "and", [IRLiteral(val - 1), operands[1]])
                    return
                # x / (2^n) -> x >> n
                if inst.opcode == "div":
                    self.updater.update(inst, "shr", [operands[1], IRLiteral(int_log2(val))])
                    return
                # x * (2^n) -> x << n
                if inst.opcode == "mul":
                    self.updater.update(inst, "shl", [operands[1], IRLiteral(int_log2(val))])
                    return
            return

        uses = self.dfg.get_uses(inst_out)

        is_truthy = all(i.opcode in TRUTHY_INSTRUCTIONS for i in uses)
        prefer_iszero = all(i.opcode in ("assert", "iszero") for i in uses)

        # TODO rules like:
        # not x | not y => not (x & y)
        # x | not y => not (not x & y)

        if inst.opcode == "or":
            # x | 0xff..ff == 0xff..ff
            if any(lit_eq(op, SizeLimits.MAX_UINT256) for op in operands):
                self.updater.mk_assign(inst, IRLiteral(SizeLimits.MAX_UINT256))
                return

            # x | n -> 1 in truthy positions (if n is non zero)
            if is_truthy and self._is_lit(operands[0]) and operands[0].value != 0:
                self.updater.mk_assign(inst, IRLiteral(1))
                return

            # x | 0 -> x
            if lit_eq(operands[0], 0):
                self.updater.mk_assign(inst, operands[1])
                return

        if inst.opcode == "eq":
            # x == x -> 1
            if operands[0] == operands[1]:
                self.updater.mk_assign(inst, IRLiteral(1))
                return

            # x == 0 -> iszero x
            if lit_eq(operands[0], 0):
                self.updater.update(inst, "iszero", [operands[1]])
                return

            # eq x -1 -> iszero(~x)
            # (saves codesize, not gas)
            if lit_eq(operands[0], -1):
                var = self.updater.add_before(inst, "not", [operands[1]])
                assert var is not None  # help mypy
                self.updater.update(inst, "iszero", [var])
                return

            if prefer_iszero:
                # (eq x y) has the same truthyness as (iszero (xor x y))
                tmp = self.updater.add_before(inst, "xor", [operands[0], operands[1]])

                assert tmp is not None  # help mypy
                self.updater.update(inst, "iszero", [tmp])
                return

        if inst.opcode in COMPARATOR_INSTRUCTIONS:
            self._optimize_comparator_instruction(inst, prefer_iszero)

    def _optimize_comparator_instruction(self, inst, prefer_iszero):
        opcode, operands = inst.opcode, inst.operands
        assert opcode in COMPARATOR_INSTRUCTIONS  # sanity
        inst_out = inst.output

        # (x > x) == (x < x) -> 0
        if operands[0] == operands[1]:
            self.updater.mk_assign(inst, IRLiteral(0))
            return

        is_gt = "g" in opcode
        signed = "s" in opcode

        # Range-based comparison optimization
        # Semantics: lt a, b computes a < b; gt a, b computes a > b
        # operands[-1] = a (first in text), operands[-2] = b (second in text)
        # We can optimize when one operand is a literal and we have range info for the other
        a_op = operands[-1]  # first in text
        b_op = operands[-2]  # second in text

        if self._is_lit(a_op) and not self._is_lit(b_op):
            # a is literal, b is variable: comparing lit <?> var
            lit = a_op.value
            var_range = self.range_analysis.get_range(b_op, inst)
            if not var_range.is_top and not var_range.is_empty:
                if signed:
                    if var_range.hi <= SizeLimits.MAX_INT256:
                        lit = wrap256(lit, signed=True)
                    else:
                        lit = None
                else:
                    if var_range.lo >= 0:
                        lit = wrap256(lit)
                    else:
                        lit = None

                if lit is not None:
                    if is_gt:
                        # lit > var: always true if lit > var.hi, always false if lit <= var.lo
                        if lit > var_range.hi:
                            self.updater.mk_assign(inst, IRLiteral(1))
                            return
                        if lit <= var_range.lo:
                            self.updater.mk_assign(inst, IRLiteral(0))
                            return
                    else:
                        # lit < var: always true if lit < var.lo, always false if lit >= var.hi
                        if lit < var_range.lo:
                            self.updater.mk_assign(inst, IRLiteral(1))
                            return
                        if lit >= var_range.hi:
                            self.updater.mk_assign(inst, IRLiteral(0))
                            return

        elif self._is_lit(b_op) and not self._is_lit(a_op):
            # a is variable, b is literal: comparing var <?> lit
            lit = b_op.value
            var_range = self.range_analysis.get_range(a_op, inst)
            if not var_range.is_top and not var_range.is_empty:
                if signed:
                    if var_range.hi <= SizeLimits.MAX_INT256:
                        lit = wrap256(lit, signed=True)
                    else:
                        lit = None
                else:
                    if var_range.lo >= 0:
                        lit = wrap256(lit)
                    else:
                        lit = None

                if lit is not None:
                    if is_gt:
                        # var > lit: always true if var.lo > lit, always false if var.hi <= lit
                        if var_range.lo > lit:
                            self.updater.mk_assign(inst, IRLiteral(1))
                            return
                        if var_range.hi <= lit:
                            self.updater.mk_assign(inst, IRLiteral(0))
                            return
                    else:
                        # var < lit: always true if var.hi < lit, always false if var.lo >= lit
                        if var_range.hi < lit:
                            self.updater.mk_assign(inst, IRLiteral(1))
                            return
                        if var_range.lo >= lit:
                            self.updater.mk_assign(inst, IRLiteral(0))
                            return

        lo, hi = int_bounds(bits=256, signed=signed)

        if not isinstance(operands[0], IRLiteral):
            return

        # for comparison operators, we have three special boundary cases:
        # almost always, never and almost never.
        # almost_always is always true for the non-strict ("ge" and co)
        # comparators. for strict comparators ("gt" and co), almost_always
        # is true except for one case. never is never true for the strict
        # comparators. never is almost always false for the non-strict
        # comparators, except for one case. and almost_never is almost
        # never true (except one case) for the strict comparators.
        if is_gt:
            almost_always, never = lo, hi
            almost_never = hi - 1
        else:
            almost_always, never = hi, lo
            almost_never = lo + 1

        if lit_eq(operands[0], never):
            self.updater.mk_assign(inst, IRLiteral(0))
            return

        if lit_eq(operands[0], almost_never):
            # (lt x 1), (gt x (MAX_UINT256 - 1)), (slt x (MIN_INT256 + 1))

            self.updater.update(inst, "eq", [operands[1], IRLiteral(never)])
            return

        # rewrites. in positions where iszero is preferred, (gt x 5) => (ge x 6)
        if prefer_iszero and lit_eq(operands[0], almost_always):
            # e.g. gt x 0, slt x MAX_INT256
            tmp = self.updater.add_before(inst, "eq", operands)
            self.updater.update(inst, "iszero", [tmp])
            return

        # since push0 was introduced in shanghai, it's potentially
        # better to actually reverse this optimization -- i.e.
        # replace iszero(iszero(x)) with (gt x 0)
        if opcode == "gt" and lit_eq(operands[0], 0):
            tmp = self.updater.add_before(inst, "iszero", [operands[1]])
            self.updater.update(inst, "iszero", [tmp])
            return

        # rewrite comparisons by either inserting or removing an `iszero`,
        # e.g. `x > N` -> `x >= (N + 1)`
        uses = self.dfg.get_uses(inst_out)
        if len(uses) != 1:
            return

        after = uses.first()
        if after.opcode not in ("iszero", "assert"):
            return

        if after.opcode == "iszero":
            # peer down the iszero chain to see if it actually makes sense
            # to remove the iszero.
            n_uses = self.dfg.get_uses(after.output)
            if len(n_uses) != 1:  # block the optimization
                return
            # "assert" inserts an iszero in assembly, so we will have
            # two iszeros in the asm. this is already optimal, so we don't
            # apply the iszero insertion
            if n_uses.first().opcode == "assert":
                return

        val = wrap256(operands[0].value, signed=signed)
        assert val != never, "unreachable"  # sanity

        if is_gt:
            val += 1
        else:
            # TODO: if resulting val is -1 (0xFF..FF), disable this
            # when optimization level == codesize
            val -= 1

        # sanity -- implied by precondition that `val != never`
        assert wrap256(val, signed=signed) == val

        new_opcode = flip_comparison_opcode(opcode)

        self.updater.update(inst, new_opcode, [IRLiteral(val), operands[1]])

        insert_iszero = after.opcode == "assert"
        if insert_iszero:
            # next instruction is an assert, so we insert an iszero so
            # that there will be two iszeros in the assembly.
            assert len(after.operands) == 1, after
            var = self.updater.add_before(after, "iszero", [inst_out])
            self.updater.update_operands(after, {after.operands[0]: var})
        else:
            # remove the iszero!
            assert len(after.operands) == 1, after
            self.updater.update(after, "assign", after.operands)

from vyper.utils import wrap256
from vyper.venom.analysis import DFGAnalysis, LivenessAnalysis
from vyper.venom.basicblock import IRLiteral
from vyper.venom.passes.base_pass import IRPass


def _push_size(value: int) -> int:
    """
    Return encoded byte size of pushing a literal on EVM stack.

    PUSH0 is 1 byte. PUSHn is 1(opcode) + n(data) bytes.
    """
    value = wrap256(value)
    if value == 0:
        return 1
    return 1 + ((value.bit_length() + 7) // 8)


class LiteralAliasPass(IRPass):
    """
    Rewire repeated large literal assignments in a block into alias chains.

    SingleUseExpansion often creates patterns like:
        %a = <big literal>
        ...
        %b = <same big literal>
        ...
        %c = <same big literal>

    This pass rewrites later definitions to alias prior results:
        %a = <big literal>
        ...
        %b = %a
        ...
        %c = %b

    Goal: reduce repeated PUSHn sites and let backend use DUP/SWAP paths.
    The transformation is strictly local (same basic block) and conservative.
    """

    def run_pass(
        self,
        min_push_bytes: int = 4,
        min_repeats: int = 2,
        max_gap: int = 80,
    ) -> None:
        changed = False

        for bb in self.function.get_basic_blocks():
            # Count candidate literal assignments in this block.
            literal_counts: dict[int, int] = {}
            for inst in bb.instructions:
                if inst.opcode != "assign":
                    continue
                if len(inst.operands) != 1:
                    continue
                op = inst.operands[0]
                if not isinstance(op, IRLiteral):
                    continue
                key = wrap256(op.value)
                literal_counts[key] = literal_counts.get(key, 0) + 1

            if not literal_counts:
                continue

            # Track the most recent SSA name for each candidate literal.
            last_var_for_literal = {}
            last_idx_for_literal = {}

            for idx, inst in enumerate(bb.instructions):
                if inst.opcode != "assign":
                    continue
                if len(inst.operands) != 1:
                    continue
                op = inst.operands[0]
                if not isinstance(op, IRLiteral):
                    continue

                lit_val = wrap256(op.value)
                if literal_counts.get(lit_val, 0) < min_repeats:
                    continue
                if _push_size(lit_val) < min_push_bytes:
                    continue

                prev_var = last_var_for_literal.get(lit_val)
                prev_idx = last_idx_for_literal.get(lit_val, -10**9)
                if prev_var is not None and (idx - prev_idx) <= max_gap:
                    inst.operands = [prev_var]
                    changed = True

                # Advance chain anchor to current assignment output.
                assert inst.output is not None
                last_var_for_literal[lit_val] = inst.output
                last_idx_for_literal[lit_val] = idx

        if changed:
            self.analyses_cache.invalidate_analysis(DFGAnalysis)
            self.analyses_cache.invalidate_analysis(LivenessAnalysis)

from vyper.venom.analysis import CFGAnalysis, DFGAnalysis, LivenessAnalysis
from vyper.venom.basicblock import IRBasicBlock, IRInstruction, IRLabel
from vyper.venom.passes.base_pass import IRPass


class BranchThreadingPass(IRPass):
    """
    Thread branch targets through trivial jump-only blocks.

    Safety constraints:
    - only thread through blocks that contain exactly one `jmp`
    - if threading enters a phi block, rewrite phi predecessor labels
      only when value mapping remains unambiguous
    """

    cfg: CFGAnalysis

    def _has_phi_entry(self, bb: IRBasicBlock) -> bool:
        for inst in bb.instructions:
            if inst.opcode == "phi":
                return True
            break
        return False

    def _is_trivial_jump_block(self, bb: IRBasicBlock) -> bool:
        return len(bb.instructions) == 1 and bb.instructions[0].opcode == "jmp"

    def _find_phi_operand_index(self, inst: IRInstruction, label: IRLabel) -> int:
        assert inst.opcode == "phi"
        for idx in range(0, len(inst.operands), 2):
            if inst.operands[idx] == label:
                return idx
        return -1

    def _rewrite_phi_predecessor(
        self, bb: IRBasicBlock, old_pred: IRLabel, new_pred: IRLabel
    ) -> bool:
        """
        Rewrite phi incoming label old_pred -> new_pred.

        If new_pred already exists in a phi:
        - allow if both incoming values are equal (drop duplicate old_pred arm)
        - reject if incoming values differ (ambiguous mapping)
        """
        assert self._has_phi_entry(bb)

        # Validate first to avoid partial mutation on failure.
        for inst in bb.instructions:
            if inst.opcode != "phi":
                break

            old_idx = self._find_phi_operand_index(inst, old_pred)
            if old_idx == -1:
                return False

            new_idx = self._find_phi_operand_index(inst, new_pred)
            if new_idx == -1:
                continue

            if inst.operands[old_idx + 1] != inst.operands[new_idx + 1]:
                return False

        # Apply rewrites.
        for inst in bb.instructions:
            if inst.opcode != "phi":
                break

            old_idx = self._find_phi_operand_index(inst, old_pred)
            assert old_idx != -1
            new_idx = self._find_phi_operand_index(inst, new_pred)

            if new_idx != -1:
                del inst.operands[old_idx : old_idx + 2]
            else:
                inst.operands[old_idx] = new_pred

        return True

    def _thread_label(self, pred: IRLabel, target: IRLabel) -> IRLabel:
        cur = target
        prev: IRLabel | None = None
        visited: set[IRLabel] = set()

        while True:
            if cur in visited:
                return target
            visited.add(cur)

            bb = self.function.get_basic_block(cur.value)
            if bb is None:
                return target
            if not self._is_trivial_jump_block(bb):
                if prev is None:
                    return target
                if self._has_phi_entry(bb):
                    if not self._rewrite_phi_predecessor(bb, prev, pred):
                        return target
                return cur

            inst: IRInstruction = bb.instructions[0]
            nxt = inst.operands[0]
            if not isinstance(nxt, IRLabel):
                return target
            prev = cur
            cur = nxt

    def run_pass(self):
        self.cfg = self.analyses_cache.request_analysis(CFGAnalysis)
        changed = False

        for bb in self.function.get_basic_blocks():
            if not bb.instructions:
                continue
            term = bb.instructions[-1]

            if term.opcode == "jmp":
                old = term.operands[0]
                if isinstance(old, IRLabel):
                    new = self._thread_label(bb.label, old)
                    if new != old:
                        term.operands[0] = new
                        changed = True
                continue

            if term.opcode != "jnz":
                continue

            for idx in (1, 2):
                old = term.operands[idx]
                if not isinstance(old, IRLabel):
                    continue
                new = self._thread_label(bb.label, old)
                if new != old:
                    term.operands[idx] = new
                    changed = True

            true_tgt = term.operands[1]
            false_tgt = term.operands[2]
            if isinstance(true_tgt, IRLabel) and true_tgt == false_tgt:
                # Branch condition is irrelevant when both edges converge.
                term.opcode = "jmp"
                term.operands = [true_tgt]
                changed = True

        if changed:
            self.analyses_cache.invalidate_analysis(CFGAnalysis)
            self.analyses_cache.invalidate_analysis(DFGAnalysis)
            self.analyses_cache.invalidate_analysis(LivenessAnalysis)

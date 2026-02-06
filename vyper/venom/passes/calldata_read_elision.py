from __future__ import annotations

from dataclasses import dataclass

from vyper.evm.address_space import MEMORY
from vyper.utils import wrap256
from vyper.venom.analysis import BasePtrAnalysis, DFGAnalysis, LivenessAnalysis, MemoryAliasAnalysis
from vyper.venom.basicblock import IRInstruction, IRLiteral
from vyper.venom.effects import Effects
from vyper.venom.memory_location import MemoryLocation
from vyper.venom.passes.base_pass import IRPass
from vyper.venom.passes.machinery.inst_updater import InstUpdater


@dataclass
class _CalldataCopy:
    dst_loc: MemoryLocation
    src_ofst: int


class CalldataReadElisionPass(IRPass):
    """
    Replace `mload` reads with `calldataload` when data is known to come from a
    dominating fixed `calldatacopy` in the same basic block.

    Phase-1 safety model (intentionally strict):
    - only fixed-size/fixed-offset copies
    - only fixed `mload` locations
    - only when copied region fully contains the loaded 32-byte word
    - invalidate tracking on any potentially-aliasing memory write
    """

    base_ptr: BasePtrAnalysis
    dfg: DFGAnalysis
    mem_alias: MemoryAliasAnalysis
    updater: InstUpdater

    def run_pass(self):
        self.base_ptr = self.analyses_cache.request_analysis(BasePtrAnalysis)
        self.mem_alias = self.analyses_cache.request_analysis(MemoryAliasAnalysis)
        self.dfg = self.analyses_cache.request_analysis(DFGAnalysis)
        self.updater = InstUpdater(self.dfg)

        changed = False
        for bb in self.function.get_basic_blocks():
            changed |= self._process_bb(bb)

        if changed:
            self.analyses_cache.invalidate_analysis(LivenessAnalysis)
            self.analyses_cache.invalidate_analysis(DFGAnalysis)

    def _process_bb(self, bb) -> bool:
        changed = False
        copies: list[_CalldataCopy] = []

        for inst in bb.instructions:
            if inst.opcode == "calldatacopy":
                write_loc = self.base_ptr.get_write_location(inst, MEMORY)
                self._invalidate_for_write(copies, write_loc)
                captured = self._capture_calldata_copy(inst, write_loc)
                if captured is not None:
                    copies.append(captured)
                continue

            if inst.opcode == "mload":
                read_loc = self.base_ptr.get_read_location(inst, MEMORY)
                copy_info = self._find_covering_copy(copies, read_loc)
                if copy_info is not None:
                    assert read_loc.offset is not None and copy_info.dst_loc.offset is not None
                    delta = read_loc.offset - copy_info.dst_loc.offset
                    src_ofst = wrap256(copy_info.src_ofst + delta)
                    self.updater.update(inst, "calldataload", [IRLiteral(src_ofst)])
                    changed = True
                continue

            if (inst.get_write_effects() & Effects.MEMORY) != Effects(0):
                write_loc = self.base_ptr.get_write_location(inst, MEMORY)
                self._invalidate_for_write(copies, write_loc)

        return changed

    def _capture_calldata_copy(
        self, inst: IRInstruction, write_loc: MemoryLocation
    ) -> _CalldataCopy | None:
        if len(inst.operands) != 3:
            return None
        size, src_ofst, _dst = inst.operands
        if not isinstance(size, IRLiteral) or not isinstance(src_ofst, IRLiteral):
            return None
        if size.value <= 0:
            return None
        if not write_loc.is_fixed:
            return None
        return _CalldataCopy(dst_loc=write_loc, src_ofst=wrap256(src_ofst.value))

    def _invalidate_for_write(self, copies: list[_CalldataCopy], write_loc: MemoryLocation) -> None:
        if not copies:
            return
        if not write_loc.is_fixed:
            copies.clear()
            return
        copies[:] = [copy for copy in copies if not self.mem_alias.may_alias(copy.dst_loc, write_loc)]

    def _find_covering_copy(
        self, copies: list[_CalldataCopy], read_loc: MemoryLocation
    ) -> _CalldataCopy | None:
        if not read_loc.is_fixed:
            return None
        if read_loc.size != 32:
            return None
        if read_loc.offset is None:
            return None

        for copy in reversed(copies):
            if copy.dst_loc.completely_contains(read_loc):
                return copy
        return None

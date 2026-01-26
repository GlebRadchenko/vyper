from vyper.venom.passes.base_pass import IRPass


class FloatAllocas(IRPass):
    """
    This pass moves allocas to the entry basic block of a function
    We could probably move them to the immediate dominator of the basic
    block defining the alloca instead of the entry (which dominates all
    basic blocks), but this is done for expedience.
    Without this step, sccp fails, possibly because dominators are not
    guaranteed to be traversed first.
    """

    def run_pass(self):
        entry_bb = self.function.entry
        assert entry_bb.is_terminated, entry_bb
        
        # Collect all allocas first (and skip entry_bb logic if none found)
        allocas_to_move = []
        for bb in self.function.get_basic_blocks():
            if bb is entry_bb:
                continue

            non_alloca_instructions = []
            for inst in bb.instructions:
                if inst.opcode in ("alloca", "palloca", "calloca"):
                    # Skip annotated allocas (pinned allocas from FixMemLocations)
                    if inst.annotation and "free var" in inst.annotation:
                        non_alloca_instructions.append(inst)
                        continue
                    
                    allocas_to_move.append(inst)
                else:
                    non_alloca_instructions.append(inst)

            # Replace original instructions with filtered list (only if changes made)
            if len(non_alloca_instructions) != len(bb.instructions):
                bb.instructions = non_alloca_instructions

        if not allocas_to_move:
            return

        # Move collected allocas to entry
        tmp = entry_bb.instructions.pop()
        for inst in allocas_to_move:
            entry_bb.insert_instruction(inst)
        entry_bb.instructions.append(tmp)
        
        # Invalidate analyses as we moved instructions
        for analysis_cls in list(self.analyses_cache.analyses_cache.keys()):
            self.analyses_cache.invalidate_analysis(analysis_cls)

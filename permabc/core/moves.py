"""
Backward-compatibility shim — real code is in permabc.sampling.moves.
"""
from ..sampling.moves import (
    move_smc,
    move_smc_gibbs_blocks,
    calculate_overall_acceptance_rate,
    create_block,
    MoveResult,
)

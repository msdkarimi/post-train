"""Trainer callbacks."""
from __future__ import annotations

import time

from transformers import TrainerCallback


class TimeBudgetCallback(TrainerCallback):
    """Stop training and force a checkpoint save once `minutes` of wall-clock time have passed.

    Rented GPUs bill by the hour; this guarantees you end with a saved checkpoint instead of an
    unfinished run when the budget is up. The LR schedule is NOT shortened, so set max_steps /
    num_epochs so the run would normally finish inside the budget.
    """

    def __init__(self, minutes: float):
        self.budget_s = minutes * 60
        self.t0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.t0 = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.t0 is not None and time.time() - self.t0 > self.budget_s:
            print(f"\n[time budget] {self.budget_s / 60:.0f} min reached at step {state.global_step}: saving and stopping")
            control.should_save = True
            control.should_training_stop = True
        return control

"""Professor Forcing baseline and the hidden-state matching ablation.

This package is opt-in. Importing it does not register a model with
``LearningAlgorithm`` and does not change the OTF training or eval pipeline.
"""

from dvae.baselines.professor_forcing.discriminator import BehaviorDiscriminator
from dvae.baselines.professor_forcing.generator import ProfessorForcingGenerator
from dvae.baselines.professor_forcing.step import hidden_match_loss, professor_forcing_loss

__all__ = [
    "BehaviorDiscriminator",
    "ProfessorForcingGenerator",
    "hidden_match_loss",
    "professor_forcing_loss",
]

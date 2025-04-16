# psi_c_ai_sdk/multi_agent/consensus_vote.py

from collections import Counter
from typing import List, Dict

class BeliefConsensus:
    @staticmethod
    def vote_on_belief_conflicts(
        belief_options: List[str],
        weights: Dict[str, float] = None
    ) -> str:
        """
        Takes a list of belief IDs or strings and votes across them.
        Optional weighting per agent.
        """
        count = Counter()
        for belief in belief_options:
            weight = weights.get(belief, 1.0) if weights else 1.0
            count[belief] += weight

        top = count.most_common(1)
        return top[0][0] if top else None

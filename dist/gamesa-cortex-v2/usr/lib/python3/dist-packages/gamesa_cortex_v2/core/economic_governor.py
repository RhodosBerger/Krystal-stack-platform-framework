import logging

from .config import GamesaConfig
from .logging_system import IntraspectralLogger

class EconomicGovernor:
    """
    Gamesa Cortex V2: Economic Governor.
    Regulates Resource Allocation based on 'Economic Planning'.
    Enforces budgets for Compute, Energy, and Time.
    """
    def __init__(self, budget_credits=None):
        self.logger = logging.getLogger("EconomicGovernor")
        self.intra_logger = IntraspectralLogger()
        self.budget_credits = budget_credits if budget_credits is not None else GamesaConfig.INITIAL_BUDGET_CREDITS
        self.cost_model = GamesaConfig.COST_MODEL
        
        self.logger.info(f"Economic Governor Online. Budget: {self.budget_credits} Credits")
        self.intra_logger.log_event("ECONOMIC", "Governor", "Online", {"budget": self.budget_credits})

    def request_allocation(self, task_type: str, priority_level: str) -> bool:
        """
        evaluates if a task affords the resource cost.
        """
        # OPTIMIZATION: Critical Path Bypass
        # If High Priority, skip the dictionary lookup and budget check latency.
        if priority_level in ["INTERDICTION_PROTOCOL", "EVOLUTIONARY_OVERDRIVE"]:
             return True

        cost = self.cost_model.get(task_type, GamesaConfig.COST_MODEL["DEFAULT"])
        
        # Regulation 2: Budget Check
        if self.budget_credits >= cost:
            self.budget_credits -= cost
            # Optimization: Only log on failure or specific debug level to save IO
            # self.logger.info(f"Approved {task_type}...") 
            return True
        else:
            self.logger.warning(f"Denied {task_type}. Insufficient Credits ({self.budget_credits} < {cost})")
            self.intra_logger.log_event("ECONOMIC", "Governor", "Task Denied", {"task": task_type, "cost": cost, "budget": self.budget_credits})
            return False

    def replentish_budget(self, amount=100):
        """
        Periodic replenishment (simulates 'Fiscal Year' or Time Window).
        """
        self.budget_credits += amount
        self.logger.info(f"Budget Replenished. Current: {self.budget_credits}")

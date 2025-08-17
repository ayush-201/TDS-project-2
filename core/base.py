import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.config import get_chat_model

logger = logging.getLogger(__name__)

class BaseWorkflow(ABC):
    """Base class for all workflows, ensuring an LLM instance is available."""
    def __init__(self, llm=None, **kwargs):
        self.llm = llm or get_chat_model()

    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the specific workflow logic."""
        pass

class WorkflowOrchestrator:
    """Orchestrates and executes the appropriate workflow based on the request."""
    def __init__(self):
        self.workflows = {}
        self.llm = get_chat_model()

    def register_workflow(self, name: str, workflow: BaseWorkflow):
        """Adds a workflow to the orchestrator's registry."""
        self.workflows[name] = workflow

    async def execute_workflow(self, workflow_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes a registered workflow by its name."""
        if workflow_type not in self.workflows:
            raise ValueError(f"Workflow '{workflow_type}' not recognized.")
        
        workflow_instance = self.workflows[workflow_type]
        return await workflow_instance.execute(input_data)

class AdvancedWorkflowOrchestrator(WorkflowOrchestrator):
    """
    The main orchestrator that registers and manages all available workflows.
    """
    def __init__(self):
        super().__init__()
        from workflows.web_scraping import MultiStepWebScrapingWorkflow
        from workflows.database_analysis import DatabaseAnalysisWorkflow
        
        self.register_workflow("multi_step_web_scraping", MultiStepWebScrapingWorkflow())
        self.register_workflow("database_analysis", DatabaseAnalysisWorkflow())
        # Register other workflows here as they are created
# LangChain workflows package
"""
This package contains LangChain workflows for data analysis tasks.
"""

from core.base import BaseWorkflow, WorkflowOrchestrator, AdvancedWorkflowOrchestrator
from .web_scraping import MultiStepWebScrapingWorkflow
from .database_analysis import DatabaseAnalysisWorkflow

__all__ = [
    "BaseWorkflow",
    "WorkflowOrchestrator",
    "AdvancedWorkflowOrchestrator",
    "MultiStepWebScrapingWorkflow",
    "DatabaseAnalysisWorkflow",
]
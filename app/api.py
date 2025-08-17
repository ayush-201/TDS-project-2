import logging
import uuid
import asyncio
from datetime import datetime
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from core.base import AdvancedWorkflowOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter()

try:
    orchestrator = AdvancedWorkflowOrchestrator()
    logger.info("✅ AdvancedWorkflowOrchestrator initialized successfully.")
    logger.info(f"   Available workflows: {list(orchestrator.workflows.keys())}")
except Exception as e:
    logger.error(f"❌ CRITICAL: Failed to initialize AdvancedWorkflowOrchestrator: {e}")
    orchestrator = None

@router.post("/")
async def analyze_data(
    questions_txt: UploadFile = File(..., description="A .txt file with the user's questions."),
    files: List[UploadFile] = File([], description="Optional additional files (e.g., CSV, images)."),
):
    """
    Main API endpoint to process data analysis tasks. It intelligently routes
    requests to the appropriate workflow (web scraping or database analysis).
    """
    if orchestrator is None:
        raise HTTPException(
            status_code=500,
            detail="Server is not configured correctly. Orchestrator could not be initialized."
        )

    task_id = str(uuid.uuid4())
    logger.info(f"🚀 Starting task {task_id}")

    try:
        # Read and decode the main questions file
        questions_content = await questions_txt.read()
        task_description = questions_content.decode("utf-8")
        logger.info("   - questions.txt processed successfully.")

        # --- Intelligent Workflow Detection ---
        # Decide which workflow to use based on the content of the request.
        if "duckdb" in task_description.lower() or "sql" in task_description.lower() or "s3://" in task_description.lower():
            workflow_type = "database_analysis"
        else:
            workflow_type = "multi_step_web_scraping"
        
        logger.info(f"   - Detected workflow type: {workflow_type}")

        workflow_input = {
            "task_description": task_description,
            # Pass other necessary data here if needed by workflows
        }

        # Execute the selected workflow with a 3-minute timeout
        result = await asyncio.wait_for(
            orchestrator.execute_workflow(workflow_type, workflow_input),
            timeout=300
        )

        logger.info(f"✅ Task {task_id} completed successfully.")
        return {
            "task_id": task_id,
            "status": "completed",
            "workflow_type": workflow_type,
            "result": result,
            "timestamp": datetime.now().isoformat(),
        }

    except asyncio.TimeoutError:
        logger.error(f"❌ Task {task_id} timed out after 3 minutes.")
        raise HTTPException(status_code=408, detail="Request timed out after 3 minutes.")
    except Exception as e:
        logger.error(f"❌ Task {task_id} failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
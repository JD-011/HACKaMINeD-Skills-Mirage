"""
Chat & Plan API endpoints.
"""

from fastapi import APIRouter, Depends

from app.models.schemas import (
    ChatRequest, ChatResponse, MemoryResponse,
    PlanRequest, PlanResponse,
)
from app.api.dependencies import get_chat_service, get_plan_service
from app.services.chat_service import ChatService
from app.services.plan_service import PlanService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    """Send a message and get an intelligent, context-aware response.

    The agentic backend decides which data sources to query (live job
    market data, course database, or both) based on the user's query.
    """
    result = await service.chat(query=request.query, user=request.user)
    return ChatResponse(**result)


@router.post("/plan", response_model=PlanResponse)
async def generate_plan(
    request: PlanRequest,
    service: PlanService = Depends(get_plan_service),
):
    """Generate a personalised upskilling plan.

    Fetches live Layer-1 market data (vulnerability scores, skill trends,
    watchlist alerts) and relevant courses, then produces an actionable
    career transition plan tailored to the user's profile.
    """
    result = await service.generate_plan(
        user=request.user, preferences=request.preferences
    )
    return PlanResponse(**result)


@router.get("/memories/{user_id}", response_model=MemoryResponse)
async def get_memories(
    user_id: str,
    service: ChatService = Depends(get_chat_service),
):
    """Retrieve all stored memories for a given user."""
    memories = await service.get_memories(user_id=user_id)
    return MemoryResponse(memories=memories)


@router.delete("/memories/{user_id}")
async def reset_memory(
    user_id: str,
    service: ChatService = Depends(get_chat_service),
):
    """Delete all memories for a given user."""
    await service.reset_memory(user_id=user_id)
    return {"message": f"Memory reset for user '{user_id}'"}

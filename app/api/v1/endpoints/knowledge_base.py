from fastapi import APIRouter, HTTPException, status, Body
from ...models.knowledge_base import KnowledgeBase
from typing import List, Optional

router = APIRouter()

# In-memory store for demonstration (replace with DB in production)
knowledge_bases: List[KnowledgeBase] = []

@router.post("/knowledge_bases", response_model=KnowledgeBase, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(kb: KnowledgeBase):
    knowledge_bases.append(kb)
    return kb

@router.get("/knowledge_bases", response_model=List[KnowledgeBase])
def list_knowledge_bases():
    return knowledge_bases

@router.get("/knowledge_bases/{kb_id}", response_model=KnowledgeBase)
def get_knowledge_base(kb_id: str):
    for kb in knowledge_bases:
        if kb.id == kb_id:
            return kb
    raise HTTPException(status_code=404, detail="KnowledgeBase not found")

@router.put("/knowledge_bases/{kb_id}", response_model=KnowledgeBase)
def update_knowledge_base(kb_id: str, kb_update: KnowledgeBase):
    for i, kb in enumerate(knowledge_bases):
        if kb.id == kb_id:
            knowledge_bases[i] = kb_update
            return kb_update
    raise HTTPException(status_code=404, detail="KnowledgeBase not found")

@router.delete("/knowledge_bases/{kb_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_knowledge_base(kb_id: str):
    for i, kb in enumerate(knowledge_bases):
        if kb.id == kb_id:
            del knowledge_bases[i]
            return
    raise HTTPException(status_code=404, detail="KnowledgeBase not found")

from sqlalchemy import Column, String, Integer, Text, DateTime, Table, ForeignKey, Boolean
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.types import JSON
import datetime

Base = declarative_base()

# Association table for many-to-many relationship between documents and KBs
document_kb_association = Table(
    'document_kb_association', Base.metadata,
    Column('document_id', String, ForeignKey('documents.id')),
    Column('kb_id', String, ForeignKey('knowledge_bases.id'))
)

class KnowledgeBaseDB(Base):
    __tablename__ = 'knowledge_bases'
    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    chunk_size = Column(Integer, default=1000)
    chunk_overlap = Column(Integer, default=200)
    embedding_model = Column(String)
    retriever_params = Column(JSON)
    prompt_template = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    documents = relationship('DocumentDB', secondary=document_kb_association, back_populates='knowledge_bases')

class DocumentDB(Base):
    __tablename__ = 'documents'
    id = Column(String, primary_key=True, index=True)
    file_path = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    metadata = Column(JSON)
    version = Column(String)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default='active')
    tags = Column(JSON)
    chunk_size = Column(Integer)
    chunk_overlap = Column(Integer)
    embedding_model = Column(String)
    is_deleted = Column(Boolean, default=False)
    knowledge_bases = relationship('KnowledgeBaseDB', secondary=document_kb_association, back_populates='documents')

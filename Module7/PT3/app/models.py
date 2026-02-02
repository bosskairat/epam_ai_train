from enum import Enum
from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class OrderStatus(str, Enum):
    created = "created"
    processing = "processing"
    completed = "completed"
    cancelled = "cancelled"


class Order(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    status: OrderStatus = Field(sa_column_kwargs={"nullable": False})
    amount: float = Field(default=0.0, sa_column_kwargs={"nullable": False})
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"nullable": False})

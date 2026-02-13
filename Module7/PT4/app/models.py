from enum import Enum
from datetime import datetime, timezone
from typing import Optional, List
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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_column_kwargs={"nullable": False})


class PaginatedOrders(SQLModel):
    items: List[Order]
    total: int
    page: int
    limit: int
    pages: int
    has_next: bool
    has_prev: bool

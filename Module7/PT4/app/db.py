from sqlmodel import create_engine, Session, SQLModel, select
from typing import Generator
from .models import Order
import os
import random
from datetime import datetime, timezone, timedelta

DB_FILE = os.path.join(os.path.dirname(__file__), "../orders.db")
DB_URL = f"sqlite:///{DB_FILE}"
engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False})


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session


def init_db():
    SQLModel.metadata.create_all(engine)


def seed_data_if_empty(count: int = 50):
    init_db()
    with Session(engine) as session:
        q = session.exec(select(Order).limit(1)).all()
        if q:
            return
        statuses = [s for s in Order.__fields__["status"].type_.__members__.keys()] if False else None
        # fallback statuses
        statuses = ["created", "processing", "completed", "cancelled"]
        now = datetime.now(timezone.utc)
        for i in range(count):
            order = Order(
                status=random.choice(statuses),
                amount=round(random.uniform(5, 500), 2),
                created_at=now - timedelta(days=random.randint(0, 365))
            )
            session.add(order)
        session.commit()

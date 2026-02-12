from fastapi import FastAPI, Depends, HTTPException, Query
from typing import List, Optional
from sqlmodel import Session, select
from .models import Order, OrderStatus
from .db import get_session, init_db, seed_data_if_empty
from datetime import datetime

app = FastAPI(title="Orders API")

# initialize DB and seed
init_db()
seed_data_if_empty(50)


@app.post("/orders", response_model=Order)
def create_order(order: Order, session: Session = Depends(get_session)):
    if order.amount < 0:
        raise HTTPException(status_code=400, detail="Amount must be non-negative")
    if not isinstance(order.status, OrderStatus):
        try:
            order.status = OrderStatus(order.status)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid status")
    session.add(order)
    session.commit()
    session.refresh(order)
    return order


@app.get("/orders", response_model=List[Order])
def list_orders(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    status: Optional[OrderStatus] = None,
    min_amount: Optional[float] = Query(None, ge=0),
    max_amount: Optional[float] = Query(None, ge=0),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    session: Session = Depends(get_session),
):
    stmt = select(Order)
    if status:
        stmt = stmt.where(Order.status == status)
    if min_amount is not None:
        stmt = stmt.where(Order.amount >= min_amount)
    if max_amount is not None:
        stmt = stmt.where(Order.amount <= max_amount)
    # date parsing
    try:
        if start_date:
            sd = datetime.fromisoformat(start_date)
            stmt = stmt.where(Order.created_at >= sd)
        if end_date:
            ed = datetime.fromisoformat(end_date)
            stmt = stmt.where(Order.created_at <= ed)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS")

    # pagination
    offset = (page - 1) * limit
    stmt = stmt.offset(offset).limit(limit)
    results = session.exec(stmt).all()
    return results

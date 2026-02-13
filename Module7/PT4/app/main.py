from fastapi import FastAPI, Depends, HTTPException, Query
from typing import Optional
from sqlmodel import Session, select, func
from .models import Order, OrderStatus, PaginatedOrders
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


@app.get("/orders", response_model=PaginatedOrders)
def list_orders(
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    limit: int = Query(10, ge=1, le=100, description="Number of items per page (max 100)"),
    status: Optional[OrderStatus] = Query(None, description="Filter by order status"),
    min_amount: Optional[float] = Query(None, ge=0, description="Minimum amount filter"),
    max_amount: Optional[float] = Query(None, ge=0, description="Maximum amount filter"),
    start_date: Optional[str] = Query(None, description="Start date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"),
    end_date: Optional[str] = Query(None, description="End date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)"),
    session: Session = Depends(get_session),
):
    # Build base query
    stmt = select(Order)
    count_stmt = select(func.count(Order.id))
    
    # Apply filters
    if status:
        stmt = stmt.where(Order.status == status)
        count_stmt = count_stmt.where(Order.status == status)
    
    if min_amount is not None:
        stmt = stmt.where(Order.amount >= min_amount)
        count_stmt = count_stmt.where(Order.amount >= min_amount)
    
    if max_amount is not None:
        stmt = stmt.where(Order.amount <= max_amount)
        count_stmt = count_stmt.where(Order.amount <= max_amount)
    
    # Date parsing and filtering
    try:
        if start_date is not None and start_date.strip():
            sd = datetime.fromisoformat(start_date)
            stmt = stmt.where(Order.created_at >= sd)
            count_stmt = count_stmt.where(Order.created_at >= sd)
        
        if end_date is not None and end_date.strip():
            ed = datetime.fromisoformat(end_date)
            stmt = stmt.where(Order.created_at <= ed)
            count_stmt = count_stmt.where(Order.created_at <= ed)
            
        # Validate date range
        if (start_date is not None and start_date.strip() and 
            end_date is not None and end_date.strip()):
            if sd > ed:
                raise HTTPException(
                    status_code=400, 
                    detail="start_date must be less than or equal to end_date"
                )
                
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail="Invalid date format. Use ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"
        )

    # Get total count
    total = session.exec(count_stmt).one()
    
    # Calculate pagination
    offset = (page - 1) * limit
    total_pages = (total + limit - 1) // limit
    
    # Apply pagination
    stmt = stmt.offset(offset).limit(limit)
    results = session.exec(stmt).all()
    
    # Build response
    return PaginatedOrders(
        items=results,
        total=total,
        page=page,
        limit=limit,
        pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1
    )

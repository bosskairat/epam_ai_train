"""
SQLModel Database Query Examples with Pagination
Demonstrates various query patterns for the Orders API
"""

from sqlmodel import Session, select, func, and_, or_
from datetime import datetime, timedelta
from typing import Optional
from app.models import Order, OrderStatus
from app.db import engine

def basic_pagination_query(page: int = 1, limit: int = 10):
    """Basic pagination query without filters"""
    with Session(engine) as session:
        # Calculate offset
        offset = (page - 1) * limit
        
        # Data query with pagination
        stmt = select(Order).offset(offset).limit(limit)
        orders = session.exec(stmt).all()
        
        # Count query for total
        count_stmt = select(func.count(Order.id))
        total = session.exec(count_stmt).one()
        
        return orders, total

def filtered_pagination_query(
    page: int = 1,
    limit: int = 10,
    status: Optional[OrderStatus] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Advanced query with filters and pagination"""
    with Session(engine) as session:
        # Build base queries
        stmt = select(Order)
        count_stmt = select(func.count(Order.id))
        
        # Apply filters to both queries
        if status:
            stmt = stmt.where(Order.status == status)
            count_stmt = count_stmt.where(Order.status == status)
        
        if min_amount is not None:
            stmt = stmt.where(Order.amount >= min_amount)
            count_stmt = count_stmt.where(Order.amount >= min_amount)
        
        if max_amount is not None:
            stmt = stmt.where(Order.amount <= max_amount)
            count_stmt = count_stmt.where(Order.amount <= max_amount)
        
        # Date filtering
        if start_date:
            sd = datetime.fromisoformat(start_date)
            stmt = stmt.where(Order.created_at >= sd)
            count_stmt = count_stmt.where(Order.created_at >= sd)
        
        if end_date:
            ed = datetime.fromisoformat(end_date)
            stmt = stmt.where(Order.created_at <= ed)
            count_stmt = count_stmt.where(Order.created_at <= ed)
        
        # Get total count first
        total = session.exec(count_stmt).one()
        
        # Apply pagination to data query
        offset = (page - 1) * limit
        stmt = stmt.offset(offset).limit(limit)
        orders = session.exec(stmt).all()
        
        return orders, total

def complex_filter_query():
    """Example of complex filtering with multiple conditions"""
    with Session(engine) as session:
        # Complex query: orders that are either completed OR processing,
        # with amount between 50-200, created in last 30 days
        thirty_days_ago = datetime.utcnow() - timedelta(days=30)
        
        stmt = select(Order).where(
            and_(
                or_(
                    Order.status == OrderStatus.completed,
                    Order.status == OrderStatus.processing
                ),
                Order.amount.between(50, 200),
                Order.created_at >= thirty_days_ago
            )
        ).order_by(Order.created_at.desc())
        
        orders = session.exec(stmt).all()
        return orders

def aggregated_query():
    """Example of aggregated queries for statistics"""
    with Session(engine) as session:
        # Count orders by status
        status_counts = session.exec(
            select(
                Order.status,
                func.count(Order.id).label('count'),
                func.avg(Order.amount).label('avg_amount'),
                func.sum(Order.amount).label('total_amount')
            ).group_by(Order.status)
        ).all()
        
        return status_counts

def sorted_query(
    sort_by: str = "created_at",
    sort_order: str = "desc"
):
    """Query with dynamic sorting"""
    with Session(engine) as session:
        stmt = select(Order)
        
        # Apply sorting based on parameters
        if sort_by == "created_at":
            if sort_order == "desc":
                stmt = stmt.order_by(Order.created_at.desc())
            else:
                stmt = stmt.order_by(Order.created_at.asc())
        elif sort_by == "amount":
            if sort_order == "desc":
                stmt = stmt.order_by(Order.amount.desc())
            else:
                stmt = stmt.order_by(Order.amount.asc())
        elif sort_by == "status":
            if sort_order == "desc":
                stmt = stmt.order_by(Order.status.desc())
            else:
                stmt = stmt.order_by(Order.status.asc())
        
        orders = session.exec(stmt).all()
        return orders

def search_query(search_term: str):
    """Example of text search (if you had searchable fields)"""
    with Session(engine) as session:
        # This would work if you had text fields to search
        # For demonstration, showing the pattern
        stmt = select(Order).where(
            # Example: Order.description.ilike(f"%{search_term}%")
            Order.status == OrderStatus.created  # placeholder
        )
        orders = session.exec(stmt).all()
        return orders

def date_range_query(start_date: str, end_date: str):
    """Optimized date range query"""
    with Session(engine) as session:
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Efficient date range query
        stmt = select(Order).where(
            and_(
                Order.created_at >= start_dt,
                Order.created_at <= end_dt
            )
        ).order_by(Order.created_at.asc())
        
        orders = session.exec(stmt).all()
        return orders

# Example usage functions
def demonstrate_queries():
    """Demonstrate all query patterns"""
    print("=== Basic Pagination ===")
    orders, total = basic_pagination_query(page=1, limit=5)
    print(f"Found {total} total orders, showing first 5")
    
    print("\n=== Filtered Pagination ===")
    orders, total = filtered_pagination_query(
        page=1, 
        limit=5, 
        status=OrderStatus.completed,
        min_amount=100
    )
    print(f"Found {total} completed orders over $100")
    
    print("\n=== Complex Filter ===")
    orders = complex_filter_query()
    print(f"Found {len(orders)} orders with complex criteria")
    
    print("\n=== Aggregated Stats ===")
    stats = aggregated_query()
    for status, count, avg_amount, total_amount in stats:
        print(f"{status}: {count} orders, avg: ${avg_amount:.2f}, total: ${total_amount:.2f}")
    
    print("\n=== Sorted Query ===")
    orders = sorted_query(sort_by="amount", sort_order="desc")
    print(f"Top 3 orders by amount: {[(o.id, o.amount) for o in orders[:3]]}")

if __name__ == "__main__":
    demonstrate_queries()

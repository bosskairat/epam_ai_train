"""
Comprehensive tests for pagination and filtering functionality
Tests the enhanced GET /orders endpoint with PaginatedOrders response
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db import engine
from sqlmodel import SQLModel, Session, select, func, delete
from datetime import datetime, timezone, timedelta
from app.models import Order, OrderStatus

client = TestClient(app)


def reset_db():
    """Reset database for clean test state"""
    # Make reset deterministic across tests even with a globally-imported FastAPI app
    # that seeded data at import time.
    try:
        engine.dispose()
    except Exception:
        pass

    # Ensure schema exists, then clear rows.
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.exec(delete(Order))
        session.commit()


@pytest.fixture(autouse=True)
def run_around_tests():
    """Setup and teardown for each test"""
    reset_db()
    yield


def create_test_orders():
    """Create test orders with known data for predictable tests"""
    with Session(engine) as session:
        now = datetime.now(timezone.utc)
        
        # Create orders with different statuses, amounts, and dates
        test_orders = [
            Order(status=OrderStatus.created, amount=10.0, created_at=now - timedelta(days=10)),
            Order(status=OrderStatus.processing, amount=25.0, created_at=now - timedelta(days=8)),
            Order(status=OrderStatus.completed, amount=50.0, created_at=now - timedelta(days=5)),
            Order(status=OrderStatus.completed, amount=75.0, created_at=now - timedelta(days=3)),
            Order(status=OrderStatus.cancelled, amount=100.0, created_at=now - timedelta(days=1)),
            Order(status=OrderStatus.created, amount=150.0, created_at=now - timedelta(hours=12)),
            Order(status=OrderStatus.processing, amount=200.0, created_at=now - timedelta(hours=6)),
            Order(status=OrderStatus.completed, amount=300.0, created_at=now - timedelta(hours=1)),
        ]
        
        for order in test_orders:
            session.add(order)
        session.commit()


class TestPaginationDefaults:
    """Test pagination default parameters"""
    
    def test_default_pagination_response_structure(self):
        """Test response has correct structure with defaults"""
        create_test_orders()
        response = client.get("/orders")
        assert response.status_code == 200
        
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert "limit" in data
        assert "pages" in data
        assert "has_next" in data
        assert "has_prev" in data
        
        # Check default values
        assert data["page"] == 1
        assert data["limit"] == 10
        assert data["has_prev"] == False
    
    def test_default_page_and_limit_values(self):
        """Test default page=1, limit=10"""
        create_test_orders()
        response = client.get("/orders")
        data = response.json()
        
        assert data["page"] == 1
        assert data["limit"] == 10
        assert len(data["items"]) <= 10
    
    def test_pagination_calculation_accuracy(self):
        """Test pagination metadata calculations"""
        create_test_orders()
        response = client.get("/orders?limit=3")
        data = response.json()
        
        expected_pages = (data["total"] + 3 - 1) // 3  # Ceiling division
        assert data["pages"] == expected_pages
        assert data["has_next"] == (data["page"] < data["pages"])
        assert data["has_prev"] == (data["page"] > 1)


class TestCustomPagination:
    """Test custom pagination parameters"""
    
    def test_custom_page_and_limit(self):
        """Test custom page and limit values"""
        create_test_orders()
        response = client.get("/orders?page=2&limit=3")
        data = response.json()
        
        assert data["page"] == 2
        assert data["limit"] == 3
        assert len(data["items"]) <= 3
    
    def test_pagination_edge_cases(self):
        """Test pagination edge cases"""
        create_test_orders()
        
        # First page
        response = client.get("/orders?page=1&limit=5")
        data = response.json()
        assert data["page"] == 1
        assert data["has_prev"] == False
        
        # Last page
        total_pages = data["pages"]
        response = client.get(f"/orders?page={total_pages}&limit=5")
        data = response.json()
        assert data["page"] == total_pages
        assert data["has_next"] == False
    
    def test_empty_page_beyond_range(self):
        """Test requesting page beyond available data"""
        create_test_orders()
        response = client.get("/orders?page=999&limit=10")
        data = response.json()
        
        assert data["page"] == 999
        assert len(data["items"]) == 0
        assert data["has_next"] == False
        assert data["has_prev"] == True


class TestFilteringByStatus:
    """Test status filtering"""
    
    def test_filter_single_status(self):
        """Test filtering by single status"""
        create_test_orders()
        response = client.get("/orders?status=completed")
        data = response.json()
        
        assert all(item["status"] == "completed" for item in data["items"])
        assert data["total"] > 0
    
    def test_filter_nonexistent_status(self):
        """Test filtering by status with no matching orders"""
        create_test_orders()
        response = client.get("/orders?status=processing")
        data = response.json()
        
        assert all(item["status"] == "processing" for item in data["items"])
        # Should have some processing orders from test data


class TestFilteringByAmount:
    """Test amount range filtering"""
    
    def test_min_amount_filter(self):
        """Test minimum amount filter"""
        create_test_orders()
        response = client.get("/orders?min_amount=100")
        data = response.json()
        
        assert all(item["amount"] >= 100 for item in data["items"])
    
    def test_max_amount_filter(self):
        """Test maximum amount filter"""
        create_test_orders()
        response = client.get("/orders?max_amount=50")
        data = response.json()
        
        assert all(item["amount"] <= 50 for item in data["items"])
    
    def test_amount_range_filter(self):
        """Test both min and max amount filters"""
        create_test_orders()
        response = client.get("/orders?min_amount=25&max_amount=150")
        data = response.json()
        
        assert all(25 <= item["amount"] <= 150 for item in data["items"])
    
    def test_amount_filter_with_no_results(self):
        """Test amount filter with no matching results"""
        create_test_orders()
        response = client.get("/orders?min_amount=1000")
        data = response.json()
        
        assert len(data["items"]) == 0
        assert data["total"] == 0


class TestFilteringByDate:
    """Test date range filtering"""
    
    def test_start_date_filter(self):
        """Test start date filter"""
        create_test_orders()
        start_date = (datetime.now(timezone.utc) - timedelta(days=4)).date().isoformat()
        response = client.get(f"/orders?start_date={start_date}")
        data = response.json()
        
        for item in data["items"]:
            item_date = datetime.fromisoformat(item["created_at"])
            assert item_date.date() >= datetime.fromisoformat(start_date).date()
    
    def test_end_date_filter(self):
        """Test end date filter"""
        create_test_orders()
        end_date = (datetime.now(timezone.utc) - timedelta(days=6)).date().isoformat()
        response = client.get(f"/orders?end_date={end_date}")
        data = response.json()
        
        for item in data["items"]:
            item_date = datetime.fromisoformat(item["created_at"])
            assert item_date.date() <= datetime.fromisoformat(end_date).date()
    
    def test_date_range_filter(self):
        """Test both start and end date filters"""
        create_test_orders()
        start_date = (datetime.now(timezone.utc) - timedelta(days=7)).date().isoformat()
        end_date = (datetime.now(timezone.utc) - timedelta(days=2)).date().isoformat()
        response = client.get(f"/orders?start_date={start_date}&end_date={end_date}")
        data = response.json()
        
        for item in data["items"]:
            item_date = datetime.fromisoformat(item["created_at"]).date()
            assert datetime.fromisoformat(start_date).date() <= item_date <= datetime.fromisoformat(end_date).date()


class TestCombinedFilters:
    """Test multiple filters combined"""
    
    def test_status_and_amount_filters(self):
        """Test status and amount filters together"""
        create_test_orders()
        response = client.get("/orders?status=completed&min_amount=50")
        data = response.json()
        
        assert all(item["status"] == "completed" for item in data["items"])
        assert all(item["amount"] >= 50 for item in data["items"])
    
    def test_all_filters_combined(self):
        """Test all filters together"""
        create_test_orders()
        start_date = (datetime.now(timezone.utc) - timedelta(days=4)).date().isoformat()
        response = client.get(f"/orders?status=completed&min_amount=40&max_amount=200&start_date={start_date}")
        data = response.json()
        
        for item in data["items"]:
            assert item["status"] == "completed"
            assert 40 <= item["amount"] <= 200
            item_date = datetime.fromisoformat(item["created_at"]).date()
            assert item_date >= datetime.fromisoformat(start_date).date()


class TestInvalidInputs:
    """Test invalid input handling"""
    
    def test_invalid_page_number(self):
        """Test invalid page numbers"""
        response = client.get("/orders?page=0")
        assert response.status_code == 422
        
        response = client.get("/orders?page=-1")
        assert response.status_code == 422
    
    def test_invalid_limit_values(self):
        """Test invalid limit values"""
        response = client.get("/orders?limit=0")
        assert response.status_code == 422
        
        response = client.get("/orders?limit=-5")
        assert response.status_code == 422
        
        response = client.get("/orders?limit=101")
        assert response.status_code == 422
    
    def test_invalid_date_format(self):
        """Test invalid date formats"""
        invalid_dates = ["not-a-date", "2024-13-01", "2024-02-30", "01/01/2024"]
        
        for invalid_date in invalid_dates:
            response = client.get(f"/orders?start_date={invalid_date}")
            assert response.status_code == 400
            assert "Invalid date format" in response.json()["detail"]
    
    def test_invalid_date_range(self):
        """Test start_date after end_date"""
        create_test_orders()
        start_date = datetime.now(timezone.utc).date().isoformat()
        end_date = (datetime.now(timezone.utc) - timedelta(days=5)).date().isoformat()
        
        response = client.get(f"/orders?start_date={start_date}&end_date={end_date}")
        assert response.status_code == 400
        assert "start_date must be less than or equal to end_date" in response.json()["detail"]
    
    def test_empty_string_dates(self):
        """Test empty string date parameters"""
        response = client.get("/orders?start_date=&end_date=")
        # Should handle gracefully (treated as None)
        assert response.status_code == 200
    
    def test_negative_amount_values(self):
        """Test negative amount values"""
        response = client.get("/orders?min_amount=-10")
        assert response.status_code == 422
        
        response = client.get("/orders?max_amount=-5")
        assert response.status_code == 422


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_no_results_with_filters(self):
        """Test filters that return no results"""
        create_test_orders()
        response = client.get("/orders?status=cancelled&min_amount=1000")
        data = response.json()
        
        assert len(data["items"]) == 0
        assert data["total"] == 0
        assert data["pages"] == 0
        assert data["has_next"] == False
        assert data["has_prev"] == False
    
    def test_single_result_pagination(self):
        """Test pagination when only one result"""
        reset_db()
        # Create order with unique amount
        client.post("/orders", json={"status": "created", "amount": 999.99})
        response = client.get("/orders?min_amount=999&max_amount=1000")
        data = response.json()
        
        assert data["total"] == 1
        assert len(data["items"]) == 1
        assert data["pages"] == 1
        assert data["has_next"] == False
        assert data["has_prev"] == False
    
    def test_large_limit_within_bounds(self):
        """Test maximum allowed limit"""
        reset_db()
        create_test_orders()
        response = client.get("/orders?limit=100")
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 100


class TestResponseConsistency:
    """Test response consistency and accuracy"""
    
    def test_total_count_accuracy(self):
        """Test total count matches actual data"""
        create_test_orders()
        
        # API enforces limit<=100, so validate total via DB count instead of requesting limit=1000
        with Session(engine) as session:
            expected_total = session.exec(select(func.count(Order.id))).one()

        response_page = client.get("/orders")
        assert response_page.status_code == 200
        data_page = response_page.json()

        assert data_page["total"] == expected_total
    
    def test_pagination_consistency(self):
        """Test pagination returns consistent results"""
        create_test_orders()
        
        # Get page 1
        response1 = client.get("/orders?page=1&limit=3")
        data1 = response1.json()
        
        # Get page 2
        response2 = client.get("/orders?page=2&limit=3")
        data2 = response2.json()
        
        # Ensure no overlap
        ids1 = {item["id"] for item in data1["items"]}
        ids2 = {item["id"] for item in data2["items"]}
        
        assert len(ids1.intersection(ids2)) == 0
    
    def test_filter_count_consistency(self):
        """Test filtered count matches filtered results"""
        create_test_orders()
        
        response = client.get("/orders?status=completed")
        assert response.status_code == 200
        data = response.json()
        
        # Verify all items match the filter
        assert all(item["status"] == "completed" for item in data["items"])

        # Validate `total` against DB count (pagination means len(items) can be < total)
        with Session(engine) as session:
            expected_total = session.exec(
                select(func.count(Order.id)).where(Order.status == OrderStatus.completed)
            ).one()

        assert data["total"] == expected_total
        assert len(data["items"]) <= data["limit"]


class TestMissingEdgeCases:
    """Additional edge cases that weren't covered in the main test suite"""
    
    def test_empty_database_pagination(self):
        """Test pagination when database is completely empty"""
        # Clear database first
        reset_db()
        
        response = client.get("/orders")
        data = response.json()
        
        assert len(data["items"]) == 0
        assert data["total"] == 0
        assert data["pages"] == 0
        assert data["has_next"] == False
        assert data["has_prev"] == False
    
    def test_unicode_date_handling(self):
        """Test date handling with unicode characters and edge date formats"""
        create_test_orders()
        
        # Test with microseconds
        now = datetime.now(timezone.utc)
        microsecond_date = now.strftime("%Y-%m-%dT%H:%M:%S.%f")
        response = client.get(f"/orders?start_date={microsecond_date}")
        assert response.status_code == 200
        
        # Test with timezone (should be handled gracefully)
        timezone_date = "2024-01-01T00:00:00Z"
        response = client.get(f"/orders?start_date={timezone_date}")
        # Should either work or return validation error, not crash
        assert response.status_code in [200, 400]
    
    def test_filter_with_extreme_values(self):
        """Test filters with extreme boundary values"""
        create_test_orders()
        
        # Test with very large amount
        response = client.get("/orders?min_amount=999999")
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 0
        
        # Test with very small amount
        response = client.get("/orders?max_amount=0.01")
        assert response.status_code == 200
        
        # Test with dates far in past/future
        past_date = "1900-01-01"
        future_date = "2100-12-31"
        response = client.get(f"/orders?start_date={past_date}&end_date={future_date}")
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

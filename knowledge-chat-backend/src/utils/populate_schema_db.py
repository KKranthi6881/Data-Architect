import json
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.agents.data_architect.schema_search_agent import SchemaSearchAgent

# Sample schema data
SAMPLE_SCHEMAS = [
    {
        "schema_name": "sales",
        "table_name": "customers",
        "description": "Contains customer information including demographics and contact details",
        "columns": ["customer_id", "first_name", "last_name", "email", "phone", "address", "city", "state", "zip_code", "country", "registration_date", "last_login_date", "customer_segment"],
        "primary_key": ["customer_id"],
        "foreign_keys": [],
        "data_types": {
            "customer_id": "INTEGER",
            "first_name": "VARCHAR(50)",
            "last_name": "VARCHAR(50)",
            "email": "VARCHAR(100)",
            "phone": "VARCHAR(20)",
            "address": "VARCHAR(200)",
            "city": "VARCHAR(50)",
            "state": "VARCHAR(50)",
            "zip_code": "VARCHAR(20)",
            "country": "VARCHAR(50)",
            "registration_date": "TIMESTAMP",
            "last_login_date": "TIMESTAMP",
            "customer_segment": "VARCHAR(20)"
        }
    },
    {
        "schema_name": "sales",
        "table_name": "orders",
        "description": "Contains order information including order date, status, and total amount",
        "columns": ["order_id", "customer_id", "order_date", "order_status", "total_amount", "payment_method", "shipping_method", "discount_amount", "tax_amount", "shipping_amount"],
        "primary_key": ["order_id"],
        "foreign_keys": [
            {"column": "customer_id", "references": "customers.customer_id"}
        ],
        "data_types": {
            "order_id": "INTEGER",
            "customer_id": "INTEGER",
            "order_date": "TIMESTAMP",
            "order_status": "VARCHAR(20)",
            "total_amount": "DECIMAL(10,2)",
            "payment_method": "VARCHAR(50)",
            "shipping_method": "VARCHAR(50)",
            "discount_amount": "DECIMAL(10,2)",
            "tax_amount": "DECIMAL(10,2)",
            "shipping_amount": "DECIMAL(10,2)"
        }
    },
    {
        "schema_name": "sales",
        "table_name": "order_items",
        "description": "Contains details of items in each order including product, quantity, and price",
        "columns": ["order_item_id", "order_id", "product_id", "quantity", "unit_price", "discount_percent", "total_price"],
        "primary_key": ["order_item_id"],
        "foreign_keys": [
            {"column": "order_id", "references": "orders.order_id"},
            {"column": "product_id", "references": "products.product_id"}
        ],
        "data_types": {
            "order_item_id": "INTEGER",
            "order_id": "INTEGER",
            "product_id": "INTEGER",
            "quantity": "INTEGER",
            "unit_price": "DECIMAL(10,2)",
            "discount_percent": "DECIMAL(5,2)",
            "total_price": "DECIMAL(10,2)"
        }
    },
    {
        "schema_name": "inventory",
        "table_name": "products",
        "description": "Contains product information including name, description, price, and category",
        "columns": ["product_id", "product_name", "description", "category_id", "supplier_id", "unit_price", "stock_quantity", "reorder_level", "discontinued", "created_date", "modified_date"],
        "primary_key": ["product_id"],
        "foreign_keys": [
            {"column": "category_id", "references": "categories.category_id"},
            {"column": "supplier_id", "references": "suppliers.supplier_id"}
        ],
        "data_types": {
            "product_id": "INTEGER",
            "product_name": "VARCHAR(100)",
            "description": "TEXT",
            "category_id": "INTEGER",
            "supplier_id": "INTEGER",
            "unit_price": "DECIMAL(10,2)",
            "stock_quantity": "INTEGER",
            "reorder_level": "INTEGER",
            "discontinued": "BOOLEAN",
            "created_date": "TIMESTAMP",
            "modified_date": "TIMESTAMP"
        }
    },
    {
        "schema_name": "inventory",
        "table_name": "categories",
        "description": "Contains product categories and their descriptions",
        "columns": ["category_id", "category_name", "description", "parent_category_id"],
        "primary_key": ["category_id"],
        "foreign_keys": [
            {"column": "parent_category_id", "references": "categories.category_id"}
        ],
        "data_types": {
            "category_id": "INTEGER",
            "category_name": "VARCHAR(50)",
            "description": "TEXT",
            "parent_category_id": "INTEGER"
        }
    },
    {
        "schema_name": "analytics",
        "table_name": "customer_engagement",
        "description": "Contains metrics related to customer engagement with the platform",
        "columns": ["engagement_id", "customer_id", "session_count", "time_spent_minutes", "actions_completed", "feature_usage", "engagement_score", "engagement_date", "platform", "device_type"],
        "primary_key": ["engagement_id"],
        "foreign_keys": [
            {"column": "customer_id", "references": "customers.customer_id"}
        ],
        "data_types": {
            "engagement_id": "INTEGER",
            "customer_id": "INTEGER",
            "session_count": "INTEGER",
            "time_spent_minutes": "INTEGER",
            "actions_completed": "INTEGER",
            "feature_usage": "JSON",
            "engagement_score": "DECIMAL(5,2)",
            "engagement_date": "DATE",
            "platform": "VARCHAR(50)",
            "device_type": "VARCHAR(50)"
        }
    },
    {
        "schema_name": "analytics",
        "table_name": "feature_adoption",
        "description": "Tracks which features customers have adopted and when",
        "columns": ["adoption_id", "customer_id", "feature_id", "first_used_date", "last_used_date", "usage_count", "is_active"],
        "primary_key": ["adoption_id"],
        "foreign_keys": [
            {"column": "customer_id", "references": "customers.customer_id"},
            {"column": "feature_id", "references": "features.feature_id"}
        ],
        "data_types": {
            "adoption_id": "INTEGER",
            "customer_id": "INTEGER",
            "feature_id": "INTEGER",
            "first_used_date": "TIMESTAMP",
            "last_used_date": "TIMESTAMP",
            "usage_count": "INTEGER",
            "is_active": "BOOLEAN"
        }
    },
    {
        "schema_name": "analytics",
        "table_name": "features",
        "description": "Contains information about product features",
        "columns": ["feature_id", "feature_name", "description", "release_date", "category", "is_premium"],
        "primary_key": ["feature_id"],
        "foreign_keys": [],
        "data_types": {
            "feature_id": "INTEGER",
            "feature_name": "VARCHAR(100)",
            "description": "TEXT",
            "release_date": "DATE",
            "category": "VARCHAR(50)",
            "is_premium": "BOOLEAN"
        }
    }
]

def main():
    """Populate the schema vector database with sample data"""
    print("Initializing SchemaSearchAgent...")
    schema_agent = SchemaSearchAgent()
    
    print(f"Adding {len(SAMPLE_SCHEMAS)} schemas to vector database...")
    success = schema_agent.bulk_add_schemas(SAMPLE_SCHEMAS)
    
    if success:
        print("Successfully added schemas to vector database!")
    else:
        print("Failed to add schemas to vector database.")

if __name__ == "__main__":
    main() 
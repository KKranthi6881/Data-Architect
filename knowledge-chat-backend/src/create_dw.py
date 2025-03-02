import sqlite3
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataWarehouseCreator:
    def __init__(self, db_path: str = "saas_dw.db"):
        """Initialize the data warehouse creator."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def connect(self):
        """Create a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise

    def create_tables(self):
        """Create dimension and fact tables."""
        try:
            # Dimension Tables
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_users (
                user_id INTEGER PRIMARY KEY,
                email TEXT NOT NULL,
                company_id INTEGER,
                user_role TEXT,
                signup_date DATE,
                acquisition_source TEXT,
                user_segment TEXT,
                is_admin BOOLEAN,
                last_login_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_companies (
                company_id INTEGER PRIMARY KEY,
                company_name TEXT NOT NULL,
                industry TEXT,
                employee_range TEXT,
                plan_tier TEXT,
                contract_start_date DATE,
                contract_end_date DATE,
                billing_frequency TEXT,
                account_manager_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_features (
                feature_id INTEGER PRIMARY KEY,
                feature_name TEXT NOT NULL,
                feature_category TEXT,
                release_date DATE,
                is_premium BOOLEAN,
                is_beta BOOLEAN,
                feature_complexity TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dim_workflows (
                workflow_id INTEGER PRIMARY KEY,
                workflow_name TEXT NOT NULL,
                workflow_category TEXT,
                expected_completion_time INTEGER,
                target_persona TEXT,
                complexity_level TEXT,
                prerequisites TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            # Fact Tables
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_feature_usage (
                feature_usage_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                feature_id INTEGER,
                session_id TEXT,
                usage_timestamp TIMESTAMP,
                usage_duration INTEGER,
                interaction_count INTEGER,
                usage_success BOOLEAN,
                error_count INTEGER,
                device_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES dim_users(user_id),
                FOREIGN KEY (feature_id) REFERENCES dim_features(feature_id)
            )
            """)

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_workflow_completions (
                completion_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                workflow_id INTEGER,
                start_timestamp TIMESTAMP,
                completion_timestamp TIMESTAMP,
                completion_status TEXT,
                step_count INTEGER,
                retry_count INTEGER,
                satisfaction_score INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES dim_users(user_id),
                FOREIGN KEY (workflow_id) REFERENCES dim_workflows(workflow_id)
            )
            """)

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_user_engagement (
                engagement_id INTEGER PRIMARY KEY,
                user_id INTEGER,
                session_id TEXT,
                session_start TIMESTAMP,
                session_end TIMESTAMP,
                page_views INTEGER,
                feature_interactions INTEGER,
                time_in_app INTEGER,
                actions_completed INTEGER,
                collaboration_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES dim_users(user_id)
            )
            """)

            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_subscription_events (
                event_id INTEGER PRIMARY KEY,
                company_id INTEGER,
                event_type TEXT,
                previous_plan TEXT,
                new_plan TEXT,
                change_reason TEXT,
                mrr_change DECIMAL(10,2),
                seats_changed INTEGER,
                event_timestamp TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (company_id) REFERENCES dim_companies(company_id)
            )
            """)

            self.conn.commit()
            logger.info("Successfully created all tables")

        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.conn.rollback()
            raise

    def insert_sample_data(self):
        """Insert sample data into the tables."""
        try:
            # Sample Companies
            companies = [
                (1, 'TechCorp', 'Software', '100-500', 'enterprise', '2024-01-01', '2025-01-01', 'annual', 1),
                (2, 'StartupIO', 'Fintech', '10-50', 'growth', '2024-01-15', '2025-01-15', 'monthly', 2),
                (3, 'DataCo', 'Data Analytics', '50-100', 'business', '2024-02-01', '2025-02-01', 'annual', 1)
            ]
            self.cursor.executemany("""
            INSERT INTO dim_companies (company_id, company_name, industry, employee_range, plan_tier, 
                                     contract_start_date, contract_end_date, billing_frequency, account_manager_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, companies)

            # Sample Users
            users = [
                (1, 'sarah@techcorp.com', 1, 'admin', '2024-01-15', 'google_ads', 'enterprise', True, '2024-02-15'),
                (2, 'mike@startup.io', 2, 'user', '2024-01-16', 'referral', 'growth', False, '2024-02-14'),
                (3, 'alex@dataco.com', 3, 'admin', '2024-02-01', 'linkedin', 'business', True, '2024-02-15')
            ]
            self.cursor.executemany("""
            INSERT INTO dim_users (user_id, email, company_id, user_role, signup_date, 
                                 acquisition_source, user_segment, is_admin, last_login_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, users)

            # Sample Features
            features = [
                (1, 'AI Workflow Builder', 'Automation', '2024-01-01', True, False, 'Advanced'),
                (2, 'Team Collaboration', 'Collaboration', '2024-01-01', False, False, 'Basic'),
                (3, 'Analytics Dashboard', 'Analytics', '2024-02-01', True, True, 'Intermediate')
            ]
            self.cursor.executemany("""
            INSERT INTO dim_features (feature_id, feature_name, feature_category, release_date, 
                                    is_premium, is_beta, feature_complexity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, features)

            self.conn.commit()
            logger.info("Successfully inserted sample data")

        except Exception as e:
            logger.error(f"Error inserting sample data: {e}")
            self.conn.rollback()
            raise

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed database connection")

def main():
    """Main function to create and populate the data warehouse."""
    logging.basicConfig(level=logging.INFO)
    
    dw_creator = DataWarehouseCreator()
    try:
        dw_creator.connect()
        dw_creator.create_tables()
        dw_creator.insert_sample_data()
    finally:
        dw_creator.close()

if __name__ == "__main__":
    main() 
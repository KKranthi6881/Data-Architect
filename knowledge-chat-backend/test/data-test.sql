-- Basic Data Validation Queries

-- 1. Check table counts
SELECT 'Users' as table_name, COUNT(*) as record_count FROM dim_users
UNION ALL
SELECT 'Companies', COUNT(*) FROM dim_companies
UNION ALL
SELECT 'Features', COUNT(*) FROM dim_features
UNION ALL
SELECT 'Feature Usage', COUNT(*) FROM fact_feature_usage;

-- 2. Verify Company Data
SELECT 
    c.company_name,
    c.industry,
    c.plan_tier,
    COUNT(u.user_id) as user_count,
    MAX(u.last_login_date) as latest_login
FROM dim_companies c
LEFT JOIN dim_users u ON c.company_id = u.company_id
GROUP BY c.company_id, c.company_name, c.industry, c.plan_tier
ORDER BY user_count DESC;

-- 3. Feature Usage Analysis
SELECT 
    f.feature_name,
    f.feature_category,
    CASE WHEN f.is_premium THEN 'Premium' ELSE 'Free' END as tier,
    COUNT(fu.feature_usage_id) as usage_count,
    AVG(fu.usage_duration) as avg_duration,
    AVG(fu.error_count) as avg_errors
FROM dim_features f
LEFT JOIN fact_feature_usage fu ON f.feature_id = fu.feature_id
GROUP BY f.feature_id, f.feature_name, f.feature_category, f.is_premium
ORDER BY usage_count DESC;

-- 4. User Engagement Metrics
SELECT 
    u.user_segment,
    COUNT(DISTINCT u.user_id) as user_count,
    AVG(ue.time_in_app) as avg_time_in_app,
    AVG(ue.page_views) as avg_page_views,
    AVG(ue.feature_interactions) as avg_interactions
FROM dim_users u
LEFT JOIN fact_user_engagement ue ON u.user_id = ue.user_id
GROUP BY u.user_segment;

-- 5. Subscription Changes Impact
SELECT 
    c.company_name,
    se.event_type,
    se.previous_plan,
    se.new_plan,
    se.mrr_change,
    se.seats_changed,
    se.event_timestamp
FROM fact_subscription_events se
JOIN dim_companies c ON se.company_id = c.company_id
ORDER BY se.event_timestamp DESC;

-- 6. Feature Usage by Company Tier
SELECT 
    c.plan_tier,
    f.feature_name,
    COUNT(fu.feature_usage_id) as usage_count,
    COUNT(DISTINCT fu.user_id) as unique_users,
    AVG(fu.usage_duration) as avg_duration
FROM dim_companies c
JOIN dim_users u ON c.company_id = u.company_id
JOIN fact_feature_usage fu ON u.user_id = fu.user_id
JOIN dim_features f ON fu.feature_id = f.feature_id
GROUP BY c.plan_tier, f.feature_name
ORDER BY c.plan_tier, usage_count DESC;

-- 7. User Adoption Funnel
WITH UserAdoption AS (
    SELECT 
        u.user_id,
        u.user_segment,
        COUNT(DISTINCT fu.feature_id) as features_used,
        COUNT(DISTINCT ue.session_id) as sessions,
        SUM(ue.time_in_app) as total_time_in_app
    FROM dim_users u
    LEFT JOIN fact_feature_usage fu ON u.user_id = fu.user_id
    LEFT JOIN fact_user_engagement ue ON u.user_id = ue.user_id
    GROUP BY u.user_id, u.user_segment
)
SELECT 
    user_segment,
    COUNT(*) as total_users,
    AVG(features_used) as avg_features_used,
    AVG(sessions) as avg_sessions,
    AVG(total_time_in_app) as avg_total_time
FROM UserAdoption
GROUP BY user_segment;

-- 8. Feature Success Rates
SELECT 
    f.feature_name,
    COUNT(fu.feature_usage_id) as total_attempts,
    SUM(CASE WHEN fu.usage_success = 1 THEN 1 ELSE 0 END) as successful_attempts,
    ROUND(AVG(CASE WHEN fu.usage_success = 1 THEN 1 ELSE 0 END) * 100, 2) as success_rate,
    AVG(fu.error_count) as avg_errors
FROM dim_features f
JOIN fact_feature_usage fu ON f.feature_id = fu.feature_id
GROUP BY f.feature_name
ORDER BY success_rate DESC;

-- 9. User Growth Over Time
SELECT 
    strftime('%Y-%m', signup_date) as month,
    COUNT(*) as new_users,
    SUM(COUNT(*)) OVER (ORDER BY signup_date) as cumulative_users
FROM dim_users
GROUP BY month
ORDER BY month;

-- 10. Workflow Completion Analysis
SELECT 
    w.workflow_name,
    COUNT(wc.completion_id) as total_attempts,
    AVG(wc.step_count) as avg_steps,
    AVG(wc.retry_count) as avg_retries,
    AVG(wc.satisfaction_score) as avg_satisfaction,
    AVG(
        ROUND(
            CAST(
                (strftime('%s', wc.completion_timestamp) - strftime('%s', wc.start_timestamp))
                AS FLOAT
            ) / 60, 2
        )
    ) as avg_completion_time_minutes
FROM dim_workflows w
LEFT JOIN fact_workflow_completions wc ON w.workflow_id = wc.workflow_id
GROUP BY w.workflow_name
ORDER BY total_attempts DESC;

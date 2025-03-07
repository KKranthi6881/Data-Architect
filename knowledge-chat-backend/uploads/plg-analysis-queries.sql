-- 1. User Adoption Funnel Analysis
-- This query analyzes the user journey through key features
WITH user_adoption AS (
    SELECT 
        du.user_segment,
        df.feature_category,
        COUNT(DISTINCT du.user_id) as total_users,
        COUNT(DISTINCT CASE WHEN ffu.usage_success = true THEN du.user_id END) as successful_users,
        COUNT(DISTINCT CASE WHEN ffu.usage_duration > 300 THEN du.user_id END) as engaged_users
    FROM dim_users du
    LEFT JOIN fact_feature_usage ffu ON du.user_id = ffu.user_id
    LEFT JOIN dim_features df ON ffu.feature_id = df.feature_id
    WHERE ffu.usage_timestamp >= date('now', '-30 days')
    GROUP BY du.user_segment, df.feature_category
)
SELECT 
    user_segment,
    feature_category,
    total_users,
    successful_users,
    engaged_users,
    ROUND(successful_users * 100.0 / total_users, 2) as success_rate,
    ROUND(engaged_users * 100.0 / total_users, 2) as engagement_rate
FROM user_adoption
ORDER BY total_users DESC;

-- 2. Workflow Completion Analysis with Time Series
-- Analyzes workflow success rates over time with moving averages
WITH daily_completions AS (
    SELECT 
        date(fwc.start_timestamp) as completion_date,
        dw.workflow_category,
        COUNT(*) as total_attempts,
        SUM(CASE WHEN fwc.completion_status = 'completed' THEN 1 ELSE 0 END) as successful_completions,
        AVG(fwc.satisfaction_score) as avg_satisfaction
    FROM fact_workflow_completions fwc
    JOIN dim_workflows dw ON fwc.workflow_id = dw.workflow_id
    WHERE fwc.start_timestamp >= date('now', '-90 days')
    GROUP BY date(fwc.start_timestamp), dw.workflow_category
)
SELECT 
    completion_date,
    workflow_category,
    total_attempts,
    successful_completions,
    ROUND(successful_completions * 100.0 / total_attempts, 2) as daily_success_rate,
    avg_satisfaction,
    AVG(successful_completions * 100.0 / total_attempts) OVER (
        PARTITION BY workflow_category 
        ORDER BY completion_date 
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) as rolling_7day_success_rate
FROM daily_completions
ORDER BY completion_date DESC, workflow_category;

-- 3. Feature Adoption Impact on Revenue
-- Correlates feature usage with subscription changes
WITH feature_adoption AS (
    SELECT 
        dc.company_id,
        dc.plan_tier,
        COUNT(DISTINCT df.feature_id) as features_used,
        COUNT(DISTINCT CASE WHEN df.is_premium = true THEN df.feature_id END) as premium_features_used,
        AVG(ffu.usage_duration) as avg_usage_duration
    FROM dim_companies dc
    JOIN dim_users du ON dc.company_id = du.company_id
    JOIN fact_feature_usage ffu ON du.user_id = ffu.user_id
    JOIN dim_features df ON ffu.feature_id = df.feature_id
    WHERE ffu.usage_timestamp >= date('now', '-90 days')
    GROUP BY dc.company_id, dc.plan_tier
),
subscription_changes AS (
    SELECT 
        company_id,
        SUM(mrr_change) as total_mrr_change,
        COUNT(*) as number_of_changes
    FROM fact_subscription_events
    WHERE event_timestamp >= date('now', '-90 days')
    GROUP BY company_id
)
SELECT 
    fa.plan_tier,
    fa.features_used,
    fa.premium_features_used,
    ROUND(AVG(fa.avg_usage_duration), 2) as avg_feature_duration,
    ROUND(AVG(sc.total_mrr_change), 2) as avg_mrr_change,
    COUNT(*) as number_of_companies
FROM feature_adoption fa
LEFT JOIN subscription_changes sc ON fa.company_id = sc.company_id
GROUP BY fa.plan_tier, fa.features_used, fa.premium_features_used
HAVING number_of_companies >= 5
ORDER BY avg_mrr_change DESC;

-- 4. User Engagement Scoring
-- Creates a comprehensive engagement score based on multiple factors
WITH user_activity AS (
    SELECT 
        du.user_id,
        du.user_segment,
        COUNT(DISTINCT fue.session_id) as total_sessions,
        SUM(fue.time_in_app) as total_time_in_app,
        SUM(fue.actions_completed) as total_actions,
        SUM(fue.collaboration_count) as total_collaborations,
        COUNT(DISTINCT ffu.feature_id) as unique_features_used,
        COUNT(DISTINCT CASE WHEN fwc.completion_status = 'completed' THEN fwc.workflow_id END) as workflows_completed
    FROM dim_users du
    LEFT JOIN fact_user_engagement fue ON du.user_id = fue.user_id
    LEFT JOIN fact_feature_usage ffu ON du.user_id = ffu.user_id
    LEFT JOIN fact_workflow_completions fwc ON du.user_id = fwc.user_id
    WHERE fue.session_start >= date('now', '-30 days')
    GROUP BY du.user_id, du.user_segment
)
SELECT 
    user_id,
    user_segment,
    total_sessions,
    total_time_in_app,
    total_actions,
    total_collaborations,
    unique_features_used,
    workflows_completed,
    ROUND(
        (
            (total_sessions * 10) + 
            (total_time_in_app / 3600.0 * 20) + 
            (total_actions * 5) + 
            (total_collaborations * 15) + 
            (unique_features_used * 25) + 
            (workflows_completed * 30)
        ) / 100.0,
        2
    ) as engagement_score
FROM user_activity
ORDER BY engagement_score DESC;

-- 5. Feature Usage Pattern Analysis
-- Identifies common feature usage sequences and patterns
WITH feature_sequence AS (
    SELECT 
        ffu.user_id,
        df.feature_category,
        ffu.usage_timestamp,
        LAG(df.feature_category) OVER (
            PARTITION BY ffu.user_id 
            ORDER BY ffu.usage_timestamp
        ) as previous_feature,
        LEAD(df.feature_category) OVER (
            PARTITION BY ffu.user_id 
            ORDER BY ffu.usage_timestamp
        ) as next_feature
    FROM fact_feature_usage ffu
    JOIN dim_features df ON ffu.feature_id = df.feature_id
    WHERE ffu.usage_timestamp >= date('now', '-30 days')
)
SELECT 
    feature_category,
    previous_feature,
    next_feature,
    COUNT(*) as sequence_count,
    ROUND(
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (PARTITION BY feature_category),
        2
    ) as sequence_percentage
FROM feature_sequence
WHERE previous_feature IS NOT NULL 
    AND next_feature IS NOT NULL
GROUP BY feature_category, previous_feature, next_feature
HAVING sequence_count >= 10
ORDER BY sequence_count DESC;

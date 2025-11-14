-- =========================================================================
-- DROP ALL EXISTING OBJECTS
-- =========================================================================
DROP VIEW IF EXISTS review_quality_signals;
DROP FUNCTION IF EXISTS update_product_rating_metrics(VARCHAR);
DROP FUNCTION IF EXISTS calculate_review_sentiment_score(INT, INT);
DROP FUNCTION IF EXISTS get_product_sentiment_stats(VARCHAR);
DROP TABLE IF EXISTS product_reviews CASCADE;
DROP INDEX IF EXISTS idx_product_reviews;
DROP INDEX IF EXISTS idx_user_reviews;
DROP INDEX IF EXISTS idx_rating;
DROP INDEX IF EXISTS idx_created_desc;

-- =========================================================================
-- CREATE TABLE product_reviews
-- =========================================================================
CREATE TABLE product_reviews (
    review_id VARCHAR(36) PRIMARY KEY,
    product_id VARCHAR(36) NOT NULL,
    sku_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    rating SMALLINT NOT NULL CHECK (rating >= 1 AND rating <= 5),
    title TEXT,
    content TEXT,
    helpful_count INT DEFAULT 0,
    sentiment_score DECIMAL(3,2),
    sentiment_label VARCHAR(20),
    created_at TIMESTAMP WITH TIME ZONE NOT NULL
);

-- CREATE INDEXES
CREATE INDEX idx_product_reviews ON product_reviews(product_id);
CREATE INDEX idx_user_reviews ON product_reviews(user_id);
CREATE INDEX idx_rating ON product_reviews(rating);
CREATE INDEX idx_created_desc ON product_reviews(created_at DESC);

-- =========================================================================
-- ALTER TABLES
-- =========================================================================
ALTER TABLE product_features 
ADD COLUMN IF NOT EXISTS review_count INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS avg_rating_updated DECIMAL(3,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS rating_distribution JSONB;

ALTER TABLE user_profiles
ADD COLUMN IF NOT EXISTS review_count INT DEFAULT 0,
ADD COLUMN IF NOT EXISTS avg_rating_given DECIMAL(3,2) DEFAULT 0,
ADD COLUMN IF NOT EXISTS is_verified_reviewer BOOLEAN DEFAULT FALSE;

-- =========================================================================
-- CREATE FUNCTION get_product_sentiment_stats
-- =========================================================================
CREATE OR REPLACE FUNCTION get_product_sentiment_stats(input_product_id VARCHAR)
RETURNS TABLE (
    avg_sentiment DECIMAL,
    positive_ratio DECIMAL,
    total_reviews BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        AVG(sentiment_score) as avg_sentiment,
        CAST(COUNT(CASE WHEN sentiment_label = 'positive' THEN 1 END) AS DECIMAL) / 
            NULLIF(COUNT(*), 0) as positive_ratio,
        COUNT(*) as total_reviews
    FROM product_reviews
    WHERE product_id = input_product_id
    AND created_at >= NOW() - INTERVAL '180 days';
END;
$$ LANGUAGE plpgsql;

-- =========================================================================
-- COMMENTS
-- =========================================================================
COMMENT ON TABLE product_reviews IS 'Review data synced from MySQL for AI training';
COMMENT ON COLUMN product_reviews.sentiment_score IS 'Calculated using sentiment analysis model';
COMMENT ON COLUMN product_reviews.helpful_count IS 'Number of users who found this review helpful';

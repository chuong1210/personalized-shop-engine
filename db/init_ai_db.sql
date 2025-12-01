-- =========================================================================
-- SIMPLIFIED AI DATABASE SCHEMA (PostgreSQL)
-- =========================================================================
-- docker exec -i postgres psql -U postgres -d shop_service < init_ai_db.sql
-- Get-Content init_ai_db.sql | docker exec -i postgres psql -U postgres -d shop_service

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector"; -- Sửa "pgvector" thành "vector" cho chuẩn với các phiên bản mới

-- =========================================================================
-- 1. BẢNG TƯƠNG TÁC USER-PRODUCT
-- =========================================================================
CREATE TABLE user_interactions (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    product_id VARCHAR(36) NOT NULL,
    shop_id VARCHAR(36) NOT NULL,
    action_type VARCHAR(20) NOT NULL,
    score DECIMAL(5,2) DEFAULT 1.0,
    quantity INT DEFAULT 1,
    price DECIMAL(15,2),
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tạo Index riêng biệt sau khi tạo bảng
CREATE INDEX idx_user_time ON user_interactions (user_id, created_at DESC);
CREATE INDEX idx_product_interactions ON user_interactions (product_id);
CREATE INDEX idx_action_type ON user_interactions (action_type);
CREATE INDEX idx_recent_interactions 
ON user_interactions (created_at)
WHERE created_at::date > CURRENT_DATE - INTERVAL '90 days';


-- =========================================================================
-- 2. BẢNG PRODUCT FEATURES
-- =========================================================================
CREATE TABLE product_features (
    product_id VARCHAR(36) PRIMARY KEY,
    category_id VARCHAR(36),
    brand_id VARCHAR(36),
    shop_id VARCHAR(36),
    current_price DECIMAL(15,2),
    view_count_7d INT DEFAULT 0,
    view_count_30d INT DEFAULT 0,
    purchase_count_7d INT DEFAULT 0,
    purchase_count_30d INT DEFAULT 0,
    conversion_rate DECIMAL(5,4) DEFAULT 0,
    avg_rating DECIMAL(3,2) DEFAULT 0,
    trending_score DECIMAL(10,2) DEFAULT 0,
    text_embedding vector(768),
    similar_product_ids VARCHAR(36)[] DEFAULT '{}',
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tạo Index riêng biệt
CREATE INDEX idx_product_category ON product_features (category_id);
CREATE INDEX idx_trending_score ON product_features (trending_score DESC);
CREATE INDEX idx_conversion_rate ON product_features (conversion_rate DESC);


-- =========================================================================
-- 3. BẢNG USER PROFILE
-- =========================================================================
CREATE TABLE user_profiles (
    user_id VARCHAR(36) PRIMARY KEY,
    total_orders INT DEFAULT 0,
    total_spent DECIMAL(15,2) DEFAULT 0,
    avg_order_value DECIMAL(15,2) DEFAULT 0,
    favorite_categories VARCHAR(36)[] DEFAULT '{}',
    favorite_brands VARCHAR(36)[] DEFAULT '{}',
    price_range_min DECIMAL(15,2) DEFAULT 0,
    price_range_max DECIMAL(15,2) DEFAULT 0,
    discount_seeker_score DECIMAL(3,2) DEFAULT 0.5,
    last_purchase_at TIMESTAMP WITH TIME ZONE,
    last_active_at TIMESTAMP WITH TIME ZONE,
    profile_updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tạo Index riêng biệt
CREATE INDEX idx_last_purchase ON user_profiles (last_purchase_at);
-- Sử dụng GIN index cho mảng để tìm kiếm hiệu quả hơn
CREATE INDEX idx_favorite_categories ON user_profiles USING GIN (favorite_categories);


-- =========================================================================
-- 4. BẢNG RECOMMENDATION LOG
-- =========================================================================
CREATE TABLE recommendation_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    product_id VARCHAR(36) NOT NULL,
    rec_type VARCHAR(30) NOT NULL,
    rec_position INT,
    rec_score DECIMAL(10,4),
    shown_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    clicked_at TIMESTAMP WITH TIME ZONE,
    purchased_at TIMESTAMP WITH TIME ZONE,
    purchase_amount DECIMAL(15,2),
    page_context VARCHAR(50)
);

-- Tạo Index riêng biệt
CREATE INDEX idx_user_shown ON recommendation_logs (user_id, shown_at DESC);
CREATE INDEX idx_product_logs ON recommendation_logs (product_id);
CREATE INDEX idx_rec_type ON recommendation_logs (rec_type);
CREATE INDEX idx_tracking ON recommendation_logs (shown_at, clicked_at, purchased_at);


-- =========================================================================
-- MATERIALIZED VIEW: Daily Performance Metrics
-- =========================================================================
CREATE MATERIALIZED VIEW daily_recommendation_stats AS
SELECT
    DATE(shown_at) as date,
    rec_type,
    COUNT(*) as impressions,
    COUNT(clicked_at) as clicks,
    COUNT(purchased_at) as conversions,
    CAST(COUNT(clicked_at) AS DECIMAL) / NULLIF(COUNT(*), 0) as ctr,
    CAST(COUNT(purchased_at) AS DECIMAL) / NULLIF(COUNT(*), 0) as conversion_rate,
    SUM(purchase_amount) as revenue,
    AVG(purchase_amount) as avg_order_value
FROM recommendation_logs
WHERE shown_at >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY DATE(shown_at), rec_type;

CREATE INDEX ON daily_recommendation_stats (date DESC, rec_type);

-- Chạy trong PostgreSQL
ALTER TABLE product_features 
ADD COLUMN IF NOT EXISTS hybrid_embedding vector(64);

-- Tạo index để tìm kiếm nhanh cho vector mới này
CREATE INDEX ON product_features USING ivfflat (hybrid_embedding vector_cosine_ops);


-- Thêm cột vector cho ảnh
-- 1. Tạo bảng riêng để lưu vector hình ảnh (Quan hệ 1-N với Product)
CREATE TABLE product_image_embeddings (
    id BIGSERIAL PRIMARY KEY,
    product_id VARCHAR(36) NOT NULL,
    image_url TEXT, -- Lưu lại URL để biết vector này của ảnh nào
    embedding vector(512), -- Vector của CLIP
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. Tạo Index để tìm kiếm nhanh
CREATE INDEX idx_img_embed ON product_image_embeddings USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_img_pid ON product_image_embeddings (product_id);


UPDATE user_interactions
SET created_at = NOW() - (random() * INTERVAL '30 days');

-- "Bơm" dữ liệu Trending
UPDATE product_features 
SET 
    trending_score = floor(random() * 100) + 1,
    view_count_7d = floor(random() * 500),
    review_count = floor(random() * 50),
    avg_rating_updated = 4.5
WHERE trending_score IS NULL OR trending_score = 0;
-- SELECT * FROM product_features WHERE product_id=74500335;
SELECT user_id, action_type, created_at, NOW() - created_at as age
FROM user_interactions 
WHERE user_id = '11810269'
AND created_at >= NOW() - INTERVAL '365 days';
-- =========================================================================
-- HELPER FUNCTIONS
-- =========================================================================

CREATE OR REPLACE FUNCTION get_user_item_matrix(days_back INT DEFAULT 90)
RETURNS TABLE (
    user_id VARCHAR,
    product_id VARCHAR,
    score DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ui.user_id,
        ui.product_id,
        SUM(
            ui.score *
            EXP(-EXTRACT(EPOCH FROM (NOW() - ui.created_at)) / (30 * 86400))
        ) as final_score
    FROM user_interactions ui
    WHERE ui.created_at >= NOW() - INTERVAL '1 day' * days_back
    AND ui.action_type IN ('view', 'cart_add', 'purchase', 'wishlist')
    GROUP BY ui.user_id, ui.product_id
    HAVING SUM(ui.score) > 0;
END;
$$ LANGUAGE plpgsql;


CREATE OR REPLACE FUNCTION get_frequently_bought_together(
    input_product_id VARCHAR,
    limit_count INT DEFAULT 10
)
RETURNS TABLE (
    product_id VARCHAR,
    frequency BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ui2.product_id,
        COUNT(*) as freq
    FROM user_interactions ui1
    JOIN user_interactions ui2 ON
        ui1.user_id = ui2.user_id
        AND ui1.created_at::date = ui2.created_at::date
        AND ui1.product_id != ui2.product_id
    WHERE ui1.product_id = input_product_id
    AND ui1.action_type = 'purchase'
    AND ui2.action_type = 'purchase'
    AND ui1.created_at >= NOW() - INTERVAL '90 days'
    GROUP BY ui2.product_id
    ORDER BY freq DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;


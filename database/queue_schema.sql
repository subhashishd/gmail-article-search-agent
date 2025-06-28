-- Persistent Article Processing Queue Schema
-- This enables restart-safe article processing

-- Article processing queue table
CREATE TABLE IF NOT EXISTS article_processing_queue (
    id SERIAL PRIMARY KEY,
    article_data BYTEA NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    retries INTEGER DEFAULT 0,
    batch_number INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_queue_status ON article_processing_queue (status);
CREATE INDEX IF NOT EXISTS idx_queue_batch ON article_processing_queue (batch_number);
CREATE INDEX IF NOT EXISTS idx_queue_created ON article_processing_queue (created_at);

-- Processing progress tracking table
CREATE TABLE IF NOT EXISTS processing_progress (
    id INTEGER PRIMARY KEY DEFAULT 1,
    total_articles INTEGER DEFAULT 0,
    processed_articles INTEGER DEFAULT 0,
    successful_articles INTEGER DEFAULT 0,
    failed_articles INTEGER DEFAULT 0,
    current_batch INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'idle',
    current_article TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    error_message TEXT,
    updated_at TIMESTAMP DEFAULT NOW(),
    
    -- Ensure only one progress record
    CONSTRAINT single_progress_record CHECK (id = 1)
);

-- Insert initial progress record if not exists
INSERT INTO processing_progress (id) VALUES (1) ON CONFLICT (id) DO NOTHING;

-- Function to update progress timestamp
CREATE OR REPLACE FUNCTION update_progress_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update timestamp
DROP TRIGGER IF EXISTS progress_update_trigger ON processing_progress;
CREATE TRIGGER progress_update_trigger
    BEFORE UPDATE ON processing_progress
    FOR EACH ROW
    EXECUTE FUNCTION update_progress_timestamp();

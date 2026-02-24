import sys
import logging
from b3_pipeline import storage, adjustments, config

logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

conn = storage.get_connection()

logger.info("Fetching raw prices from DB...")
prices = storage.get_all_prices(conn)

logger.info("Fetching corporate actions from DB...")
corp_actions = storage.get_all_corporate_actions(conn)
stock_actions = storage.get_all_stock_actions(conn)

logger.info("Computing all adjustments...")
adj_prices, splits = adjustments.compute_all_adjustments(prices, corp_actions, stock_actions)

logger.info("Updating adjusted columns in DB...")
storage.update_adjusted_columns(conn, adj_prices)

if not splits.empty:
    logger.info("Saving detected splits...")
    storage.upsert_detected_splits(conn, splits)

logger.info("Done!")
conn.close()

import duckdb
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def run_duckdb_query(query: str) -> pd.DataFrame:
    """
    Connects to an in-memory DuckDB, installs necessary extensions (resiliently),
    and runs a given SQL query. Returns a pandas DataFrame.
    """
    con = None
    try:
        con = duckdb.connect(database=':memory:', read_only=False)
        # Install / load extensions if available; ignore errors if they are already installed
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception as e:
            logger.debug("httpfs install/load skipped or failed (may already be present): %s", e)
        try:
            con.execute("INSTALL parquet; LOAD parquet;")
        except Exception as e:
            logger.debug("parquet install/load skipped or failed (may already be present): %s", e)

        # Execute the query and fetch results as a pandas DataFrame
        result_df = con.execute(query).fetchdf()
        return result_df
    finally:
        if con is not None:
            con.close()

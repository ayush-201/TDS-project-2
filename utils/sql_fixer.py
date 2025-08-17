import sqlglot
from sqlglot import exp
import logging
import re

logger = logging.getLogger(__name__)

def fix_sql_query(sql: str) -> str:
    """
    Applies a two-stage process to fix common LLM-generated SQL errors.
    1. Attempts a robust AST transformation to correct the query structure.
    2. Falls back to regex-based replacements for known bad patterns.
    """
    logger.debug(f"Original SQL query: {sql}")

    # --- Stage 1: AST-Based Transformation ---
    try:
        ast = sqlglot.parse_one(sql, read="duckdb")

        def transformer(node):
            # Fix: JULIANDAY(a) - JULIANDAY(b) -> DATE_DIFF('day', b, a)
            if isinstance(node, exp.Sub) and \
               isinstance(node.left, exp.Func) and node.left.name.upper() == "JULIANDAY" and \
               isinstance(node.right, exp.Func) and node.right.name.upper() == "JULIANDAY":
                start_date = node.right.expressions[0]
                end_date = node.left.expressions[0]
                return sqlglot.func("DATE_DIFF", exp.Literal.string("day"), start_date, end_date)

            # Fix: STRPTIME(a, b, c) -> STRPTIME(a, b)
            if isinstance(node, exp.Func) and node.name.upper() == "STRPTIME" and len(node.expressions) > 2:
                logger.debug(f"Trimming extra arguments from STRPTIME in AST: {node}")
                return sqlglot.func("STRPTIME", node.expressions[0], node.expressions[1])

            return node

        transformed_ast = ast.transform(transformer)
        fixed_sql = transformed_ast.sql(dialect="duckdb")
        logger.debug("AST transformation successful.")

    except Exception as e:
        logger.warning(f"AST transformation failed: {e}. Falling back to string replacements.")
        fixed_sql = sql

    # --- Stage 2: Regex-Based Cleanup (Failsafe) ---
    # Fix: Replace JULIANDAY(a) - JULIANDAY(b) -> DATE_DIFF('day', b, a)
    final_sql = re.sub(
        r"JULIANDAY\s*\(\s*(.*?)\s*\)\s*-\s*JULIANDAY\s*\(\s*(.*?)\s*\)",
        r"DATE_DIFF('day', \2, \1)",
        fixed_sql,
        flags=re.IGNORECASE | re.DOTALL
    )

    if "JULIANDAY" in final_sql.upper():
        logger.warning("JULIANDAY still present after regex. This may indicate a complex, unhandled pattern.")

    # More robust STRPTIME fallback regex to fix 3-arg STRPTIME(a, b, c) to STRPTIME(a, b)
    # Handles nested expressions like CAST(...), function calls, etc.
    strptime_fixed_sql = re.sub(
        r"STRPTIME\s*\(\s*((?:[^(),]+|\([^()]*\))+)\s*,\s*((?:'[^']*'|\"[^\"]*\"|[^(),]+)+?)\s*,\s*((?:[^()]+|\([^()]*\))+)\s*\)",
        r"STRPTIME(\1, \2)",
        final_sql,
        flags=re.IGNORECASE
    )


    if strptime_fixed_sql != final_sql:
        logger.info("Applied regex fix for 3-arg STRPTIME() to 2-arg format.")

    final_sql = strptime_fixed_sql

    logger.debug(f"Final normalized query: {final_sql}")
    return final_sql

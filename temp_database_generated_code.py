import duckdb
import pandas as pd
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def solve():
    s3_path = 's3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1'

    con = duckdb.connect(database=':memory:', read_only=False)
    con.execute('INSTALL httpfs;')
    con.execute('LOAD httpfs;')
    con.execute('INSTALL parquet;')
    con.execute('LOAD parquet;')

    # Question 1: Which high court disposed the most cases from 2019 - 2022?
    court_disposal_query = f"""
    SELECT court, COUNT(*) AS disposal_count
    FROM read_parquet('{s3_path}')
    WHERE year BETWEEN 2019 AND 2022
    GROUP BY court
    ORDER BY disposal_count DESC
    LIMIT 1
    """
    court_disposal_result = con.execute(court_disposal_query).fetchdf()
    most_cases_court = court_disposal_result['court'][0] if not court_disposal_result.empty else None

    # Question 2: What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?
    regression_query = f"""
    SELECT
        year,
        AVG(DATE_DIFF('day', STRPTIME(date_of_registration, '%d-%m-%Y'), decision_date)) AS avg_delay
    FROM read_parquet('{s3_path}')
    WHERE court = '33_10'
    GROUP BY year
    """
    regression_data = con.execute(regression_query).fetchdf()

    if not regression_data.empty:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(regression_data['year'], regression_data['avg_delay'])
        regression_slope = slope
    else:
        regression_slope = None

    # Question 3: Plot the year and # of days of delay from the above question as a scatterplot with a regression line.
    img_buf = io.BytesIO()
    if not regression_data.empty:
        plt.figure(figsize=(10, 6))
        sns.regplot(x='year', y='avg_delay', data=regression_data)
        plt.xlabel('Year')
        plt.ylabel('Average Delay (Days)')
        plt.title('Average Delay vs. Year for court 33_10')
        plt.savefig(img_buf, format='webp')
        plt.close()
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        image_data_uri = f"data:image/webp;base64,{img_base64}"
    else:
        image_data_uri = None

    con.close()

    return {
        "Which high court disposed the most cases from 2019 - 2022?": most_cases_court,
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": regression_slope,
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": image_data_uri
    }

if __name__ == "__main__":
    print(json.dumps(solve()))
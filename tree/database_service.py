import psycopg2
import pandas as pd

class DatabaseService:
    def __init__(self, db_name="mspr_api", user="postgres", password="mspr_password", host="localhost", port=5432):
        self.conn = psycopg2.connect(
            dbname=db_name,
            user=user,
            password=password,
            host=host,
            port=port
        )

    def get_full_dataset(self):
        query = """
            SELECT 
                p.name AS pandemic_name,
                i.total_cases,
                i.total_deaths,
                r.date AS report_date,
                c.name AS country_name,
                c.population,
                r.new_cases,
                r.new_deaths
            FROM report r
            JOIN infection i ON r.infection_id = i.infection_id
            JOIN country c ON i.country_id = c.country_id
            JOIN pandemic p ON i.pandemic_id = p.pandemic_id;
        """
        df = pd.read_sql_query(query, self.conn)
        return df

    def close(self):
        self.conn.close()

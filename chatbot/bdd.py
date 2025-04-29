import psycopg2
from datetime import datetime

def get_connection():
    return psycopg2.connect(
        dbname="mspr_api", 
        user="postgres", 
        password="mspr_password",
        host="localhost", 
        port="5432" 
    )

def lire_pays():
    connection = get_connection()
    cursor = connection.cursor()
    try:
        query = "SELECT country_id, name, continent, iso3, population FROM Country;"
        cursor.execute(query)
        countries = cursor.fetchall() 
        return countries
    except Exception as e:
        print("Erreur lors de la lecture des pays:", e)
    finally:
        cursor.close()
        connection.close()

def lire_pandemies():
    connection = get_connection()
    cursor = connection.cursor()
    try:
        query = "SELECT pandemic_id, name, pathogen, description, start_date, end_date, notes FROM Pandemic;"
        cursor.execute(query)
        pandemics = cursor.fetchall() 
        return pandemics
    except Exception as e:
        print("Erreur lors de la lecture des pandémies:", e)
    finally:
        cursor.close()
        connection.close()

def lire_infections():
    connection = get_connection()
    cursor = connection.cursor()
    try:
        query = """
        SELECT i.infection_id, c.name AS country_name, p.name AS pandemic_name, i.total_cases, i.total_deaths
        FROM Infection i
        JOIN Country c ON i.country_id = c.country_id
        JOIN Pandemic p ON i.pandemic_id = p.pandemic_id;
        """
        cursor.execute(query)
        infections = cursor.fetchall() 
        return infections
    except Exception as e:
        print("Erreur lors de la lecture des infections:", e)
    finally:
        cursor.close()
        connection.close()

def lire_rapports():
    connection = get_connection()
    cursor = connection.cursor()
    try:
        query = """
        SELECT r.report_id, i.country_id, i.pandemic_id, r.date, r.new_cases, r.new_deaths
        FROM Report r
        JOIN Infection i ON r.infection_id = i.infection_id;
        """
        cursor.execute(query)
        reports = cursor.fetchall() 
        return reports
    except Exception as e:
        print("Erreur lors de la lecture des rapports:", e)
    finally:
        cursor.close()
        connection.close()

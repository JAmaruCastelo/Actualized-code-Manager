import sqlite3 as sq
import pandas as pd

# creamos el path de la base de datos que se va a utilizar
path="D:/00006-dataBase_CREES/LongTermMonitoringProject.db"

### realizamos la conexion de los datos
conexion=sq.connect(path)

#### funcion para obtener todos los nombres de la tabla de mi conexion
def see_tables(conexion):
    cursor=conexion.cursor()
    query="""SELECT name FROM sqlite_master WHERE type='table';"""
    cursor.execute(query)
    return [ e[0] for e in cursor.fetchall()]

def ver_tabla (name):
    """ funcion para poder llamar a una de las tablas en formato df. necesita el nombre de la tabla entre comillas"""
    sql_query=f"SELECT * FROM {name}"
    df1 = pd.read_sql_query(sql_query, conexion)
    return df1

tablas_names=see_tables (conexion)
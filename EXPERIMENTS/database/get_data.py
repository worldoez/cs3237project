import psycopg2

conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    host="localhost",
    password="11223344",
    port=5431,
)

cur = conn.cursor()
cur.execute("SELECT * FROM cam_data;")
rows = cur.fetchall()
conn.commit()
conn.close()
for row in rows:
    print(row)

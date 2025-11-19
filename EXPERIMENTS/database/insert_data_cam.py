import psycopg2

conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    host="localhost",
    password="11223344",
    port=5431,
)

cur = conn.cursor()

cur.execute(
    """ 
    INSERT INTO cam_data (distance, top_left_x, top_left_y, top_right_x, 
        top_right_y, bottom_right_x, bottom_right_y, 
        bottom_left_x, bottom_left_y) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """,
    (15.750, 100.5, 50.3, 200.7, 50.1, 200.9, 150.2, 100.2, 150.8),
)

conn.commit()

cur.close()
conn.close()

import psycopg2

conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    host="localhost",
    password="11223344",
    port=5431,
)

cur = conn.cursor()

cur.execute("""CREATE TABLE cam_data(
            id SERIAL PRIMARY KEY,
            distance DECIMAL(10, 3) NOT NULL,
            top_left_x DECIMAL(10, 3),
            top_left_y DECIMAL(10, 3),
            top_right_x DECIMAL(10, 3),
            top_right_y DECIMAL(10, 3),
            bottom_right_x DECIMAL(10, 3),
            bottom_right_y DECIMAL(10, 3),
            bottom_left_x DECIMAL(10, 3),
            bottom_left_y DECIMAL(10, 3)
            );
            """)

conn.commit()

cur.close()
conn.close()

import psycopg2

conn = psycopg2.connect(
    database="postgres",
    user="postgres",
    host="localhost",
    password="11223344",
    port=5431,
)

cur = conn.cursor()

cur.execute("""
            CREATE TABLE imu_cam_data(
            id SERIAL PRIMARY KEY,
            command varchar(255) NOT NULL,
            turn_angle_deg DECIMAL(10,3) NOT NULL,
            distance DECIMAL(10, 3) NOT NULL,
            is_april BOOLEAN DEFAULT FALSE,
            timestamp varchar(255) NOT NULL
            );
            """)

conn.commit()

cur.close()
conn.close()

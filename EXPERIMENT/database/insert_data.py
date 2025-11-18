import psycopg2


def insert_data():
    conn = psycopg2.connect(
        database="postgres",
        user="postgres",
        host="localhost",
        password="11223344",
        port=5431,
    )

    cur = conn.cursor()

    cur.execute("INSERT INTO movement_data(direction, speed) VALUES('Right', 10)")

    conn.commit()
    cur.close()
    conn.close()

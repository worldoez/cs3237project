import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from datetime import datetime
from typing import Dict


class RobotCommandAnalytics:
    def __init__(self, db_pool):
        self.db_pool = db_pool

    def get_connection(self):
        return self.db_pool.getconn()

    def return_connection(self, conn):
        self.db_pool.putconn(conn)

    def get_command_distribution(
        self, table_name: str = "imu_cam_data"
    ) -> Dict[str, int]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT command, COUNT(*) as count 
                    FROM {table_name} 
                    GROUP BY command 
                    ORDER BY count DESC
                """)
                results = cur.fetchall()
                return {row["command"]: row["count"] for row in results}
        finally:
            self.return_connection(conn)

    def get_turn_statistics(self, table_name: str = "imu_cam_data") -> Dict[str, float]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT 
                        AVG(turn_angle_deg) as avg_turn,
                        MIN(turn_angle_deg) as min_turn,
                        MAX(turn_angle_deg) as max_turn,
                        STDDEV(turn_angle_deg) as stddev_turn
                    FROM {table_name}
                    WHERE turn_angle_deg != 0
                """)
                result = cur.fetchone()
                return dict(result) if result else {}
        finally:
            self.return_connection(conn)

    def get_distance_statistics(
        self, table_name: str = "imu_cam_data"
    ) -> Dict[str, float]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT 
                        AVG(distance) as avg_distance,
                        MIN(distance) as min_distance,
                        MAX(distance) as max_distance,
                        STDDEV(distance) as stddev_distance,
                        SUM(distance) as total_distance
                    FROM {table_name}
                    WHERE distance > 0
                """)
                result = cur.fetchone()
                return dict(result) if result else {}
        finally:
            self.return_connection(conn)

    def get_april_tag_usage(self, table_name: str = "imu_cam_data") -> Dict[str, int]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT 
                        is_april,
                        COUNT(*) as count
                    FROM {table_name}
                    GROUP BY is_april
                """)
                results = cur.fetchall()
                return {
                    ("with_april" if row["is_april"] else "without_april"): row["count"]
                    for row in results
                }
        finally:
            self.return_connection(conn)

    def get_time_range(self, table_name: str = "imu_cam_data") -> Dict[str, str]:
        conn = self.get_connection()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(f"""
                    SELECT 
                        MIN(timestamp) as first_command,
                        MAX(timestamp) as last_command
                    FROM {table_name}
                """)
                result = cur.fetchone()

                if result and result["first_command"] and result["last_command"]:
                    first_dt = datetime.fromtimestamp(float(result["first_command"]))
                    last_dt = datetime.fromtimestamp(float(result["last_command"]))

                    return {
                        "first_command_raw": result["first_command"],
                        "first_command": first_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "last_command_raw": result["last_command"],
                        "last_command": last_dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                        "duration_seconds": float(result["last_command"])
                        - float(result["first_command"]),
                    }
                return {}
        finally:
            self.return_connection(conn)

    def generate_full_report(self, table_name: str = "imu_cam_data") -> str:
        report = []
        report.append("=" * 60)
        report.append("ROBOT COMMAND ANALYTICS REPORT")
        report.append("=" * 60)

        report.append("\n COMMAND DISTRIBUTION:")
        cmd_dist = self.get_command_distribution(table_name)
        for cmd, count in cmd_dist.items():
            report.append(f"  {cmd}: {count}")

        report.append("\n TURN ANGLE STATISTICS:")
        turn_stats = self.get_turn_statistics(table_name)
        for key, value in turn_stats.items():
            if value is not None:
                report.append(f"  {key}: {value:.2f}°")

        report.append("\n DISTANCE STATISTICS:")
        dist_stats = self.get_distance_statistics(table_name)
        for key, value in dist_stats.items():
            if value is not None:
                report.append(f"  {key}: {value:.3f} cm")

        report.append("\n APRIL TAG USAGE:")
        april_usage = self.get_april_tag_usage(table_name)
        for key, count in april_usage.items():
            report.append(f"  {key}: {count} imu_cam_data")

        report.append("\n TIME RANGE:")
        time_range = self.get_time_range(table_name)
        for key, value in time_range.items():
            report.append(f"  {key}: {value}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)


if __name__ == "__main__":
    db_pool = psycopg2.pool.SimpleConnectionPool(
        1,
        10,
        database="postgres",
        user="postgres",
        host="localhost",
        password="11223344",
        port=5431,
    )

    try:
        analytics = RobotCommandAnalytics(db_pool)

        print(analytics.generate_full_report())

    finally:
        db_pool.closeall()

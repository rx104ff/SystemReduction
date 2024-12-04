import sqlite3
from datetime import datetime


class NetworkDB:
    """A persistent class for managing metadata and associated file paths using SQLite."""

    def __init__(self, db_path: str = "network.db"):
        """
        Initialize the NetworkDB instance and connect to the database.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self._create_table()

    def _create_table(self):
        """Create the metadata table if it does not already exist."""
        with self.connection:
            self.connection.execute('''
                CREATE TABLE IF NOT EXISTS network (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    n INTEGER NOT NULL,
                    d INTEGER NOT NULL,
                    lp_first INTEGER NOT NULL,
                    lp_second INTEGER NOT NULL,
                    rp_first INTEGER NOT NULL,
                    rp_second INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    file_name TEXT NOT NULL
                )
            ''')

    def add_entry(self, n: int, d: int, lp: (int, int), rp: (int, int)) -> str:
        """
        Add a new entry to the metadata table and auto-generate a file name.

        Args:
            :param n:
            :param d:
            :param lp:
            :param rp:

        Returns:
            str: The generated file name.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Generate timestamp
        file_name = f"{n}_{d}_{lp}_{rp}_{timestamp}.csv"  # Generate file name

        with self.connection:
            self.connection.execute('''
                INSERT INTO network (n, d, lp_first, lp_second, rp_first, rp_second, timestamp, file_name)
                VALUES (?, ?, ?)
            ''', (n, d, lp[0], lp[1], rp[0], rp[1], datetime.now().isoformat(), file_name))

        return file_name

    def fetch_all(self):
        """
        Fetch all entries from the metadata table.

        Returns:
            list: A list of all rows in the metadata table.
        """
        with self.connection:
            cursor = self.connection.execute('SELECT * FROM network')
            return cursor.fetchall()

    def fetch_by_parameter(self, parameter: str):
        """
        Fetch entries filtered by parameter.

        Args:
            parameter (str): The parameter to filter by.

        Returns:
            list: A list of rows matching the parameter.
        """
        with self.connection:
            cursor = self.connection.execute('''
                SELECT * FROM metadata WHERE parameter = ?
            ''', (parameter,))
            return cursor.fetchall()

    def fetch_by_file_name(self, file_name: str):
        """
        Fetch entries filtered by file name.

        Args:
            file_name (str): The file name to filter by.

        Returns:
            list: A list of rows matching the file name.
        """
        with self.connection:
            cursor = self.connection.execute('''
                SELECT * FROM metadata WHERE file_name = ?
            ''', (file_name,))
            return cursor.fetchall()

    def delete_all(self):
        """Delete all entries from the metadata table."""
        with self.connection:
            self.connection.execute('DELETE FROM metadata')

    def close(self):
        """Close the database connection."""
        self.connection.close()

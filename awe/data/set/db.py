import os
import sqlite3
from typing import Optional


class Database:
    """
    SQLite database wrapper for holding a dataset of HTML pages and their
    visuals.
    """

    def __init__(self, db_file_path: str):
        self.db_file_path = db_file_path
        self.fresh = not os.path.exists(self.db_file_path)
        self.db = sqlite3.connect(self.db_file_path)

        # Initialize database.
        if self.fresh:
            with self.db:
                self.db.executescript('''
                    create table migrations(id integer primary key);
                    insert into migrations(id) values(1);
                    create table pages(
                        id integer primary key,
                        url text not null,
                        html text not null,
                        visuals text
                    );
                ''')

    def add(self,
        idx: int,
        url: str,
        html_text: str,
        visuals: Optional[str] = None,
    ):
        """Adds new page."""

        self.db.execute('''
            insert into pages(id, url, html, visuals)
                values(:id, :url, :html, :visuals)
        ''', {
            'id': idx,
            'url': url,
            'html': html_text,
            'visuals': visuals,
        })

    def save(self):
        """Saves the database to disk."""

        self.db.commit()

    def replace(self,
        idx: int,
        url: str,
        html_text: str,
        visuals: Optional[str] = None,
    ):
        """Replaces page at `idx`."""

        with self.db:
            self.db.execute('''
                update pages
                    set
                        url = :url,
                        html = :html,
                        visuals = :visuals
                    where id = :id
            ''', {
                'id': idx,
                'url': url,
                'html': html_text,
                'visuals': visuals,
            })

    def __len__(self) -> int:
        q = self.db.execute('select count(id) from pages')
        for (c,) in q:
            return c
        return 0

    def _get(self, idx: int, col: str):
        """Gets `col` of page at `idx`."""

        q = self.db.execute(f'select {col} from pages where id = :id', {
            'id': idx
        })
        for (value,) in q:
            return value
        raise RuntimeError(f'Page {idx} not found in {self.db_file_path!r}.')

    def get_url(self, idx: int) -> str:
        """Gets URL of page at `idx`."""

        return self._get(idx, 'url')

    def get_html_text(self, idx: int) -> str:
        """Gets HTML string of page at `idx`."""

        return self._get(idx, 'html')

    def get_visuals(self, idx: int) -> Optional[str]:
        """Gets visuals JSON string of page at `idx`."""

        return self._get(idx, 'visuals')

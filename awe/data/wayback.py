from dataclasses import dataclass

import requests


@dataclass
class WaybackPage:
    """Endpoint for https://archive.org/help/wayback_api.php."""
    original_url: str
    archive_url: str
    timestamp: str
    status: int

    @staticmethod
    def get(url: str, timestamp: str):
        response = requests.get('https://archive.org/wayback/available/',
            params={
                'url': url,
                'timestamp': timestamp
            })
        data = response.json()

        snapshots = data['archived_snapshots']
        if len(snapshots) == 0:
            return None

        snapshot = snapshots['closest']
        return WaybackPage(
            url,
            snapshot['url'],
            snapshot['timestamp'],
            int(snapshot['status'])
        )

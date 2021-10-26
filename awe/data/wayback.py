from dataclasses import dataclass
from json.decoder import JSONDecodeError

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

        try:
            data = response.json()
        except JSONDecodeError as error:
            data = {
                'url': url,
                'status': response.status_code,
                'error': error,
                'response': response.text
            }
            print(f'JSON error: {data}')
            if 'Too Many Requests' in response.text:
                raise RuntimeError('Too many requests.') from error
            return False # will try again next time

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

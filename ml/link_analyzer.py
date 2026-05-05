# ml/link_analyzer.py
import re
import requests
from urllib.parse import urlparse

# The URL pattern used in both functions — defined once so both stay consistent
URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def extract_urls(text):
    """
    Simply extracts and returns a list of raw URLs found in the text.
    Used by app.py to display actual links to the user on result.html.
    """
    return re.findall(URL_PATTERN, text)


def extract_and_analyze_links(text):
    """
    Extracts URLs from text and analyzes them for suspicious features.
    Returns a string of feature tokens without making any network requests.
    """
    urls = re.findall(URL_PATTERN, text)

    if not urls:
        return ''

    features = []

    for url in urls:
        try:
            parsed = urlparse(url)
            host = parsed.hostname or ''

            # Feature 1: Shortened URLs
            if any(s in host for s in ['bit.ly', 'tinyurl', 't.co', 'short', 'goo.gl']):
                features.append('shortened_url')

            # Feature 2: IP-based URL
            if re.match(r'\d+\.\d+\.\d+\.\d+', host):
                features.append('ip_based_url')

            # Feature 3: Suspicious TLDs
            if any(tld in host for tld in ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top']):
                features.append('suspicious_tld')

            # Feature 4: Too many subdomains
            if host.count('.') > 3:
                features.append('many_subdomains')

            # Feature 5: Long URL
            if len(url) > 75:
                features.append('long_url')

            # Feature 6: Not HTTPS
            if parsed.scheme == 'http':
                features.append('not_https')

            # Feature 7: Suspicious keywords in path
            path = (parsed.path or '').lower()
            if any(word in path for word in ['login', 'verify', 'account', 'secure', 'update']):
                features.append('suspicious_path')

        except:
            features.append('malformed_url')

    return ' '.join(set(features))
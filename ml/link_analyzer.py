import re
import requests
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

URL_PATTERN         = r'(?:https?://|www\.)[^\s<>"\'()]+'
BARE_DOMAIN_PATTERN = r'\b(?:[a-zA-Z0-9-]+\.)+(?:com|org|net|edu|gov|io|co|ng|uk|de|fr|au)\b'

SHORTENED_HOSTS = {
    'bit.ly', 'tinyurl.com', 't.co', 'goo.gl', 'short.ly',
    'ow.ly', 'buff.ly', 'rb.gy', 'cutt.ly', 'is.gd'
}

SUSPICIOUS_TLDS = {'.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.club'}

SUSPICIOUS_PATHS = {'verify', 'confirm', 'update', 'suspend', 'recover'}

SAFE_STATUS_CODES = {200, 301, 302, 303, 307, 308}

TRUSTED_DOMAINS = {
    'google.com', 'gmail.com', 'youtube.com', 'github.com', 'stackoverflow.com',
    'wikipedia.org', 'amazon.com', 'paypal.com', 'microsoft.com', 'apple.com',
    'zoom.us', 'slack.com', 'drive.google.com', 'docs.google.com',
    'linkedin.com', 'twitter.com', 'facebook.com', 'instagram.com',
    'jumia.com.ng', 'substack.com', 'techcrunch.com', 'reddit.com',
    'github.io', 'vercel.app', 'netlify.app', 'whatsapp.com', 'telegram.org',
    'bing.com', 'yahoo.com', 'duckduckgo.com', 'spotify.com', 'netflix.com'
}

def _normalise_url(raw):
    raw = raw.strip().rstrip('.,;)')
    if not raw.startswith(('http://', 'https://')):
        return 'https://' + raw
    return raw

def extract_urls(text):
    found = re.findall(URL_PATTERN, text, re.IGNORECASE)
    bare  = re.findall(BARE_DOMAIN_PATTERN, text, re.IGNORECASE)
    all_urls = list(found)
    for b in bare:
        if not any(b in u for u in found):
            all_urls.append(b)
    return [u.rstrip('.,;)') for u in all_urls]


# ── TRAINING: fast structural analysis, no network calls ─────────────────────
def extract_and_analyze_links(text):
    raw_urls = extract_urls(text)
    if not raw_urls:
        return ''

    features = set()
    for raw in raw_urls:
        url = _normalise_url(raw)
        try:
            parsed    = urlparse(url)
            full_host = (parsed.hostname or '').lower()
            path      = (parsed.path or '').lower()

            if any(s == full_host for s in SHORTENED_HOSTS):
                features.add('shortened_url')
            if re.match(r'^\d+\.\d+\.\d+\.\d+$', full_host):
                features.add('ip_based_url')
            if any(full_host.endswith(tld) for tld in SUSPICIOUS_TLDS):
                features.add('suspicious_tld')
            if full_host.count('.') > 3:
                features.add('many_subdomains')
            if len(url) > 100:
                features.add('long_url')
            if any(full_host == dom or full_host.endswith('.' + dom) for dom in TRUSTED_DOMAINS):
                features.add('trusted_domain')
            if parsed.scheme == 'http' and len(features) > 0:
                features.add('not_https')
            if any(word in path for word in SUSPICIOUS_PATHS) and len(features) > 0:
                features.add('suspicious_path')
        except Exception:
            features.add('malformed_url')

    return ' '.join(features)


# ── PREDICTION: live HTTP checks + detailed analysis ───────────────────────
def check_links_for_decision(text):
    """
    Returns:
        is_any_suspicious (bool),
        suspicious_urls (list),
        safe_urls (list),
        analysis_messages (list of human-readable strings)
    """
    raw_urls = extract_urls(text)
    if not raw_urls:
        return False, [], [], []

    suspicious_urls = []
    safe_urls = []
    analysis_messages = []

    def _analyze_one(raw):
        url = _normalise_url(raw)
        try:
            parsed    = urlparse(url)
            full_host = (parsed.hostname or '').lower()
            path      = (parsed.path or '').lower()
        except Exception:
            return raw, True, 'malformed', '❌ Malformed URL'

        # Hard structural checks
        if re.match(r'^\d+\.\d+\.\d+\.\d+$', full_host):
            return raw, True, 'ip_address', '⚠️ IP address URL (often malicious)'
        if any(full_host.endswith(tld) for tld in SUSPICIOUS_TLDS):
            return raw, True, 'suspicious_tld', f'⚠️ Suspicious TLD: {full_host}'
        if any(s == full_host for s in SHORTENED_HOSTS):
            return raw, True, 'shortened', '⚠️ Shortened URL (may hide destination)'

        is_trusted = any(full_host == dom or full_host.endswith('.' + dom) for dom in TRUSTED_DOMAINS)

        # Live HEAD request
        try:
            resp = requests.head(
                url,
                timeout=2,
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 SpamGuard-LinkChecker/1.0'}
            )
            if resp.status_code in SAFE_STATUS_CODES:
                if is_trusted:
                    return raw, False, 'trusted_domain', f'✅ Trusted domain: {full_host}'
                else:
                    return raw, False, 'reachable', f'✅ Reachable (HTTP {resp.status_code})'
            else:
                return raw, True, 'bad_status', f'⚠️ Bad HTTP status {resp.status_code}'
        except requests.exceptions.SSLError:
            return raw, True, 'ssl_error', '⚠️ SSL certificate error'
        except requests.exceptions.ConnectionError:
            return raw, True, 'unreachable', '⚠️ Unreachable or domain does not exist'
        except requests.exceptions.Timeout:
            return raw, True, 'timeout', '⚠️ Connection timeout'
        except Exception:
            return raw, True, 'error', '⚠️ Unknown error checking link'

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_analyze_one, url): url for url in raw_urls}
        for future in as_completed(futures):
            raw, is_suspicious, reason, msg = future.result()
            if is_suspicious:
                suspicious_urls.append(raw)
                analysis_messages.append(msg)
            else:
                safe_urls.append(raw)
                analysis_messages.append(msg)

    return len(suspicious_urls) > 0, suspicious_urls, safe_urls, analysis_messages


# Keep original function for backward compatibility with feature extraction
def analyze_links_for_prediction(text):
    """Returns (feature_string, suspicious_urls) for model input."""
    raw_urls = extract_urls(text)
    if not raw_urls:
        return '', []

    features = set()
    suspicious_urls = []

    def _check_url(raw):
        url = _normalise_url(raw)
        try:
            parsed    = urlparse(url)
            full_host = (parsed.hostname or '').lower()
            path      = (parsed.path or '').lower()
        except Exception:
            return raw, True, 'malformed'

        if re.match(r'^\d+\.\d+\.\d+\.\d+$', full_host):
            return raw, True, 'ip_address'
        if any(full_host.endswith(tld) for tld in SUSPICIOUS_TLDS):
            return raw, True, 'suspicious_tld'
        if any(s == full_host for s in SHORTENED_HOSTS):
            return raw, True, 'shortened'

        is_trusted = any(full_host == dom or full_host.endswith('.' + dom) for dom in TRUSTED_DOMAINS)

        try:
            resp = requests.head(
                url,
                timeout=2,
                allow_redirects=True,
                headers={'User-Agent': 'Mozilla/5.0 SpamGuard-LinkChecker/1.0'}
            )
            if resp.status_code in SAFE_STATUS_CODES:
                if is_trusted:
                    return raw, False, 'trusted_domain'
                else:
                    return raw, False, f'ok_{resp.status_code}'
            else:
                return raw, True, f'status_{resp.status_code}'
        except requests.exceptions.SSLError:
            return raw, True, 'ssl_error'
        except requests.exceptions.ConnectionError:
            return raw, True, 'unreachable'
        except requests.exceptions.Timeout:
            return raw, True, 'timeout'
        except Exception:
            return raw, True, 'error'

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_check_url, url): url for url in raw_urls}
        for future in as_completed(futures):
            raw, is_suspicious, reason = future.result()
            if is_suspicious:
                suspicious_urls.append(raw)
                if reason == 'ip_address':       features.add('ip_based_url')
                elif reason == 'suspicious_tld': features.add('suspicious_tld')
                elif reason == 'shortened':      features.add('shortened_url')
                elif reason == 'ssl_error':      features.add('ssl_error')
                elif reason in ('unreachable', 'timeout', 'error'):
                    features.add('unreachable_url')
                elif reason == 'malformed':      features.add('malformed_url')
                elif reason.startswith('status_'):
                    features.add('bad_status_url')
            else:
                if reason == 'trusted_domain':
                    features.add('trusted_domain')
                features.add('reachable_url')

    return ' '.join(features), suspicious_urls
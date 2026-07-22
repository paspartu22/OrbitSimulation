import os
import ssl
import urllib.request

import certifi
import truststore


def configure_ssl_certificates() -> None:
    """Configure HTTPS trust for urllib/wget used by pyatmos downloads."""
    try:
        truststore.inject_into_ssl()
        expected_context = ssl.create_default_context()
    except Exception:
        cafile = certifi.where()
        os.environ["SSL_CERT_FILE"] = cafile
        os.environ["REQUESTS_CA_BUNDLE"] = cafile
        expected_context = ssl.create_default_context(cafile=cafile)

    ssl._create_default_https_context = lambda: expected_context
    https_handler = urllib.request.HTTPSHandler(context=expected_context)
    opener = urllib.request.build_opener(https_handler)
    urllib.request.install_opener(opener)

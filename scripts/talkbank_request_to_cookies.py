#!/usr/bin/env python3
"""Convert a user-exported TalkBank cURL request into a private cookie jar.

The request is parsed as text. It is NEVER executed and no credential is logged.
"""
import argparse
from http.cookiejar import Cookie, MozillaCookieJar
from http.cookies import SimpleCookie
import os
from pathlib import Path
import shlex
from urllib.parse import urlsplit


def request_cookies(text):
    args = shlex.split(text)
    if not args or args[0] != 'curl':
        raise ValueError('Expected Chrome Copy as cURL (bash) output')
    urls = [value for value in args[1:] if value.startswith(('http://', 'https://'))]
    if len(urls) != 1:
        raise ValueError('Expected exactly one TalkBank request URL')
    url = urlsplit(urls[0])
    if url.scheme != 'https' or url.hostname != 'media.talkbank.org' or url.username or url.password:
        raise ValueError('Only an official HTTPS TalkBank media request is accepted')
    values = []
    for index, token in enumerate(args[:-1]):
        if token in ('-b', '--cookie'):
            values.append(args[index+1])
        elif token in ('-H', '--header') and args[index+1].lower().startswith('cookie:'):
            values.append(args[index+1].split(':', 1)[1].strip())
    if len(values) != 1 or '=' not in values[0]:
        raise ValueError('No single signed-in Cookie header; export the signed-in media page request again')
    cookies = SimpleCookie()
    cookies.load(values[0])
    if not cookies:
        raise ValueError('No cookies parsed from the signed-in request')
    return cookies


def convert(source, destination):
    cookies = request_cookies(Path(source).read_text())
    destination = Path(destination).expanduser()
    # Private and no-clobber from first creation, not a later chmod.
    descriptor = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    jar = MozillaCookieJar(str(destination))
    for item in cookies.values():
        jar.set_cookie(Cookie(0, item.key, item.value, None, False,
            'media.talkbank.org', False, False, '/', True, True,
            None, True, None, None, {}, False))
    jar.save(ignore_discard=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request_file', type=Path)
    parser.add_argument('cookie_file', type=Path)
    args = parser.parse_args()
    convert(args.request_file, args.cookie_file)
    print('Saved a private TalkBank media cookie file. No request was executed.')


if __name__ == '__main__':
    main()

// SSRF guard for the sidecar. Mirrors extraction_service/ssrf.py: http(s) only,
// no credentials, port allow-list, hostname deny-list, and every resolved
// address must be globally routable.
import dns from 'node:dns/promises';
import ipaddr from 'ipaddr.js';

export const ALLOWED_SCHEMES = new Set(['http:', 'https:']);
export const BLOCKED_HOSTS = new Set(['localhost', 'metadata.google.internal', 'instance-data']);
export const BLOCKED_HOST_SUFFIXES = ['.localhost', '.local', '.internal', '.intranet', '.home', '.lan', '.corp'];
const PUBLIC_RANGES = new Set(['unicast']);
const EXTRA_BLOCKED_V4 = [
  ['100.64.0.0', 10], // carrier-grade NAT
  ['192.0.0.0', 24],
  ['198.18.0.0', 15],
  ['240.0.0.0', 4],
  ['0.0.0.0', 8],
];
const EXTRA_BLOCKED_V6 = [
  ['64:ff9b::', 96], // NAT64
  ['2001:db8::', 32],
];

export class SsrfError extends Error {
  constructor(message, details = {}) {
    super(message);
    this.name = 'SsrfError';
    this.status = 403;
    this.details = details;
  }
}

export function isBlockedIp(address) {
  let ip;
  try {
    ip = ipaddr.parse(address);
  } catch {
    return true; // unparsable -> treat as hostile
  }
  if (ip.kind() === 'ipv6' && ip.isIPv4MappedAddress()) ip = ip.toIPv4Address();
  const range = ip.range();
  if (!PUBLIC_RANGES.has(range)) return true;
  const extras = ip.kind() === 'ipv4' ? EXTRA_BLOCKED_V4 : EXTRA_BLOCKED_V6;
  return extras.some(([net, bits]) => ip.match(ipaddr.parse(net), bits));
}

export async function systemResolver(hostname) {
  const records = await dns.lookup(hostname, { all: true, verbatim: true });
  return records.map((r) => r.address);
}

/**
 * Validate a URL before Chromium (or JSDOM) touches it.
 * @returns {Promise<{url: URL, addresses: string[]}>}
 */
export async function validateTargetUrl(rawUrl, { allowedPorts = [80, 443], resolver = systemResolver } = {}) {
  let url;
  try {
    url = new URL(String(rawUrl).trim());
  } catch {
    throw new SsrfError('invalid URL');
  }
  if (!ALLOWED_SCHEMES.has(url.protocol)) throw new SsrfError(`scheme not allowed: ${url.protocol}`);
  if (url.username || url.password) throw new SsrfError('credentials in URL are not allowed');
  const host = url.hostname.replace(/\.$/, '').toLowerCase();
  if (!host) throw new SsrfError('missing host');
  const port = Number(url.port || (url.protocol === 'https:' ? 443 : 80));
  if (!allowedPorts.includes(port)) throw new SsrfError(`port not allowed: ${port}`);
  if (BLOCKED_HOSTS.has(host) || BLOCKED_HOST_SUFFIXES.some((s) => host.endsWith(s))) {
    throw new SsrfError(`host not allowed: ${host}`);
  }
  const literal = host.startsWith('[') ? host.slice(1, -1) : host;
  let addresses;
  if (ipaddr.isValid(literal)) {
    addresses = [literal];
  } else {
    try {
      addresses = await resolver(host);
    } catch {
      throw new SsrfError(`dns resolution failed for ${host}`);
    }
    if (!addresses.length) throw new SsrfError(`dns resolution returned no addresses for ${host}`);
  }
  for (const address of addresses) {
    if (isBlockedIp(address)) {
      throw new SsrfError('target resolves to a non-public address', { host, ip: address });
    }
  }
  return { url, addresses };
}

/** Per-render cache so every sub-resource of a page is checked once per host. */
export function createHostGuard(options = {}) {
  const cache = new Map();
  return async (rawUrl) => {
    const key = (() => {
      try {
        const u = new URL(rawUrl);
        return `${u.protocol}//${u.host}`;
      } catch {
        return rawUrl;
      }
    })();
    if (!cache.has(key)) {
      cache.set(key, validateTargetUrl(rawUrl, options).then(() => true, (err) => err));
    }
    const result = await cache.get(key);
    if (result !== true) throw result;
  };
}

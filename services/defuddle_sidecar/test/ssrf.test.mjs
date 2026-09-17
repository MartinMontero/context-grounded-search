import assert from 'node:assert/strict';
import { test } from 'node:test';
import { JSDOM } from 'jsdom';
import { flattenShadowDom, parseHtml } from '../src/extract.mjs';
import { SsrfError, createHostGuard, isBlockedIp, validateTargetUrl } from '../src/ssrf.mjs';

const resolver = async (host) => {
  const table = {
    'public.example': ['93.184.216.34'],
    'mixed.example': ['93.184.216.34', '10.0.0.5'],
    'mapped.example': ['::ffff:192.168.1.10'],
  };
  if (!table[host]) throw new Error('nxdomain');
  return table[host];
};

test('private, loopback, link-local, CGNAT, mapped and reserved ranges are blocked', () => {
  for (const ip of ['10.1.2.3', '172.16.0.1', '192.168.0.1', '127.0.0.1', '169.254.169.254', '0.0.0.0',
    '100.64.0.1', '224.0.0.1', '240.0.0.1', '::1', '::', 'fe80::1', 'fc00::1', '::ffff:10.0.0.1', '64:ff9b::a00:1', '2001:db8::1']) {
    assert.equal(isBlockedIp(ip), true, ip);
  }
  for (const ip of ['93.184.216.34', '8.8.8.8', '2606:4700::1111']) assert.equal(isBlockedIp(ip), false, ip);
});

test('validateTargetUrl accepts public hosts and rejects everything else', async () => {
  const ok = await validateTargetUrl('https://public.example/x', { resolver });
  assert.equal(ok.url.hostname, 'public.example');
  for (const bad of ['ftp://public.example/', 'http://u:p@public.example/', 'http://public.example:22/',
    'http://localhost/', 'http://svc.cluster.local/', 'http://127.0.0.1/', 'http://[::1]/',
    'http://mixed.example/', 'http://mapped.example/', 'http://nope.example/', 'not a url']) {
    await assert.rejects(validateTargetUrl(bad, { resolver }), SsrfError, bad);
  }
});

test('host guard caches per origin and keeps rejecting', async () => {
  let calls = 0;
  const guard = createHostGuard({ resolver: async (h) => { calls += 1; return resolver(h); } });
  await guard('https://public.example/a.js');
  await guard('https://public.example/b.css');
  assert.equal(calls, 1);
  await assert.rejects(guard('http://mixed.example/'), SsrfError);
  await assert.rejects(guard('http://mixed.example/again'), SsrfError);
});

test('flattenShadowDom inlines nested open shadow roots so Defuddle can see them', async () => {
  const dom = new JSDOM('<html><body><article><h1>Widget page</h1><x-outer></x-outer></article></body></html>');
  const { document } = dom.window;
  const outer = document.querySelector('x-outer');
  const outerRoot = outer.attachShadow({ mode: 'open' });
  outerRoot.innerHTML = '<style>p{}</style><p>Outer shadow paragraph with enough words to be kept by the content heuristics of the parser.</p><x-inner></x-inner>';
  const inner = outerRoot.querySelector('x-inner');
  inner.attachShadow({ mode: 'open' }).innerHTML = '<p>Inner shadow paragraph, also long enough to be considered real article content by the scoring.</p>';
  const hosts = flattenShadowDom(document);
  assert.equal(hosts, 2);
  const html = dom.serialize();
  assert.match(html, /Outer shadow paragraph/);
  assert.match(html, /Inner shadow paragraph/);
  assert.doesNotMatch(html, /<style>/);
  const result = await parseHtml(html, 'https://example.com/widgets');
  assert.match(result.content, /Outer shadow paragraph/);
  assert.match(result.content, /Inner shadow paragraph/);
  assert.equal(result.title, 'Widget page');
});

// Defuddle parsing plus the Shadow-DOM flattening that runs inside Chromium.
import { Defuddle } from 'defuddle/node';

/**
 * Runs in the page (Playwright `page.evaluate`) and also in JSDOM for tests.
 * Recursively inlines every *open* shadow root into its host so that a plain
 * `document.documentElement.outerHTML` serialisation contains the shadow
 * content. Closed roots are made open beforehand by `FORCE_OPEN_SHADOW_ROOTS`.
 */
export function flattenShadowDom(root) {
  const doc = root.ownerDocument || root;
  const hosts = [];
  const walk = (node) => {
    for (const el of node.querySelectorAll('*')) {
      if (el.shadowRoot) hosts.push(el);
    }
  };
  walk(root);
  for (const host of hosts) {
    walk(host.shadowRoot);
  }
  // Deepest hosts first so nested shadow content is already inlined when the parent is copied.
  for (const host of hosts.reverse()) {
    const shadow = host.shadowRoot;
    if (!shadow) continue;
    const wrapper = doc.createElement('div');
    wrapper.setAttribute('data-shadow-host', host.tagName.toLowerCase());
    for (const child of Array.from(shadow.childNodes)) {
      if (child.nodeName === 'STYLE' || child.nodeName === 'SCRIPT') continue;
      wrapper.appendChild(child.cloneNode(true));
    }
    host.appendChild(wrapper);
  }
  return hosts.length;
}

// Injected before any page script runs: forces every shadow root to be open.
export const FORCE_OPEN_SHADOW_ROOTS = `(() => {
  const original = Element.prototype.attachShadow;
  Element.prototype.attachShadow = function (init) {
    return original.call(this, { ...init, mode: 'open' });
  };
})();`;

export function normaliseResult(result, finalUrl) {
  return {
    content: (result.content || '').trim(),
    wordCount: Number(result.wordCount || 0),
    title: result.title || null,
    author: result.author || null,
    description: result.description || null,
    published: result.published || null,
    site: result.site || result.domain || null,
    language: result.language || null,
    finalUrl: finalUrl || null,
  };
}

export async function parseHtml(html, url) {
  const result = await Defuddle(html, url || undefined, { markdown: true });
  return normaliseResult(result, url);
}

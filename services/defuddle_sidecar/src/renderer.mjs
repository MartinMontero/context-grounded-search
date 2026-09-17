// Headless Chromium rendering with SSRF enforcement on every network request.
import { chromium } from 'playwright';
import { FORCE_OPEN_SHADOW_ROOTS, flattenShadowDom, parseHtml } from './extract.mjs';
import { SsrfError, createHostGuard, validateTargetUrl } from './ssrf.mjs';

export class RenderTimeoutError extends Error {
  constructor(message) {
    super(message);
    this.name = 'RenderTimeoutError';
    this.status = 504;
  }
}

export class PayloadTooLargeError extends Error {
  constructor(message) {
    super(message);
    this.name = 'PayloadTooLargeError';
    this.status = 413;
  }
}

export class Renderer {
  constructor({ timeoutMs = 20000, maxBytes = 10 * 1024 * 1024, logger, resolver } = {}) {
    this.timeoutMs = timeoutMs;
    this.maxBytes = maxBytes;
    this.logger = logger;
    this.resolver = resolver;
    this.browser = null;
  }

  async browserInstance() {
    if (this.browser?.isConnected()) return this.browser;
    this.browser = await chromium.launch({
      headless: true,
      args: ['--disable-dev-shm-usage', '--disable-gpu', '--no-first-run', '--no-default-browser-check'],
    });
    this.browser.on('disconnected', () => {
      this.browser = null;
    });
    return this.browser;
  }

  async close() {
    if (this.browser) await this.browser.close().catch(() => {});
    this.browser = null;
  }

  async render(rawUrl) {
    const guardOptions = this.resolver ? { resolver: this.resolver } : {};
    const { url } = await validateTargetUrl(rawUrl, guardOptions);
    const guard = createHostGuard(guardOptions);
    const browser = await this.browserInstance();
    const context = await browser.newContext({
      javaScriptEnabled: true,
      acceptDownloads: false,
      ignoreHTTPSErrors: false,
      serviceWorkers: 'block',
      viewport: { width: 1280, height: 1600 },
      userAgent: 'contextual-rag-renderer/0.1 (+headless chromium)',
    });
    context.setDefaultTimeout(this.timeoutMs);
    context.setDefaultNavigationTimeout(this.timeoutMs);
    let ssrfViolation = null;
    await context.route('**/*', async (route) => {
      const request = route.request();
      const target = request.url();
      const type = request.resourceType();
      if (['media', 'font', 'websocket', 'manifest', 'eventsource'].includes(type)) {
        return route.abort('blockedbyclient');
      }
      try {
        await guard(target); // scheme, port, host deny-list, DNS -> public IP only
      } catch (err) {
        if (request.isNavigationRequest() && request.frame() === request.frame().page().mainFrame()) {
          ssrfViolation = err; // redirect of the top document into a private network
        }
        return route.abort('blockedbyclient');
      }
      return route.continue();
    });
    const page = await context.newPage();
    await page.addInitScript(FORCE_OPEN_SHADOW_ROOTS);
    page.on('dialog', (dialog) => dialog.dismiss().catch(() => {}));
    try {
      const response = await page.goto(url.href, { waitUntil: 'domcontentloaded' }).catch((err) => {
        if (ssrfViolation) throw ssrfViolation;
        if (/timeout/i.test(String(err))) throw new RenderTimeoutError(`navigation timed out after ${this.timeoutMs}ms`);
        throw err;
      });
      if (ssrfViolation) throw ssrfViolation;
      if (!response) throw new SsrfError('navigation produced no response');
      if (response.status() >= 400) {
        const err = new Error(`upstream returned HTTP ${response.status()}`);
        err.status = 502;
        throw err;
      }
      await page.waitForLoadState('networkidle', { timeout: Math.min(this.timeoutMs, 8000) }).catch(() => {});
      const hosts = await page.evaluate(`(${flattenShadowDom.toString()})(document)`);
      const html = await page.content();
      if (Buffer.byteLength(html, 'utf8') > this.maxBytes) {
        throw new PayloadTooLargeError(`rendered document exceeds ${this.maxBytes} bytes`);
      }
      const finalUrl = page.url();
      const result = await parseHtml(html, finalUrl);
      this.logger?.info({ url: url.href, finalUrl, shadowHosts: hosts, wordCount: result.wordCount }, 'rendered');
      return { ...result, shadowHosts: hosts };
    } finally {
      await context.close().catch(() => {});
    }
  }
}

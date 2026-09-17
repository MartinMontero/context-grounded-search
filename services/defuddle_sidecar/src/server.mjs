// Defuddle sidecar: JWT-protected HTTP API in front of Defuddle (JSDOM) and
// Playwright/Chromium rendering with Shadow-DOM flattening.
import express from 'express';
import jwt from 'jsonwebtoken';
import pino from 'pino';
import { parseHtml } from './extract.mjs';
import { Renderer } from './renderer.mjs';
import { SsrfError } from './ssrf.mjs';

const PORT = Number(process.env.PORT || 3000);
const MAX_PAYLOAD_BYTES = Number(process.env.MAX_PAYLOAD_BYTES || 10 * 1024 * 1024);
const RENDER_TIMEOUT_MS = Number(process.env.RENDER_TIMEOUT_MS || 20000);
const JWT_SECRET = process.env.SERVICE_JWT_SECRET;
const JWT_ISSUER = process.env.SERVICE_JWT_ISSUER || 'contextual-rag';
const JWT_AUDIENCE = process.env.SERVICE_JWT_AUDIENCE || 'rag-services';

const logger = pino({ level: process.env.LOG_LEVEL || 'info', base: { service: 'defuddle-sidecar' } });

if (!JWT_SECRET || JWT_SECRET.length < 32) {
  logger.fatal('SERVICE_JWT_SECRET must be set (>= 32 chars)');
  process.exit(1);
}

export function createApp({ renderer = new Renderer({ timeoutMs: RENDER_TIMEOUT_MS, maxBytes: MAX_PAYLOAD_BYTES, logger }) } = {}) {
  const app = express();
  app.disable('x-powered-by');
  app.use(express.json({ limit: MAX_PAYLOAD_BYTES, type: ['application/json'] }));

  app.get('/health', (_req, res) => res.json({ status: 'ok', service: 'defuddle-sidecar' }));

  // Service-to-service JWT (HS256 only; alg confusion impossible with an explicit list).
  app.use((req, res, next) => {
    const header = req.get('authorization') || '';
    const [scheme, token] = header.split(' ');
    if (scheme?.toLowerCase() !== 'bearer' || !token) {
      res.set('WWW-Authenticate', 'Bearer');
      return res.status(401).json({ error: 'missing bearer token' });
    }
    try {
      req.principal = jwt.verify(token, JWT_SECRET, {
        algorithms: ['HS256'],
        issuer: JWT_ISSUER,
        audience: JWT_AUDIENCE,
        clockTolerance: 30,
      });
      return next();
    } catch (err) {
      return res.status(401).json({ error: `invalid token: ${err.name}` });
    }
  });

  app.post('/parse', async (req, res, next) => {
    try {
      const { html, url } = req.body || {};
      if (typeof html !== 'string' || !html.trim()) return res.status(422).json({ error: 'html is required' });
      if (Buffer.byteLength(html, 'utf8') > MAX_PAYLOAD_BYTES) {
        return res.status(413).json({ error: `html exceeds ${MAX_PAYLOAD_BYTES} bytes` });
      }
      const result = await parseHtml(html, typeof url === 'string' ? url : undefined);
      return res.json(result);
    } catch (err) {
      return next(err);
    }
  });

  app.post('/render', async (req, res, next) => {
    try {
      const { url } = req.body || {};
      if (typeof url !== 'string' || !url) return res.status(422).json({ error: 'url is required' });
      const result = await renderer.render(url);
      return res.json(result);
    } catch (err) {
      return next(err);
    }
  });

  // eslint-disable-next-line no-unused-vars
  app.use((err, req, res, _next) => {
    const status = err instanceof SsrfError ? 403 : err.status || (err.type === 'entity.too.large' ? 413 : 500);
    const message = status === 500 ? 'internal error' : err.message;
    logger[status >= 500 ? 'error' : 'warn']({ err: err.message, status, path: req.path, caller: req.principal?.sub }, 'request failed');
    res.status(status).json({ error: message, details: err.details || undefined });
  });

  return app;
}

if (process.argv[1] && process.argv[1].endsWith('server.mjs')) {
  const renderer = new Renderer({ timeoutMs: RENDER_TIMEOUT_MS, maxBytes: MAX_PAYLOAD_BYTES, logger });
  const server = createApp({ renderer }).listen(PORT, '0.0.0.0', () => logger.info({ port: PORT }, 'sidecar listening'));
  server.requestTimeout = RENDER_TIMEOUT_MS + 10000;
  const shutdown = async (signal) => {
    logger.info({ signal }, 'shutting down');
    server.close();
    await renderer.close();
    process.exit(0);
  };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

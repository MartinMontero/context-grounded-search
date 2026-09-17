# Playwright's image ships Chromium + all system libraries; the tag must match
# the pinned `playwright` npm version.
FROM mcr.microsoft.com/playwright:v1.63.0-noble

ENV NODE_ENV=production \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PORT=3000

WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --omit=dev --ignore-scripts && npm cache clean --force
COPY src ./src

# pwuser exists in the base image; never run Chromium as root.
USER pwuser
EXPOSE 3000
CMD ["node", "src/server.mjs"]

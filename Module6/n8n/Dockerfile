FROM n8nio/n8n

# Установка Playwright
USER root
RUN npm install playwright
RUN npx playwright install chromium

# Возврат к n8n пользователю
USER node

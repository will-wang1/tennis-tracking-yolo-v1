# Uses the pre-built frontend/dist/ committed to the repo instead of running
# npm install/build here - this image needs no Node.js, no npm, and no
# network access to the npm registry to build. That matters when npm is
# blocked (a locked-down network, a corporate firewall) on the machine
# running `docker compose build`.
#
# Rebuild dist/ after any frontend source change and commit it:
#   cd frontend && npm run build
FROM nginx:1.27-alpine
COPY dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80

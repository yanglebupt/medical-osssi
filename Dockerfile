# docker build -t medical-osssi:0.0.0 .
FROM nginx
COPY nginx.conf /etc/nginx/nginx.conf
RUN mkdir -p /usr/share/nginx/cert/ssl
COPY ./ssl-cert /usr/share/nginx/cert/ssl
COPY ./dist /usr/share/nginx/html
EXPOSE 443

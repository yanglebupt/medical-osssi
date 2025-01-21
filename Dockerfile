# docker build -t medical-osssi:0.0.0 .
FROM nginx
RUN mkdir -p /ssl
COPY nginx.conf /etc/nginx/nginx.conf
COPY ./dist /usr/share/nginx/html
EXPOSE 443

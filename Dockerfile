# docker build -t medical-osssi:0.0.0 .
FROM nginx
COPY ./dist /usr/share/nginx/html
EXPOSE 80

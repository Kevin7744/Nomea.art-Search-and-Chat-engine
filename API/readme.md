Next start the Gunicorn socket with

- sudo systemctl start gunicorn.socket

Then enable it by running

- sudo systemctl enable gunicorn.socket

Configure nginx
Open nginx’s default configuration file with

- sudo nano /etc/nginx/sites-available/default

Restart default daemons, gunicorn, and nginx
- sudo systemctl daemon-reload
- sudo systemctl restart gunicorn
- sudo systemctl restart nginx

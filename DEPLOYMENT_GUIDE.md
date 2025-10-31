# Guide de Déploiement - Moteur d'Analyse de Sentiment

## 🚀 Déploiement en Production

### Prérequis Système

#### Configuration Minimale
- **OS** : Linux, macOS, ou Windows
- **Python** : 3.8 ou supérieur
- **RAM** : 512 MB minimum (2 GB recommandé)
- **Stockage** : 200 MB espace libre
- **Réseau** : Accès internet pour l'installation des dépendances

#### Configuration Recommandée
- **OS** : Linux Ubuntu 20.04+ ou CentOS 8+
- **Python** : 3.9 ou 3.10
- **RAM** : 4 GB ou plus
- **CPU** : 2 cœurs ou plus
- **Stockage** : 1 GB espace libre (pour logs et données temporaires)

### Installation en Production

#### 1. Préparation de l'Environnement

```bash
# Mise à jour du système (Ubuntu/Debian)
sudo apt update && sudo apt upgrade -y

# Installation de Python et pip
sudo apt install python3 python3-pip python3-venv -y

# Création d'un utilisateur dédié (recommandé)
sudo useradd -m -s /bin/bash sentiment-analyzer
sudo su - sentiment-analyzer
```

#### 2. Déploiement de l'Application

```bash
# Clonage du projet
git clone <repository-url> sentiment-analysis-engine
cd sentiment-analysis-engine

# Création de l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installation des dépendances
pip install --upgrade pip
pip install -r requirements.txt

# Validation de l'installation
python validate_installation.py
```

#### 3. Configuration de Production

```bash
# Copie de la configuration par défaut
cp config.json config.prod.json

# Édition de la configuration de production
nano config.prod.json
```

**Configuration de production recommandée** :

```json
{
  "sentiment_thresholds": {
    "positive": 0.05,
    "negative": -0.05
  },
  "output": {
    "summary_format": "json",
    "results_format": "csv",
    "summary_filename": "summary",
    "results_filename": "results"
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/sentiment-analysis/app.log",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  },
  "processing": {
    "batch_size": 200,
    "encoding_fallbacks": ["utf-8", "latin-1", "cp1252"]
  }
}
```

#### 4. Configuration des Logs

```bash
# Création du répertoire de logs
sudo mkdir -p /var/log/sentiment-analysis
sudo chown sentiment-analyzer:sentiment-analyzer /var/log/sentiment-analysis

# Configuration de la rotation des logs
sudo tee /etc/logrotate.d/sentiment-analysis << EOF
/var/log/sentiment-analysis/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 sentiment-analyzer sentiment-analyzer
}
EOF
```

### Déploiement avec Docker

#### Dockerfile

```dockerfile
FROM python:3.10-slim

# Variables d'environnement
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Création de l'utilisateur non-root
RUN useradd --create-home --shell /bin/bash app

# Répertoire de travail
WORKDIR /app

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY . .

# Changement de propriétaire
RUN chown -R app:app /app

# Changement d'utilisateur
USER app

# Port d'exposition (si nécessaire pour une API future)
EXPOSE 8000

# Commande par défaut
CMD ["python", "main.py", "--help"]
```

#### docker-compose.yml

```yaml
version: '3.8'

services:
  sentiment-analyzer:
    build: .
    container_name: sentiment-analyzer
    volumes:
      - ./data:/app/data:ro
      - ./output:/app/output
      - ./logs:/app/logs
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped
    command: python main.py /app/data/reviews.json --output-dir /app/output --config /app/config.prod.json
```

#### Commandes Docker

```bash
# Construction de l'image
docker build -t sentiment-analyzer:latest .

# Exécution avec Docker Compose
docker-compose up -d

# Vérification des logs
docker-compose logs -f sentiment-analyzer
```

### Automatisation avec Systemd

#### Fichier de Service

```bash
sudo tee /etc/systemd/system/sentiment-analyzer.service << EOF
[Unit]
Description=Sentiment Analysis Engine
After=network.target

[Service]
Type=oneshot
User=sentiment-analyzer
Group=sentiment-analyzer
WorkingDirectory=/home/sentiment-analyzer/sentiment-analysis-engine
Environment=PATH=/home/sentiment-analyzer/sentiment-analysis-engine/venv/bin
ExecStart=/home/sentiment-analyzer/sentiment-analysis-engine/venv/bin/python main.py /data/reviews.json --output-dir /data/output --config config.prod.json
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sentiment-analyzer

[Install]
WantedBy=multi-user.target
EOF
```

#### Gestion du Service

```bash
# Rechargement de systemd
sudo systemctl daemon-reload

# Activation du service
sudo systemctl enable sentiment-analyzer.service

# Démarrage du service
sudo systemctl start sentiment-analyzer.service

# Vérification du statut
sudo systemctl status sentiment-analyzer.service

# Consultation des logs
sudo journalctl -u sentiment-analyzer.service -f
```

### Monitoring et Surveillance

#### 1. Monitoring des Performances

**Script de monitoring** (`monitor.sh`) :

```bash
#!/bin/bash

LOG_FILE="/var/log/sentiment-analysis/monitor.log"
THRESHOLD_CPU=80
THRESHOLD_MEMORY=80

# Fonction de logging
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" >> $LOG_FILE
}

# Vérification de l'utilisation CPU
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | awk -F'%' '{print $1}')
if (( $(echo "$CPU_USAGE > $THRESHOLD_CPU" | bc -l) )); then
    log_message "WARNING: High CPU usage: ${CPU_USAGE}%"
fi

# Vérification de l'utilisation mémoire
MEMORY_USAGE=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')
if (( $(echo "$MEMORY_USAGE > $THRESHOLD_MEMORY" | bc -l) )); then
    log_message "WARNING: High memory usage: ${MEMORY_USAGE}%"
fi

# Vérification de l'espace disque
DISK_USAGE=$(df -h /var/log | awk 'NR==2 {print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 90 ]; then
    log_message "WARNING: High disk usage: ${DISK_USAGE}%"
fi

log_message "System check completed - CPU: ${CPU_USAGE}%, Memory: ${MEMORY_USAGE}%, Disk: ${DISK_USAGE}%"
```

#### 2. Cron Job pour Monitoring

```bash
# Ajout au crontab
crontab -e

# Monitoring toutes les 5 minutes
*/5 * * * * /home/sentiment-analyzer/monitor.sh

# Nettoyage des logs anciens (hebdomadaire)
0 2 * * 0 find /var/log/sentiment-analysis -name "*.log" -mtime +30 -delete
```

#### 3. Alertes par Email

**Configuration d'alertes** (`alert.py`) :

```python
#!/usr/bin/env python3
import smtplib
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_alert(subject, message):
    # Configuration SMTP
    smtp_server = "smtp.example.com"
    smtp_port = 587
    sender_email = "alerts@example.com"
    sender_password = "password"
    recipient_email = "admin@example.com"
    
    # Création du message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"[SENTIMENT-ANALYZER] {subject}"
    
    msg.attach(MIMEText(message, 'plain'))
    
    # Envoi
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        print("Alert sent successfully")
    except Exception as e:
        print(f"Failed to send alert: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python alert.py <subject> <message>")
        sys.exit(1)
    
    send_alert(sys.argv[1], sys.argv[2])
```

### Sauvegarde et Récupération

#### 1. Script de Sauvegarde

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/sentiment-analyzer"
APP_DIR="/home/sentiment-analyzer/sentiment-analysis-engine"
DATE=$(date +%Y%m%d_%H%M%S)

# Création du répertoire de sauvegarde
mkdir -p $BACKUP_DIR

# Sauvegarde de l'application
tar -czf $BACKUP_DIR/app_$DATE.tar.gz -C $APP_DIR .

# Sauvegarde des logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz -C /var/log/sentiment-analysis .

# Nettoyage des anciennes sauvegardes (> 30 jours)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $DATE"
```

#### 2. Script de Restauration

```bash
#!/bin/bash
# restore.sh

if [ $# -ne 1 ]; then
    echo "Usage: $0 <backup_date>"
    echo "Example: $0 20231031_120000"
    exit 1
fi

BACKUP_DATE=$1
BACKUP_DIR="/backup/sentiment-analyzer"
APP_DIR="/home/sentiment-analyzer/sentiment-analysis-engine"

# Arrêt du service
sudo systemctl stop sentiment-analyzer.service

# Sauvegarde de l'état actuel
mv $APP_DIR $APP_DIR.backup.$(date +%Y%m%d_%H%M%S)

# Restauration
mkdir -p $APP_DIR
tar -xzf $BACKUP_DIR/app_$BACKUP_DATE.tar.gz -C $APP_DIR

# Restauration des permissions
chown -R sentiment-analyzer:sentiment-analyzer $APP_DIR

# Redémarrage du service
sudo systemctl start sentiment-analyzer.service

echo "Restoration completed from backup: $BACKUP_DATE"
```

### Sécurité

#### 1. Permissions de Fichiers

```bash
# Permissions restrictives sur les fichiers de configuration
chmod 600 config.prod.json

# Permissions sur les répertoires
chmod 755 /home/sentiment-analyzer/sentiment-analysis-engine
chmod 750 /var/log/sentiment-analysis

# Permissions sur les scripts
chmod 750 monitor.sh backup.sh restore.sh
```

#### 2. Pare-feu (si nécessaire)

```bash
# Configuration UFW (Ubuntu)
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Autoriser SSH (ajuster le port si nécessaire)
sudo ufw allow 22/tcp
```

### Dépannage

#### Problèmes Courants

**1. Erreur de mémoire insuffisante**
```bash
# Vérification de la mémoire
free -h

# Ajustement de la taille des lots
# Éditer config.prod.json et réduire batch_size
```

**2. Erreurs d'encodage**
```bash
# Vérification de la locale
locale

# Configuration UTF-8
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
```

**3. Permissions insuffisantes**
```bash
# Vérification des permissions
ls -la /var/log/sentiment-analysis/

# Correction des permissions
sudo chown -R sentiment-analyzer:sentiment-analyzer /var/log/sentiment-analysis/
```

#### Logs de Diagnostic

```bash
# Logs de l'application
tail -f /var/log/sentiment-analysis/app.log

# Logs système
sudo journalctl -u sentiment-analyzer.service -f

# Logs de performance
tail -f /var/log/sentiment-analysis/monitor.log
```

### Mise à Jour

#### Procédure de Mise à Jour

```bash
# 1. Sauvegarde
./backup.sh

# 2. Arrêt du service
sudo systemctl stop sentiment-analyzer.service

# 3. Mise à jour du code
cd /home/sentiment-analyzer/sentiment-analysis-engine
git pull origin main

# 4. Mise à jour des dépendances
source venv/bin/activate
pip install -r requirements.txt --upgrade

# 5. Validation
python validate_installation.py

# 6. Redémarrage
sudo systemctl start sentiment-analyzer.service

# 7. Vérification
sudo systemctl status sentiment-analyzer.service
```

### Support et Maintenance

#### Contacts
- **Support technique** : Consulter les logs détaillés
- **Documentation** : README.md et TECHNICAL_DOCUMENTATION.md
- **Issues** : Utiliser le mode `--verbose` pour diagnostic

#### Maintenance Préventive
- **Hebdomadaire** : Vérification des logs et de l'espace disque
- **Mensuelle** : Mise à jour des dépendances de sécurité
- **Trimestrielle** : Test de restauration des sauvegardes
- **Annuelle** : Audit de sécurité complet
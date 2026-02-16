# Deploying Cerebro

> Multiple deployment options from simple to production-grade

## Quick Start (Recommended)

### Option 1: pip install (Memory Server Only)

For users who just want AI memory for Claude Code. No backend or frontend required.

```bash
pip install cerebro-ai
cerebro init
cerebro doctor  # Verify installation
```

This installs:
- The MCP Memory Server (49 tools)
- Claude Code hook scripts
- Sentence-transformer embeddings model
- FAISS vector index

After installation, restart Claude Code and the memory system is active.

### Option 2: Docker Compose (Full Stack)

For the complete Cerebro experience including the web interface, cognitive loop, and browser automation.

```bash
git clone https://github.com/Professor-Low/Cerebro.git
cd Cerebro
cp .env.example .env
# Edit .env with your preferences (see Configuration below)
docker compose up -d
```

This starts:
- **cerebro-memory**: MCP Memory Server
- **cerebro-backend**: FastAPI + Cognitive Loop
- **cerebro-frontend**: PWA web interface
- **cerebro-redis**: State management
- **cerebro-ollama** (optional): Local LLM

Verify with:
```bash
docker compose ps          # All services running
docker compose logs -f     # Watch logs
curl http://localhost:8000/api/health  # Backend health check
```

### Option 3: Manual Installation

Step-by-step installation for development or custom setups.

#### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend)
- Redis (for backend state)
- Git

#### Memory Server

```bash
cd memory-server
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
python -m cerebro.server  # Start MCP server
```

#### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### Frontend

```bash
cd frontend
npm install
npm run build
npm run preview  # or serve with nginx
```

#### Hook Scripts

```bash
cerebro hooks install  # Auto-configure Claude Code hooks
# or manually copy hook scripts to ~/.claude/hooks/
```

---

## Configuration

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
# ──────────────────────────────────────────
# Storage
# ──────────────────────────────────────────
CEREBRO_MEMORY_PATH=/data/cerebro          # Where to store all memory data
CEREBRO_STORAGE_TYPE=local                  # local | nas | s3

# NAS storage (if STORAGE_TYPE=nas)
CEREBRO_NAS_IP=your-nas-ip
CEREBRO_NAS_SHARE=home
CEREBRO_NAS_MOUNT_POINT=Z:                 # Windows
# CEREBRO_NAS_MOUNT_POINT=/mnt/nas         # Linux

# S3 storage (if STORAGE_TYPE=s3)
# CEREBRO_S3_BUCKET=cerebro-memory
# CEREBRO_S3_REGION=us-east-1
# CEREBRO_S3_PREFIX=memory/

# ──────────────────────────────────────────
# Embeddings
# ──────────────────────────────────────────
CEREBRO_EMBEDDING_MODEL=all-mpnet-base-v2   # Sentence-transformer model
CEREBRO_EMBEDDING_DIM=768                   # Must match model output
CEREBRO_EMBEDDING_DEVICE=cpu               # cpu | cuda
CEREBRO_FAISS_INDEX_TYPE=flat              # flat | ivf | hnsw

# ──────────────────────────────────────────
# Backend
# ──────────────────────────────────────────
CEREBRO_BACKEND_HOST=0.0.0.0
CEREBRO_BACKEND_PORT=8000
CEREBRO_REDIS_URL=redis://localhost:6379/0

# ──────────────────────────────────────────
# LLM (optional)
# ──────────────────────────────────────────
CEREBRO_LLM_PROVIDER=ollama                # ollama | openai | anthropic
CEREBRO_OLLAMA_URL=http://localhost:11434
CEREBRO_OLLAMA_MODEL=llama3.2

# OpenAI (if LLM_PROVIDER=openai)
# CEREBRO_OPENAI_API_KEY=sk-...
# CEREBRO_OPENAI_MODEL=gpt-4

# Anthropic (if LLM_PROVIDER=anthropic)
# CEREBRO_ANTHROPIC_API_KEY=sk-ant-...
# CEREBRO_ANTHROPIC_MODEL=claude-sonnet-4-5-20250929

# ──────────────────────────────────────────
# Hooks
# ──────────────────────────────────────────
CEREBRO_HOOK_TIMEOUT=5000                  # Hook timeout in ms
CEREBRO_SEARCH_LIMIT=5                     # Max search results per hook
CEREBRO_BREAKTHROUGH_AUTO_SAVE=true        # Auto-save breakthroughs

# ──────────────────────────────────────────
# Frontend
# ──────────────────────────────────────────
CEREBRO_FRONTEND_PORT=3000
CEREBRO_SOCKET_URL=http://localhost:8000
```

### Storage Options

| Option | Best For | Pros | Cons |
|--------|----------|------|------|
| **Local filesystem** | Single machine, development | Fast, simple, no network dependency | No multi-device access |
| **NAS/Network storage** | Home lab, multi-device | Shared across devices, large capacity | Network dependency, mount management |
| **S3-compatible** | Cloud deployment, teams | Scalable, durable, accessible anywhere | Latency, cost, requires internet |

---

## Production Deployment

### System Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 2 cores | 4+ cores |
| **RAM** | 4 GB | 8+ GB |
| **Disk** | 10 GB | 50+ GB (grows with conversations) |
| **GPU** | None | NVIDIA (for local embeddings/LLM) |
| **Python** | 3.10 | 3.12+ |
| **OS** | Windows 10, Ubuntu 20.04, macOS 12 | Latest stable |

### Docker Compose (Production)

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  memory:
    image: your-registry/cerebro-memory:latest
    restart: always
    volumes:
      - cerebro-data:/data
    environment:
      - CEREBRO_MEMORY_PATH=/data
      - CEREBRO_EMBEDDING_DEVICE=cpu
    healthcheck:
      test: ["CMD", "python", "-c", "import cerebro; cerebro.health_check()"]
      interval: 30s
      timeout: 10s
      retries: 3

  backend:
    image: your-registry/cerebro-backend:latest
    restart: always
    ports:
      - "8000:8000"
    environment:
      - CEREBRO_REDIS_URL=redis://redis:6379/0
      - CEREBRO_MEMORY_PATH=/data
    volumes:
      - cerebro-data:/data
    depends_on:
      redis:
        condition: service_healthy
      memory:
        condition: service_healthy

  frontend:
    image: your-registry/cerebro-frontend:latest
    restart: always
    ports:
      - "3000:3000"
    environment:
      - CEREBRO_SOCKET_URL=http://backend:8000
    depends_on:
      - backend

  redis:
    image: redis:7-alpine
    restart: always
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

volumes:
  cerebro-data:
  redis-data:
```

### Reverse Proxy (nginx)

```nginx
# /etc/nginx/sites-available/cerebro
server {
    listen 443 ssl http2;
    server_name cerebro.example.com;

    ssl_certificate /etc/letsencrypt/live/cerebro.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/cerebro.example.com/privkey.pem;

    # Frontend
    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Backend API
    location /api/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # Socket.IO
    location /socket.io/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;
    }
}

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name cerebro.example.com;
    return 301 https://$host$request_uri;
}
```

### Systemd Services

#### Memory Server

```ini
# /etc/systemd/system/cerebro-memory.service
[Unit]
Description=Cerebro Memory Server
After=network.target

[Service]
Type=simple
User=cerebro
WorkingDirectory=/opt/cerebro/memory-server
ExecStart=/opt/cerebro/memory-server/venv/bin/python -m cerebro.server
Restart=always
RestartSec=5
Environment=CEREBRO_MEMORY_PATH=/data/cerebro
Environment=CEREBRO_EMBEDDING_DEVICE=cpu

[Install]
WantedBy=multi-user.target
```

#### Backend

```ini
# /etc/systemd/system/cerebro-backend.service
[Unit]
Description=Cerebro Backend
After=network.target redis.service cerebro-memory.service

[Service]
Type=simple
User=cerebro
WorkingDirectory=/opt/cerebro/backend
ExecStart=/opt/cerebro/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=5
Environment=CEREBRO_REDIS_URL=redis://localhost:6379/0
Environment=CEREBRO_MEMORY_PATH=/data/cerebro

[Install]
WantedBy=multi-user.target
```

#### Enable and Start

```bash
sudo systemctl daemon-reload
sudo systemctl enable cerebro-memory cerebro-backend
sudo systemctl start cerebro-memory cerebro-backend
sudo systemctl status cerebro-memory cerebro-backend
```

---

## Monitoring

### Health Checks

Cerebro provides multiple health check mechanisms:

#### CLI

```bash
cerebro doctor
```

Output:
```
Cerebro Health Check
====================
Storage:      OK (~/.cerebro/data, 2.1 TB free)
FAISS Index:  OK (768-dim, 15,234 vectors)
Embeddings:   OK (all-mpnet-base-v2 loaded)
Knowledge:    OK (8,456 facts, avg confidence 0.78)
Learnings:    OK (342 solutions, 89 antipatterns)
MCP Server:   OK (49 tools registered)
====================
Status: HEALTHY
```

#### HTTP Endpoint

```bash
curl http://localhost:8000/api/health
```

```json
{
  "status": "healthy",
  "uptime": "3d 14h 22m",
  "components": {
    "memory_server": "connected",
    "redis": "connected",
    "faiss_index": "loaded",
    "embedding_model": "loaded"
  },
  "metrics": {
    "conversations_stored": 1523,
    "facts_extracted": 8456,
    "search_latency_ms": 45
  }
}
```

#### MCP Tool

```
system_health_check()
```

Returns detailed component diagnostics including NAS connectivity, FAISS index health, memory statistics, and MCP tool availability.

### Logs

| Component | Location | Format |
|-----------|----------|--------|
| Memory Server | `~/.cerebro/logs/memory.log` | JSON structured |
| Backend | `~/.cerebro/logs/backend.log` | JSON structured |
| Hooks | `~/.cerebro/logs/hooks.log` | Plain text |
| Docker | `docker compose logs -f` | Container stdout |

#### Log Configuration

```bash
# Set log level
export CEREBRO_LOG_LEVEL=INFO  # DEBUG | INFO | WARNING | ERROR

# Rotate logs
export CEREBRO_LOG_MAX_SIZE=50M
export CEREBRO_LOG_BACKUPS=5
```

---

## Backup & Recovery

### What to Back Up

| Data | Location | Priority | Size |
|------|----------|----------|------|
| Conversations | `$CEREBRO_MEMORY_PATH/conversations/` | Critical | Grows over time |
| Knowledge Base | `$CEREBRO_MEMORY_PATH/knowledge_base/` | Critical | < 100 MB |
| Learnings | `$CEREBRO_MEMORY_PATH/learnings/` | High | < 50 MB |
| FAISS Index | `$CEREBRO_MEMORY_PATH/embeddings/` | Medium (rebuildable) | 10-500 MB |
| User Profile | `$CEREBRO_MEMORY_PATH/user_profile/` | High | < 10 MB |
| Quick Facts | `$CEREBRO_MEMORY_PATH/quick_facts.json` | High | < 1 MB |
| Skills | `$CEREBRO_MEMORY_PATH/cerebro/skills/` | Medium | < 50 MB |

### Backup Script

```bash
#!/bin/bash
# backup-cerebro.sh
BACKUP_DIR="/backups/cerebro/$(date +%Y-%m-%d)"
MEMORY_PATH="${CEREBRO_MEMORY_PATH:-/data/cerebro}"

mkdir -p "$BACKUP_DIR"

# Critical data (always backup)
tar czf "$BACKUP_DIR/conversations.tar.gz" "$MEMORY_PATH/conversations/"
tar czf "$BACKUP_DIR/knowledge_base.tar.gz" "$MEMORY_PATH/knowledge_base/"
tar czf "$BACKUP_DIR/learnings.tar.gz" "$MEMORY_PATH/learnings/"
tar czf "$BACKUP_DIR/user_profile.tar.gz" "$MEMORY_PATH/user_profile/"
cp "$MEMORY_PATH/quick_facts.json" "$BACKUP_DIR/"

# Optional (rebuildable)
tar czf "$BACKUP_DIR/embeddings.tar.gz" "$MEMORY_PATH/embeddings/"

echo "Backup complete: $BACKUP_DIR"
```

### Restore from Backup

```bash
#!/bin/bash
# restore-cerebro.sh
BACKUP_DIR="$1"
MEMORY_PATH="${CEREBRO_MEMORY_PATH:-/data/cerebro}"

if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: restore-cerebro.sh /path/to/backup/2026-02-11"
    exit 1
fi

# Stop services
systemctl stop cerebro-backend cerebro-memory

# Restore data
tar xzf "$BACKUP_DIR/conversations.tar.gz" -C /
tar xzf "$BACKUP_DIR/knowledge_base.tar.gz" -C /
tar xzf "$BACKUP_DIR/learnings.tar.gz" -C /
tar xzf "$BACKUP_DIR/user_profile.tar.gz" -C /
cp "$BACKUP_DIR/quick_facts.json" "$MEMORY_PATH/"

# Rebuild FAISS index (if not in backup)
cerebro rebuild-index

# Restart services
systemctl start cerebro-memory cerebro-backend

echo "Restore complete. Run 'cerebro doctor' to verify."
```

### Automated Backups

```bash
# Add to crontab: daily backup at 3 AM
0 3 * * * /opt/cerebro/scripts/backup-cerebro.sh >> /var/log/cerebro-backup.log 2>&1
```

---

## Upgrading

### pip Install

```bash
pip install --upgrade cerebro-ai
cerebro migrate  # Run any data migrations
cerebro doctor   # Verify everything works
```

### Docker

```bash
cd Cerebro
git pull
docker compose pull
docker compose down
docker compose up -d
docker compose logs -f  # Watch for migration messages
```

### Manual

```bash
cd /opt/cerebro
git pull

# Memory server
cd memory-server
source venv/bin/activate
pip install -r requirements.txt
python -m cerebro.migrate

# Backend
cd ../backend
source venv/bin/activate
pip install -r requirements.txt
python -m cerebro.migrate

# Frontend
cd ../frontend
npm install
npm run build

# Restart services
sudo systemctl restart cerebro-memory cerebro-backend
```

### Version Compatibility

| From | To | Migration Required | Notes |
|------|----|--------------------|-------|
| 1.x | 2.x | Yes (`cerebro migrate`) | OODA engine restructured, agents replace LLM |
| 2.0 | 2.1+ | Automatic | Minor schema updates handled on startup |

### Rolling Back

If an upgrade causes issues:

```bash
# Docker
docker compose down
git checkout v2.0.0  # or previous version tag
docker compose up -d

# pip
pip install cerebro-ai==2.0.0

# Manual
git checkout v2.0.0
# Re-install dependencies and restart services
```

Always back up before upgrading. Data migrations are forward-only.

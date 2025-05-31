# ncrsh-Swarm API Documentation 📚

**By Dineth Nethsara ([@hexlorddev](https://github.com/hexlorddev))**

Complete API reference for the ncrsh-Swarm distributed neural network framework.

## 🌐 REST API Endpoints

### Authentication
All API endpoints require JWT authentication unless otherwise specified.

```bash
# Get authentication token
curl -X POST http://localhost:8080/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your_password"}'
```

### Node Management

#### `GET /api/v1/nodes`
List all nodes in the swarm

**Response:**
```json
{
  "nodes": [
    {
      "id": "node_123abc",
      "address": "192.168.1.100:8080",
      "status": "active",
      "model_hash": "sha256:def456...",
      "last_seen": "2025-05-31T04:30:00Z",
      "performance": {
        "training_speed": 0.95,
        "memory_usage": 0.67,
        "network_latency": 15.2
      }
    }
  ],
  "total": 1,
  "active": 1
}
```

#### `POST /api/v1/nodes`
Create a new swarm node

**Request:**
```json
{
  "config": {
    "model_config": {
      "hidden_size": 512,
      "num_layers": 6,
      "num_heads": 8
    },
    "network_config": {
      "port": 8080,
      "max_peers": 10
    }
  }
}
```

#### `GET /api/v1/nodes/{node_id}`
Get detailed node information

#### `DELETE /api/v1/nodes/{node_id}`
Remove node from swarm

### Training Management

#### `POST /api/v1/training/start`
Start distributed training session

**Request:**
```json
{
  "dataset": "huggingface:wikitext-103",
  "epochs": 100,
  "batch_size": 32,
  "learning_rate": 0.001,
  "nodes": ["node_123abc", "node_456def"],
  "strategy": "cooperative_sgd"
}
```

#### `GET /api/v1/training/status`
Get current training status

**Response:**
```json
{
  "session_id": "training_789ghi",
  "status": "running",
  "progress": {
    "epoch": 25,
    "total_epochs": 100,
    "loss": 2.34,
    "accuracy": 0.87
  },
  "nodes": [
    {
      "id": "node_123abc",
      "status": "training",
      "batch_progress": 0.75
    }
  ]
}
```

#### `POST /api/v1/training/stop`
Stop current training session

### Model Management

#### `GET /api/v1/models`
List available models

#### `POST /api/v1/models/save`
Save current model state

#### `POST /api/v1/models/load`
Load saved model

#### `GET /api/v1/models/{model_id}/download`
Download model checkpoint

### Network Operations

#### `GET /api/v1/network/topology`
Get current network topology

#### `POST /api/v1/network/discover`
Trigger peer discovery

#### `GET /api/v1/network/metrics`
Get network performance metrics

### Consensus Operations

#### `GET /api/v1/consensus/status`
Get consensus protocol status

#### `POST /api/v1/consensus/vote`
Submit vote for model update

#### `GET /api/v1/consensus/history`
Get consensus decision history

## 🔌 WebSocket API

Connect to real-time events:

```javascript
const ws = new WebSocket('ws://localhost:8080/api/v1/ws');

ws.on('message', (data) => {
  const event = JSON.parse(data);
  console.log('Event:', event.type, event.data);
});
```

### Event Types

- `node_joined`: New node joined the swarm
- `node_left`: Node left the swarm
- `training_progress`: Training progress update
- `consensus_vote`: New consensus vote
- `model_update`: Model state updated

## 🐍 Python SDK

```python
import ncrsh_swarm as swarm

# Initialize client
client = swarm.SwarmClient(
    base_url="http://localhost:8080",
    auth_token="your_jwt_token"
)

# List nodes
nodes = await client.nodes.list()

# Start training
training = await client.training.start(
    dataset="custom_dataset",
    epochs=50,
    batch_size=16
)

# Monitor progress
async for progress in client.training.stream_progress():
    print(f"Epoch: {progress.epoch}, Loss: {progress.loss}")
```

## 📊 Metrics and Monitoring

### Prometheus Metrics

Available at `/metrics` endpoint:

- `swarm_nodes_total`: Total number of nodes
- `swarm_nodes_active`: Active nodes count
- `swarm_training_loss`: Current training loss
- `swarm_network_latency`: Network latency histogram
- `swarm_consensus_votes`: Consensus votes counter

### Health Checks

#### `GET /health`
Basic health check

#### `GET /health/ready`
Readiness probe for Kubernetes

#### `GET /health/live`
Liveness probe for Kubernetes

## 🔐 Security

### Authentication

Uses JWT tokens with RS256 signing:

```bash
# Generate key pair
openssl genrsa -out private.pem 2048
openssl rsa -in private.pem -pubout -out public.pem
```

### Rate Limiting

- Default: 100 requests/minute per IP
- Authenticated: 1000 requests/minute
- Training endpoints: 10 requests/minute

### CORS

Configure CORS in production:

```yaml
cors:
  origins:
    - https://dashboard.yourcompany.com
  methods: [GET, POST, PUT, DELETE]
  headers: [Authorization, Content-Type]
```

## 📖 Error Codes

| Code | Message | Description |
|------|---------|-------------|
| 1001 | NODE_NOT_FOUND | Specified node ID not found |
| 1002 | TRAINING_IN_PROGRESS | Cannot start training, session active |
| 1003 | CONSENSUS_FAILED | Consensus protocol failed |
| 1004 | MODEL_CORRUPT | Model checkpoint corrupted |
| 1005 | NETWORK_PARTITION | Network partition detected |

## 🔧 Configuration

API server configuration:

```yaml
api:
  host: "0.0.0.0"
  port: 8080
  cors_enabled: true
  rate_limit: 100
  jwt_secret: "your_secret_key"
  
authentication:
  provider: "jwt"
  token_expiry: "24h"
  
logging:
  level: "INFO"
  format: "json"
```

## 📚 Additional Resources

- [Python SDK Documentation](./python-sdk.md)
- [JavaScript SDK Documentation](./js-sdk.md)
- [OpenAPI Specification](./openapi.yaml)
- [Postman Collection](./postman-collection.json)
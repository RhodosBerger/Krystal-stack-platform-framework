# APPLICATION LAYERS ARCHITECTURE - BUILDER PATTERN
## Complete API Interface Layers for Fanuc Rise Platform

---

## 🏗️ ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────┐
│               PRESENTATION LAYER                     │
│  ┌─────────┐  ┌─────────┐  ┌──────────┐            │
│  │ REST API│  │WebSocket│  │ GraphQL  │            │
│  └────┬────┘  └────┬────┘  └────┬─────┘            │
└───────┼───────────┼─────────────┼──────────────────┘
        │           │             │
┌───────┼───────────┼─────────────┼──────────────────┐
│       │    API GATEWAY LAYER    │                   │
│  ┌────▼────────────▼─────────────▼──────┐          │
│  │  Router │ Auth │ Rate Limit │ Cache  │          │
│  └────┬────────────────────────────┬────┘          │
└───────┼────────────────────────────┼───────────────┘
        │                            │
┌───────┼────────────────────────────┼───────────────┐
│       │     SERVICE LAYER          │               │
│  ┌────▼────┐  ┌──────┐  ┌──────────▼────┐         │
│  │Business │  │DTO   │  │ Validators    │         │
│  │Logic    │  │Mapper│  │               │         │
│  └────┬────┘  └──────┘  └───────────────┘         │
└───────┼─────────────────────────────────────────────┘
        │
┌───────┼─────────────────────────────────────────────┐
│       │   DATA ACCESS LAYER (Repository Pattern)    │
│  ┌────▼────┐  ┌──────┐  ┌──────────┐               │
│  │Repository│  │Query │  │ Cache    │               │
│  │          │  │Builder│  │ Strategy │               │
│  └────┬────┘  └──────┘  └──────────┘               │
└───────┼─────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────┐
│              DATABASE LAYER                          │
│  ┌─────────┐  ┌────────┐  ┌──────────┐             │
│  │SQLite/  │  │Redis   │  │TimescaleDB│             │
│  │Postgres │  │Cache   │  │(Telemetry)│             │
│  └─────────┘  └────────┘  └──────────┘             │
└──────────────────────────────────────────────────────┘
```

---

## LAYER 1: PRESENTATION LAYER

### Components to Build:

#### 1.1 REST API Controllers
```
erp/api/
├── v1/
│   ├── __init__.py
│   ├── machine_controller.py
│   ├── project_controller.py
│   ├── tool_controller.py
│   ├── job_controller.py
│   └── analytics_controller.py
└── v2/  # Future API version
```

#### 1.2 WebSocket Handlers
```
erp/websockets/
├── __init__.py
├── telemetry_handler.py
├── chat_handler.py
└── notification_handler.py
```

#### 1.3 GraphQL Schema (Optional)
```
erp/graphql/
├── schema.py
├── queries.py
└── mutations.py
```

---

## LAYER 2: API GATEWAY

### Components:

#### 2.1 Authentication Middleware
```python
# erp/middleware/auth_middleware.py
- JWT validation
- API key verification
- Session management
- Permission checks
```

#### 2.2 Rate Limiting
```python
# erp/middleware/rate_limit.py
- Per-user limits
- Per-IP limits
- Endpoint-specific limits
```

#### 2.3 Request/Response Interceptors
```python
# erp/middleware/interceptors.py
- Logging
- Error handling
- Response formatting
- CORS handling
```

---

## LAYER 3: SERVICE LAYER (Business Logic)

### Components:

#### 3.1 Domain Services
```
erp/services/
├── __init__.py
├── machine_service.py      # Machine operations
├── telemetry_service.py    # Real-time data processing
├── dopamine_service.py     # AI decision making
├── economics_service.py    # Cost calculations
├── oee_service.py          # OEE calculations
├── scheduling_service.py   # Job scheduling
└── llm_service.py          # LLM integrations
```

#### 3.2 DTOs (Data Transfer Objects)
```
erp/dto/
├── __init__.py
├── machine_dto.py
├── telemetry_dto.py
├── job_dto.py
└── response_dto.py
```

#### 3.3 Validators
```
erp/validators/
├── __init__.py
├── machine_validator.py
├── project_validator.py
└── custom_validators.py
```

---

## LAYER 4: DATA ACCESS (Repository Pattern)

### Components:

#### 4.1 Repositories
```
erp/repositories/
├── __init__.py
├── base_repository.py
├── machine_repository.py
├── telemetry_repository.py
├── project_repository.py
└── cache_repository.py
```

#### 4.2 Query Builders
```
erp/query/
├── __init__.py
├── telemetry_query.py
├── analytics_query.py
└── aggregation_builder.py
```

---

## LAYER 5: CROSS-CUTTING CONCERNS

### Components:

#### 5.1 Error Handling
```
erp/exceptions/
├── __init__.py
├── custom_exceptions.py
├── error_codes.py
└── error_handler.py
```

#### 5.2 Logging
```
erp/logging/
├── __init__.py
├── logger_config.py
└── audit_logger.py
```

#### 5.3 Caching Strategy
```
erp/cache/
├── __init__.py
├── redis_cache.py
├── memory_cache.py
└── cache_decorators.py
```

---

## BUILDER CHECKLIST (Like Web Dev Builder)

### Phase 1: Foundation (Week 1)
- [ ] Create directory structure
- [ ] Setup base repository pattern
- [ ] Implement DTO mappers
- [ ] Create error handling framework
- [ ] Setup logging infrastructure

### Phase 2: Service Layer (Week 2)
- [ ] Machine service with CRUD
- [ ] Telemetry service with streaming
- [ ] Dopamine service integration
- [ ] Economics calculation service
- [ ] OEE service implementation

### Phase 3: API Gateway (Week 3)
- [ ] JWT authentication middleware
- [ ] Rate limiting implementation
- [ ] Request/response interceptors
- [ ] CORS configuration
- [ ] API versioning setup

### Phase 4: Presentation (Week 4)
- [ ] REST API controllers v1
- [ ] WebSocket handlers
- [ ] Swagger/OpenAPI documentation
- [ ] GraphQL schema (optional)
- [ ] Testing endpoints

### Phase 5: Optimization (Week 5)
- [ ] Redis caching layer
- [ ] Database query optimization
- [ ] Load testing
- [ ] Performance monitoring
- [ ] Documentation completion

---

## BUILDER PATTERNS TO IMPLEMENT

### Pattern 1: Repository Pattern
```python
class BaseRepository:
    def get_by_id(id)
    def get_all()
    def create(entity)
    def update(entity)
    def delete(id)
    def find_by(criteria)
```

### Pattern 2: Service Pattern
```python
class BaseService:
    def __init__(repository)
    def execute(command)
    def validate(dto)
    def map_to_dto(model)
    def map_from_dto(dto)
```

### Pattern 3: Factory Pattern
```python
class ServiceFactory:
    @staticmethod
    def create_machine_service()
    def create_telemetry_service()
    def create_llm_service()
```

### Pattern 4: Strategy Pattern
```python
class CacheStrategy:
    RedisStrategy
    MemoryStrategy
    HybridStrategy
```

### Pattern 5: Observer Pattern
```python
class EventBus:
    def subscribe(event, handler)
    def publish(event, data)
    def unsubscribe(event, handler)
```

---

*Architecture Blueprint - Ready for Implementation*

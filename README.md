# ORCA — Global AI Agricultural Procurement Agent

An enterprise-grade, conversational agricultural procurement agent connecting farmers to structured commercial procurement, authoritative pricing, deterministic billing, payment execution, and logistics collection.

Based on **Global AI Agricultural Procurement Agent SRS v2.0**.

---

## 🎯 Guiding Principles

> **"Global by architecture. Regional by configuration. AI for conversation. Backend for truth and execution."**

1. **Global by Architecture**: Core conversation state, order lifecycle, billing engine, and collection workflows are geography-agnostic.
2. **Regional by Configuration**: Currencies, measurement units, tax schemes, regional rate tables, and provider adapters are configured per regional profile.
3. **AI for Conversation**: LLMs handle natural language comprehension, produce extraction, and targeted clarification questions.
4. **Backend for Truth**: Authoritative rates, arithmetic calculations, financial state mutations, and runner assignments are executed strictly by deterministic backend services. The AI never fabricates prices or modifies database state directly.

---

## 🏗️ Project Architecture (Modular Monolith)

The system is structured as a modular monolith in `src/orca/`:

```
orca/
├── docs/
│   └── ARCHITECTURE.md          # Comprehensive architecture & design document
├── src/
│   └── orca/
│       ├── core/                # System settings & regional configuration profiles
│       │   ├── config.py        # Environment settings (Pydantic Settings)
│       │   └── regional.py      # Regional profiles (units, currency, tax, pricing rules)
│       ├── domain/              # Core business entities & invariants
│       │   ├── state_machine.py # 13-state Order State Machine & transition graph
│       │   ├── models.py        # Pure domain entities (Farmer, Order, Rate, etc.)
│       │   └── schemas.py       # Pydantic validation schemas & DTOs
│       ├── services/            # Pure deterministic business logic
│       │   ├── pricing.py       # Authoritative rate lookup (no LLM price hallucination)
│       │   ├── billing.py       # Deterministic billing calculation
│       │   ├── order.py         # Order lifecycle & idempotency control
│       │   ├── payment.py       # Payment initiation & status tracking
│       │   └── collection.py    # Logistics task creation & runner assignment
│       ├── agent/               # Conversational AI layer
│       │   ├── prompts.py       # Layered English system prompts
│       │   └── tools.py         # Approved tool definitions (SRS Section 8)
│       ├── adapters/            # External integration boundaries
│       │   ├── messaging/       # Channel adapters (WhatsApp, Webhook, Mock)
│       │   ├── payment/         # Payment provider adapters (Sandbox, Gateway)
│       │   └── logistics/       # Runner & collection dispatch adapters
│       ├── db/                  # Persistence layer
│       │   └── session.py       # SQLAlchemy 2.0 async engine & session management
│       └── api/                 # Web API & Webhooks
│           └── app.py           # FastAPI application & endpoints
├── tests/
│   ├── unit/                    # Unit test suite
│   │   ├── test_state_machine.py
│   │   └── test_pricing_billing.py
│   └── conftest.py
├── pyproject.toml               # Python project configuration (PEP 621)
└── README.md
```

---

## 🔄 Order State Machine

The transaction strictly progresses through the 13 states defined in SRS Section 10:

```
OFFER_RECEIVED ──► DETAILS_PENDING ──► RATE_VALIDATED ──► AWAITING_FARMER_CONFIRMATION
                                                                    │
                                                                    ▼
COMPLETED ◄── PICKED_UP ◄── COLLECTION_ASSIGNED ◄── COLLECTION_PENDING ◄── PAYMENT_CONFIRMED ◄── PAYMENT_PENDING ◄── ORDER_CONFIRMED
```

*(Terminal states: `COMPLETED`, `CANCELLED`. Exceptions: `EXCEPTION`)*

---

## 🚀 Quickstart & Verification

### Prerequisites
- Python 3.12+
- `uv` (recommended) or standard `venv` + `pip`

### 1. Install Dependencies
```bash
uv pip install -e ".[dev]"
```

### 2. Run Test Suite
```bash
pytest
```

### 3. Run FastAPI Application
```bash
uvicorn orca.api.app:app --reload --port 8000
```
- API Docs: `http://localhost:8000/docs`
- Health Probe: `http://localhost:8000/health`

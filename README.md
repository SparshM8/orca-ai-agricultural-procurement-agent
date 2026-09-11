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

## 🖥️ Presentation & Operational Portals

ORCA provides an enterprise-grade single-page application served at `http://localhost:8008/` with three synchronized portals:

1. **🌾 Farmer Chat** (Primary Experience):
   - Conversational intake for produce offers, quantities, pickup availability, and inquiries.
   - Natural language comprehension with animated typing indicators and responsive SaaS layout.
   - Grounded pricing negotiation rationale and isolated candidate offer amendments.
   - One-click confirmation and instant sandbox payout execution.
   - Live order fulfillment tracking (Order ID, Payment Status, Logistics Task, and Human Escalation badge).
   - Instant quick-action pills for 1-click execution of all 14 capabilities (50kg Potato offer, negotiate, amend, confirm, pay, pickup FAQ, catalog, personalized recommendation, human support).
2. **🚚 Runner Console**:
   - Operational task board with two distinct views: *Available Pickups* (unassigned pool) and *My Active Pickups* (fleet-assigned).
   - Dedicated fleet selectors (`Runner-01`, `Runner-02`, `Runner-03`).
   - Task lifecycle execution: Accept, Reject, Confirm Pickup, Report Failure, and Reschedule (with styled modal dialogs).
3. **📊 Admin Dashboard**:
   - Real-time executive KPIs: Total Procurement Orders, Authoritative GMV, Active Logistics Tasks, Completed Orders, and Open Human Escalations.
   - Dedicated sub-navigation tabs:
     - 📦 **Procurement Orders**: Full order registry with search, status filters, and deep-dive Inspection Drawer.
     - 💳 **Payments**: Authoritative disbursement logs with provider references and timestamps.
     - 🚚 **Logistics Tasks**: Fleet assignment, pickup windows, and runner status.
     - ⚠️ **Human Escalations**: Operational exception records with 1-click resolution actions.
   - Comprehensive **Order Audit Drawer**: 8-step chronological progression stepper, deterministic payout verification, collection task details, conversational transcript, and observational **Agent Trace & Decision Audit**.
   - One-click **🧹 Reset Demo Data** button to clear demo data and return to baseline rates in 1 second.

---

## 🧠 Autonomous Intelligence & Deterministic Guardrails

1. **Farmer Conversational Procurement**: Natural language intake, multi-turn amendment, and pricing inquiries.
2. **Deterministic Pricing Service**: Strict authoritative price lookups (`pricing_service`). Zero LLM pricing hallucinations.
3. **Billing Service**: Exact arithmetic calculations, regional tax schemes, and transparent itemized bills.
4. **Order State Machine**: 13-state deterministic lifecycle (`OrderState`) with atomic transitions and idempotency locks.
5. **Payment Service**: Multi-turn payout initiation with sandbox adapter and deterministic provider reference verification.
6. **Collection Logistics**: Automated pickup task dispatching, fleet assignment, rescheduling, and failure routing.
7. **Admin Dashboard**: Real-time KPI aggregation, audit drawers, and operational oversight.
8. **Runner Console**: Mobile-ready dispatcher board for physical pickups, reschedule requests, and failure reports.
9. **AgentTrace Observability**: Structured diagnostic logs for every agent interaction with recursive credential redaction.
10. **One-Click Demo Reset**: In-memory and persistence reset to clean baseline in < 1 second.
11. **Business Knowledge Layer**: Grounded, read-only operational knowledge service (`BusinessKnowledgeService`) answering policies and FAQs without external vector DBs or RAG hallucinations.
12. **Grounded Recommendations**: Catalog exploration grounded exclusively in ORCA's verified supported crops and active rates.
13. **Factual Personalization**: Minimal observable profile tracking explicit farmer offers, completed orders, and stated crop preferences. Never predicts or infers wealth, farm size, or risk scores.
14. **Human Handoff & Exception Routing**: Deterministic operational escalation (`HumanHandoffService`) for disputes, collection failures, or explicit farmer requests with zero LLM state mutation.

---

## 🚀 Quickstart & Verification

### Prerequisites
- Python 3.12+
- `uv` (recommended) or standard `venv` + `pip`

### 1. Install Dependencies
```bash
uv pip install -e ".[dev]"
```

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Configure your Gemini API key (optional — falls back to rule-based classification automatically if unset):
```env
GEMINI_API_KEY=your_gemini_api_key_here
AI_PROVIDER=gemini
AI_MODEL_NAME=gemini-2.5-flash
```

### 3. Run Test Suite (340 / 340 Passing)
Execute the complete test suite:
```bash
python -m pytest tests -q
```
*Current test suite baseline: **352 passed in ~16s** (100% pass rate with zero regressions).*

---

## 🚀 Running the Application

### Local Development
Launch the FastAPI server on port 8008:
```bash
uvicorn orca.api.app:app --host 127.0.0.1 --port 8008 --reload
```

- **Presentation UI**: `http://localhost:8008/` or `http://localhost:8008/ui`
- **Interactive Swagger Docs**: `http://localhost:8008/docs`
- **System Health Probe**: `http://localhost:8008/health`

### Render Cloud Deployment (Public Stakeholder Demo)
> **Public stakeholder demo deployment.**
> **Sandbox environment — no real payments.**

ORCA includes native [Render](https://render.com) Blueprint infrastructure (`render.yaml`) and `Procfile` for one-click stakeholder demo hosting.

**Cloud Start Command:**
```bash
uvicorn orca.api.app:app --host 0.0.0.0 --port $PORT
```

**Cloud Environment Variables:**
| Variable | Value / Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | *(Provided by Render)* | Dynamically assigned HTTP port |
| `HOST` | `0.0.0.0` | Public interface binding |
| `PYTHON_VERSION` | `3.12.0` | Python runtime version |
| `AI_PROVIDER` | `rule_based` | Deterministic intent classification |
| `AI_FALLBACK_ON_FAILURE` | `true` | Automatic rule-based fallback |
| `ENVIRONMENT` | `production` | Deployment environment label |
| `DEBUG` | `false` | Debug mode disabled |
| `DEFAULT_REGION` | `GLOBAL_DEFAULT` | Default regional procurement configuration |
| `GEMINI_API_KEY` | *(Optional)* | Optional Gemini 2.5 Flash API key |
| `DEMO_ADMIN_KEY` | *(Optional)* | Optional token protecting admin reset |

**Public Deployment Notes & Limitations:**
- **Ephemeral Storage**: Render free-tier instances use ephemeral container storage. On cold start or redeploy, ORCA's lifespan service automatically initializes the database schema and seeds the guided demo scenarios (`Hero Procurement`, `FAQ & Recommendations`, `Human Support`).
- **Deterministic High Availability**: In standard demo mode, ORCA runs with `AI_PROVIDER=rule_based`, ensuring 100% availability without external API quotas, network timeouts, or rate limits.
- **Sandbox Scope**: Financial transactions and logistics dispatches are executed through sandboxed mock adapters. Do NOT treat this public stakeholder demo deployment as production banking or logistics infrastructure.

---

## 🎬 10-Minute Live Demo Walkthrough

Follow this scripted path for a comprehensive stakeholder presentation:

1. **Step 0 — Clean Baseline**:
   - Open `http://localhost:8008/`.
   - Click **🧹 Reset Demo Data** in the top navigation header.
   - Verify GMV is `$0.00`, order count is `0`, and open escalations is `0`.
2. **Step 1 — Conversational Intake (Farmer Chat)**:
   - Click quick-action pill `🥔 50kg Potatoes in Nairobi` (or type: *"I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM"*).
   - Observe the typing indicator. The agent returns authoritative pricing: `50 kg @ USD 0.40/kg = USD 20.00`.
3. **Step 2 — Grounded Price Negotiation**:
   - Click `💰 Negotiate: Can you pay $0.45?`.
   - Observe the agent's grounded explanation: rates are benchmarked to wholesale market averages and include guaranteed immediate payout.
4. **Step 3 — Candidate Amendment**:
   - Click `✏️ Amend: Change to 80 kg`.
   - Agent recalculates: `80 kg @ USD 0.40/kg = USD 32.00`.
5. **Step 4 — Confirmation & Instant Payout**:
   - Click `✅ Confirm`. Order moves to `ORDER_CONFIRMED` -> `PAYMENT_PENDING`.
   - Click `💳 Pay`. Payment executes via sandbox -> `PAYMENT_CONFIRMED`.
   - Collection task is automatically dispatched to the logistics pool (`COLLECTION_PENDING`).
6. **Step 5 — Logistics Runner Execution (Runner Console)**:
   - Switch to **🚚 Runner Console** tab (`Runner-01`).
   - Click **✅ Accept Task** in *Available Pickups*. Task moves to *My Active Pickups* (`COLLECTION_ASSIGNED`).
   - Click **✅ Confirm Pickup**. State machine completes transaction (`COMPLETED`).
7. **Step 6 — Business Knowledge & Grounded Recommendations**:
   - Return to **🌾 Farmer Chat**.
   - Click `ℹ️ FAQ: Pickup`: Agent explains the collection and physical inspection process.
   - Click `🌾 Produce Catalog`: Agent returns verified supported produce (potato, tomato, onion).
   - Click `✨ Recommendations`: Agent observes completed sale of 80 kg potatoes and returns a grounded recommendation citing prior sale history.
8. **Step 7 — Operational Exception & Human Escalation**:
   - Click `🧑‍💼 Human Support` (or type: *"I want to talk to a human"*).
   - Agent creates support case (e.g. `HC-XXXXXX`) and updates the sidebar *Support Escalation* status badge.
   - Active offer and order states remain strictly preserved.
9. **Step 8 — Operations & Admin Oversight (Admin Dashboard)**:
   - Switch to **📊 Admin Dashboard**.
   - Observe KPIs: Total Orders: `1`, GMV: `$32.00`, Completed: `1`, Escalations: `1`.
   - Switch subtabs to inspect **💳 Payments**, **🚚 Logistics Tasks**, and **⚠️ Human Escalations**.
   - Click **Resolve** on the escalation case to mark it `RESOLVED`.
   - Inspect the order's **🛡️ Agent Trace & Decision Audit** to showcase intent confidence, classifier name, fallback status, and sanitized arguments.
10. **Step 9 — Teardown**:
   - Click **🧹 Reset Demo Data** to return to a clean baseline.


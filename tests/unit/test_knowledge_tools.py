"""Unit tests for knowledge intent mapping, tool execution, and safety.

Verifies:
1. INQUIRE_BUSINESS_INFO mapping to get_business_knowledge
2. INQUIRE_OPERATIONAL_FAQ mapping to get_business_knowledge
3. get_business_knowledge returns valid KnowledgeResult dict
4. Correct article selection for company inquiries
5. Correct article selection for operational FAQs
6. Unknown question returns non-answerable result
7. Weak queries do not produce false answers
8. Zero hardcoded rates in responses
9. Zero order mutation
10. Zero payment mutation
11. Zero collection mutation
12. Zero OrderState mutation
13. Existing explain_requirement tool behavior remains intact
14. Arbitrary / unapproved tool names rejected
15. AgentTrace safely records knowledge tool usage
"""

import pytest
import re
from orca.domain.intents import (
    IntentType,
    ExtractedEntities,
    ConversationalIntentResult,
)
from orca.domain.knowledge import KnowledgeCategory, KnowledgeResult
from orca.agent.intent.mapping import IntentToToolMapper
from orca.agent.tools import tool_registry
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service
from orca.domain.trace import AgentTrace


# -----------------------------------------------------------------------------
# 1-2. Tool Mapping Tests
# -----------------------------------------------------------------------------
def test_inquire_business_info_mapping():
    intent_res = ConversationalIntentResult(
        intent=IntentType.INQUIRE_BUSINESS_INFO,
        confidence=0.95,
        entities=ExtractedEntities(knowledge_topic="COMPANY_INFO"),
        raw_query="What is ORCA and how does it work?",
    )
    plan = IntentToToolMapper.map_intent_to_tool(intent_res)
    assert plan.intent == IntentType.INQUIRE_BUSINESS_INFO
    assert plan.tool_name == "get_business_knowledge"
    assert plan.is_state_advancing is False
    assert plan.arguments["query"] == "What is ORCA and how does it work?"
    assert plan.arguments["topic"] == "COMPANY_INFO"


def test_inquire_operational_faq_mapping():
    intent_res = ConversationalIntentResult(
        intent=IntentType.INQUIRE_OPERATIONAL_FAQ,
        confidence=0.92,
        entities=ExtractedEntities(),
        raw_query="When will the runner arrive?",
    )
    plan = IntentToToolMapper.map_intent_to_tool(intent_res)
    assert plan.intent == IntentType.INQUIRE_OPERATIONAL_FAQ
    assert plan.tool_name == "get_business_knowledge"
    assert plan.is_state_advancing is False
    assert plan.arguments["query"] == "When will the runner arrive?"
    assert plan.arguments["topic"] is None


# -----------------------------------------------------------------------------
# 3-5. Tool Execution & Knowledge Selection
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_business_knowledge_returns_knowledge_result():
    tool_res = await tool_registry.execute_tool(
        "get_business_knowledge",
        {"query": "What is ORCA?"},
    )
    assert tool_res.tool_name == "get_business_knowledge"
    assert tool_res.success is True
    assert tool_res.data["matched"] is True
    assert tool_res.data["answerable"] is True
    assert tool_res.data["source_article_id"] == "kb_orca_overview"
    assert tool_res.data["category"] == "COMPANY_INFO"
    assert "direct digital agricultural procurement platform" in tool_res.data["answer"].lower()

    # Validates against Pydantic KnowledgeResult schema
    k_result = KnowledgeResult.model_validate(tool_res.data)
    assert k_result.matched is True
    assert k_result.category == KnowledgeCategory.COMPANY_INFO


@pytest.mark.asyncio
async def test_correct_article_selected_for_company_question():
    tool_res = await tool_registry.execute_tool(
        "get_business_knowledge",
        {"query": "Tell me about the ORCA procurement platform", "topic": "COMPANY_INFO"},
    )
    assert tool_res.success is True
    assert tool_res.data["source_article_id"] == "kb_orca_overview"
    assert tool_res.data["category"] == "COMPANY_INFO"


@pytest.mark.asyncio
async def test_correct_article_selected_for_operational_faq():
    # Runner / Pickup status FAQ
    runner_res = await tool_registry.execute_tool(
        "get_business_knowledge",
        {"query": "When will the runner arrive to pick up my produce?"},
    )
    assert runner_res.success is True
    assert runner_res.data["source_article_id"] == "kb_pickup_status"
    assert runner_res.data["category"] == "LOGISTICS_POLICY"

    # Payment FAQ
    payment_res = await tool_registry.execute_tool(
        "get_business_knowledge",
        {"query": "When do I get paid?"},
    )
    assert payment_res.success is True
    assert payment_res.data["source_article_id"] == "kb_payment_workflow"
    assert payment_res.data["category"] == "PAYMENT_POLICY"

    # Dispute / exception FAQ
    dispute_res = await tool_registry.execute_tool(
        "get_business_knowledge",
        {"query": "What happens if the runner does not show up?"},
    )
    assert dispute_res.success is True
    assert dispute_res.data["source_article_id"] == "kb_dispute_exception_handling"
    assert dispute_res.data["category"] == "DISPUTE_RESOLUTION"


# -----------------------------------------------------------------------------
# 6-7. Non-Answerable & Weak Query Behavior
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unknown_question_returns_non_answerable_result():
    tool_res = await tool_registry.execute_tool(
        "get_business_knowledge",
        {"query": "What is the capital of Mars?"},
    )
    assert tool_res.success is False
    assert tool_res.data["matched"] is False
    assert tool_res.data["answerable"] is False
    assert tool_res.data["clarification_required"] is True
    assert tool_res.data["source_article_id"] is None
    assert tool_res.data["answer"] is None


@pytest.mark.asyncio
async def test_weak_query_does_not_produce_false_knowledge_answer():
    weak_queries = ["", "   ", "a", "the", "hello", "ok", "stuff", "yes"]
    for q in weak_queries:
        tool_res = await tool_registry.execute_tool(
            "get_business_knowledge",
            {"query": q},
        )
        assert tool_res.success is False
        assert tool_res.data["matched"] is False
        assert tool_res.data["answerable"] is False
        assert tool_res.data["clarification_required"] is True
        assert tool_res.data["answer"] is None


# -----------------------------------------------------------------------------
# 8. No Hardcoded Rates
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_hardcoded_rates_in_knowledge_tool_output():
    queries = [
        "Why is the rate fixed?",
        "How is billing calculated?",
        "What is ORCA?",
        "How do payments work?",
    ]
    price_pattern = re.compile(r"\$\s*\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s*(?:usd|kes|inr|cents?|dollars?)\b", re.IGNORECASE)

    for q in queries:
        tool_res = await tool_registry.execute_tool("get_business_knowledge", {"query": q})
        if tool_res.data.get("answer"):
            matches = price_pattern.findall(tool_res.data["answer"])
            assert not matches, f"Found hardcoded price in answer for '{q}': {matches}"


# -----------------------------------------------------------------------------
# 9-12. Tool Safety: Zero Mutation Invariants
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_zero_order_mutation_via_tool():
    initial_orders = list(order_service._orders.keys())
    
    await tool_registry.execute_tool("get_business_knowledge", {"query": "What is ORCA?"})
    await tool_registry.execute_tool("get_business_knowledge", {"query": "How do I sell produce?"})
    
    after_orders = list(order_service._orders.keys())
    assert initial_orders == after_orders


@pytest.mark.asyncio
async def test_zero_payment_mutation_via_tool():
    initial_payments = list(payment_service._payments.keys())
    
    await tool_registry.execute_tool("get_business_knowledge", {"query": "When do I get paid?"})
    
    after_payments = list(payment_service._payments.keys())
    assert initial_payments == after_payments


@pytest.mark.asyncio
async def test_zero_collection_mutation_via_tool():
    initial_tasks = list(collection_service._tasks.keys())
    
    await tool_registry.execute_tool("get_business_knowledge", {"query": "Who is the runner?"})
    
    after_tasks = list(collection_service._tasks.keys())
    assert initial_tasks == after_tasks


@pytest.mark.asyncio
async def test_zero_order_state_mutation_via_tool():
    test_order = order_service.create_order(
        farmer_id="farmer_tool_safe_test",
        produce_type="potato",
        quantity=60.0,
        unit="kg",
        pickup_location="West Valley Farm",
        validated_rate=0.40,
        region_code="GLOBAL_DEFAULT",
    )
    initial_status = test_order.status

    await tool_registry.execute_tool("get_business_knowledge", {"query": "How does pickup work?"})
    await tool_registry.execute_tool("get_business_knowledge", {"query": "Why is rate fixed?"})

    refetched = order_service.get_order(test_order.id)
    assert refetched is not None
    assert refetched.status == initial_status


# -----------------------------------------------------------------------------
# 13. Existing explain_requirement Backward Compatibility
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_existing_explain_requirement_intact():
    # Pricing explanation
    price_res = await tool_registry.execute_tool(
        "explain_requirement",
        {"subject": "pricing", "region_code": "GLOBAL_DEFAULT"},
    )
    assert price_res.success is True
    assert price_res.data["topic"] == "pricing"
    assert "rates" in price_res.data
    assert "authoritative procurement purchase rates" in price_res.data["explanation"].lower()

    # Availability window explanation
    avail_res = await tool_registry.execute_tool(
        "explain_requirement",
        {"subject": "availability window", "region_code": "GLOBAL_DEFAULT"},
    )
    assert avail_res.success is True
    assert avail_res.data["topic"] == "availability_window"
    assert "availability window" in avail_res.data["explanation"].lower()

    # General explanation
    general_res = await tool_registry.execute_tool(
        "explain_requirement",
        {"subject": "general", "region_code": "GLOBAL_DEFAULT"},
    )
    assert general_res.success is True
    assert general_res.data["topic"] == "general"
    assert "supported_produce" in general_res.data


# -----------------------------------------------------------------------------
# 14. Arbitrary Tool Names Rejected
# -----------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_arbitrary_tool_names_rejected():
    invalid_tools = [
        "fake_knowledge_tool",
        "custom_query_tool",
        "llm_evaluator",
        "execute_script",
    ]
    for bad_tool in invalid_tools:
        res = await tool_registry.execute_tool(bad_tool, {"query": "What is ORCA?"})
        assert res.success is False
        assert f"Tool '{bad_tool}' is not an approved backend tool." in res.error_message


# -----------------------------------------------------------------------------
# 15. AgentTrace Records Knowledge Tool Usage Safely
# -----------------------------------------------------------------------------
def test_agent_trace_records_knowledge_tool_usage_safely():
    trace = AgentTrace.create_sanitized(
        conversation_id="conv_kb_trace_test",
        sender_id="farmer_trace_123",
        state_before="OFFER_RECEIVED",
        state_after="OFFER_RECEIVED",
        input_text="What is ORCA and how does the platform work?",
        detected_intent="INQUIRE_BUSINESS_INFO",
        confidence=0.98,
        selected_tool="get_business_knowledge",
        tool_arguments={"query": "What is ORCA?", "topic": "COMPANY_INFO"},
        tool_success=True,
        response_path="deterministic",
    )
    assert trace.detected_intent == "INQUIRE_BUSINESS_INFO"
    assert trace.selected_tool == "get_business_knowledge"
    assert trace.tool_success is True
    assert trace.state_before == "OFFER_RECEIVED"
    assert trace.state_after == "OFFER_RECEIVED"
    assert trace.input_text == "What is ORCA and how does the platform work?"
    assert trace.tool_arguments["topic"] == "COMPANY_INFO"

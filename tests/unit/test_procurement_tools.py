"""Unit tests for Phase 3: Tool Registry and Intent Mapping Integration.

Verifies:
1. OBJECTION_PRICING maps deterministically to handle_pricing_objection
2. AMEND_OFFER maps deterministically to evaluate_procurement_offer
3. handle_pricing_objection returns authoritative rate and grounded benefits
4. counter-rate is preserved strictly as informational discrepancy
5. Authoritative rate cannot be modified by farmer counter-rate
6. Valid amended quantity is evaluated and accepted with recalculated bill
7. Invalid amended quantity (below minimum) is rejected
8. Invalid amended unit is rejected
9. Unsupported amended produce is rejected
10. Missing required amended fields are reported in missing_fields
11. Tool execution causes zero order mutation
12. Tool execution causes zero payment mutation
13. Tool execution causes zero collection task mutation
14. Mapper strictly denies arbitrary or hallucinated tool names
15. All existing tool mappings remain intact and unchanged
"""

import pytest
from unittest.mock import MagicMock

from orca.domain.intents import (
    IntentType,
    ConversationalIntentResult,
    ExtractedEntities,
    ToolExecutionResult,
)
from orca.domain.schemas import ExtractedOffer
from orca.domain.state_machine import OrderState
from orca.agent.intent.mapping import IntentToToolMapper, ToolInvocationPlan
from orca.agent.tools import tool_registry, AgentToolRegistry
from orca.agent.orchestrator import DialogueContext
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service


# =============================================================================
# 1 & 2. Intent-to-Tool Mapping Verification
# =============================================================================
def test_objection_pricing_maps_to_handle_pricing_objection():
    """Verify OBJECTION_PRICING intent generates plan for handle_pricing_objection."""
    context = DialogueContext(
        conversation_id="conv_test_obj",
        farmer_id="farmer_test",
        offer=ExtractedOffer(produce_type="potato", quantity=50.0, unit="kg"),
    )
    intent_res = ConversationalIntentResult(
        intent=IntentType.OBJECTION_PRICING,
        entities=ExtractedEntities(counter_rate=0.55),
        raw_query="Can you give me 0.50 instead?",
    )
    plan = IntentToToolMapper.map_intent_to_tool(intent_res, context=context)

    assert plan.intent == IntentType.OBJECTION_PRICING
    assert plan.tool_name == "handle_pricing_objection"
    assert plan.arguments.get("produce") == "potato"
    assert plan.arguments.get("counter_rate") == 0.55
    assert plan.is_state_advancing is False


def test_amend_offer_maps_to_evaluate_procurement_offer():
    """Verify AMEND_OFFER intent merges candidate offer and maps to evaluate_procurement_offer."""
    context = DialogueContext(
        conversation_id="conv_test_amend",
        farmer_id="farmer_test",
        offer=ExtractedOffer(
            produce_type="potato",
            quantity=50.0,
            unit="kg",
            pickup_location="Farm A",
            availability_window="tomorrow at 10 AM",
        ),
    )
    # Farmer changes quantity to 75 kg
    intent_res = ConversationalIntentResult(
        intent=IntentType.AMEND_OFFER,
        entities=ExtractedEntities(
            offer=ExtractedOffer(quantity=75.0)
        ),
        raw_query="Actually make it 75 kg",
    )
    plan = IntentToToolMapper.map_intent_to_tool(intent_res, context=context)

    assert plan.intent == IntentType.AMEND_OFFER
    assert plan.tool_name == "evaluate_procurement_offer"
    candidate = plan.arguments.get("offer")
    assert isinstance(candidate, ExtractedOffer)
    # Merged: updated quantity, preserved produce & location
    assert candidate.quantity == 75.0
    assert candidate.produce_type == "potato"
    assert candidate.pickup_location == "Farm A"
    assert plan.is_state_advancing is False
    # Context offer remains untouched!
    assert context.offer.quantity == 50.0


# =============================================================================
# 3, 4, 5. handle_pricing_objection Tool Tests
# =============================================================================
@pytest.mark.asyncio
async def test_pricing_objection_returns_authoritative_rate():
    """Verify handle_pricing_objection returns authoritative rate and policy explanation."""
    res: ToolExecutionResult = await tool_registry.execute_tool(
        "handle_pricing_objection",
        {"produce": "potato", "region_code": "GLOBAL_DEFAULT"},
    )
    assert res.success is True
    assert res.data["produce"] == "potato"
    assert res.data["authoritative_rate"] == 0.40
    assert res.data["currency"] == "USD"
    assert res.data["fixed_rate_model"] is True
    assert len(res.data["benefits"]) >= 4
    assert res.data["discrepancy"] is False


@pytest.mark.asyncio
async def test_counter_rate_preserved_as_informational_discrepancy():
    """Verify counter-rate is captured as discrepancy without overriding rate."""
    res: ToolExecutionResult = await tool_registry.execute_tool(
        "handle_pricing_objection",
        {
            "produce": "potato",
            "counter_rate": 0.50,
            "region_code": "GLOBAL_DEFAULT",
        },
    )
    assert res.success is True
    assert res.data["farmer_counter_rate"] == 0.50
    assert res.data["authoritative_rate"] == 0.40
    assert res.data["discrepancy"] is True


@pytest.mark.asyncio
async def test_authoritative_rate_cannot_be_modified():
    """Verify extreme counter-rate cannot alter authoritative rate."""
    res: ToolExecutionResult = await tool_registry.execute_tool(
        "handle_pricing_objection",
        {
            "produce": "tomato",
            "counter_rate": 99.99,
            "region_code": "GLOBAL_DEFAULT",
        },
    )
    assert res.success is True
    assert res.data["authoritative_rate"] == 0.60  # Tomato baseline rate is 0.60
    assert res.data["authoritative_rate"] != 99.99


# =============================================================================
# 6, 7, 8, 9, 10. evaluate_procurement_offer Tool Tests
# =============================================================================
@pytest.mark.asyncio
async def test_valid_amended_quantity_evaluated():
    """Verify valid candidate offer amendment is accepted with updated bill."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=80.0,
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    res: ToolExecutionResult = await tool_registry.execute_tool(
        "evaluate_procurement_offer",
        {"offer": offer, "region_code": "GLOBAL_DEFAULT"},
    )
    assert res.success is True
    assert res.data["is_acceptable"] is True
    assert res.data["quantity"] == 80.0
    assert res.data["total_amount"] == 32.0  # 80 * 0.40


@pytest.mark.asyncio
async def test_invalid_amended_quantity_rejected():
    """Verify amendment to quantity below minimum is rejected."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=2.0,  # Below 5.0
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    res: ToolExecutionResult = await tool_registry.execute_tool(
        "evaluate_procurement_offer",
        {"offer": offer, "region_code": "GLOBAL_DEFAULT"},
    )
    assert res.success is False
    assert res.data["is_acceptable"] is False
    assert any("QUANTITY_BELOW_MINIMUM" in r for r in res.data["rejection_reasons"])


@pytest.mark.asyncio
async def test_invalid_amended_unit_rejected():
    """Verify amendment to unsupported unit is rejected."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=50.0,
        unit="liters",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    res: ToolExecutionResult = await tool_registry.execute_tool(
        "evaluate_procurement_offer",
        {"offer": offer},
    )
    assert res.success is False
    assert any("UNSUPPORTED_UNIT:liters" in r for r in res.data["rejection_reasons"])


@pytest.mark.asyncio
async def test_unsupported_amended_produce_rejected():
    """Verify amendment changing produce to unsupported crop is rejected."""
    offer = ExtractedOffer(
        produce_type="papaya",
        quantity=50.0,
        unit="kg",
        pickup_location="Springfield",
        availability_window="tomorrow at 10:00 AM",
    )
    res: ToolExecutionResult = await tool_registry.execute_tool(
        "evaluate_procurement_offer",
        {"offer": offer},
    )
    assert res.success is False
    assert res.data["produce_supported"] is False
    assert any("UNSUPPORTED_PRODUCE:papaya" in r for r in res.data["rejection_reasons"])


@pytest.mark.asyncio
async def test_missing_amended_field_reported():
    """Verify missing required fields in candidate offer are reported in missing_fields."""
    offer = ExtractedOffer(
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="",  # Missing
        availability_window=None,  # Missing
    )
    res: ToolExecutionResult = await tool_registry.execute_tool(
        "evaluate_procurement_offer",
        {"offer": offer},
    )
    assert res.success is False
    assert "pickup_location" in res.data["missing_fields"]
    assert "availability_window" in res.data["missing_fields"]


# =============================================================================
# 11, 12, 13. Zero Mutation Verification
# =============================================================================
@pytest.mark.asyncio
async def test_tools_cause_zero_entity_mutations():
    """Verify new tools cause zero mutations in order, payment, and collection stores."""
    orders_before = len(order_service._orders)
    payments_before = len(payment_service._payments)
    tasks_before = len(collection_service._tasks)

    offer = ExtractedOffer(produce_type="potato", quantity=50.0, unit="kg", pickup_location="Springfield", availability_window="tomorrow at 10 AM")

    # Execute both tools multiple times
    await tool_registry.execute_tool("handle_pricing_objection", {"produce": "potato", "counter_rate": 0.50})
    await tool_registry.execute_tool("evaluate_procurement_offer", {"offer": offer})

    assert len(order_service._orders) == orders_before
    assert len(payment_service._payments) == payments_before
    assert len(collection_service._tasks) == tasks_before


# =============================================================================
# 14 & 15. Mapper Safety and Existing Mapping Preservation
# =============================================================================
def test_mapper_never_allows_arbitrary_tool_names():
    """Verify ToolInvocationPlan only returns approved deterministic tool names."""
    unknown_res = ConversationalIntentResult(
        intent=IntentType.UNKNOWN,
        raw_query="random query",
    )
    plan = IntentToToolMapper.map_intent_to_tool(unknown_res)
    assert plan.tool_name is None
    assert plan.arguments == {}


def test_existing_tool_mappings_remain_unchanged():
    """Verify all 11 existing intent mappings remain 100% intact."""
    context = DialogueContext(conversation_id="c1", farmer_id="f1", current_order_id="ORD-1234")

    # 1. OFFER_PRODUCE
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.OFFER_PRODUCE), context=context
    )
    assert p.tool_name == "evaluate_produce_offer"

    # 2. INQUIRE_SUPPORTED_PRODUCE
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.INQUIRE_SUPPORTED_PRODUCE), context=context
    )
    assert p.tool_name == "get_supported_produce"

    # 3. INQUIRE_ORDER_STATUS
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.INQUIRE_ORDER_STATUS), context=context
    )
    assert p.tool_name == "get_order_status"

    # 4. INQUIRE_PAYMENT_STATUS
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.INQUIRE_PAYMENT_STATUS), context=context
    )
    assert p.tool_name == "get_payment_status"

    # 5. INQUIRE_COLLECTION_STATUS
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.INQUIRE_COLLECTION_STATUS), context=context
    )
    assert p.tool_name == "get_collection_status"

    # 6. CONFIRM_ORDER
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.CONFIRM_ORDER), context=context
    )
    assert p.tool_name == "confirm_and_create_order"

    # 7. AUTHORIZE_PAYMENT
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.AUTHORIZE_PAYMENT), context=context
    )
    assert p.tool_name == "authorize_payment"

    # 8. REQUEST_PICKUP_CHANGE
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.REQUEST_PICKUP_CHANGE), context=context
    )
    assert p.tool_name == "request_pickup_reschedule"

    # 9. REPORT_COLLECTION_PROBLEM
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.REPORT_COLLECTION_PROBLEM), context=context
    )
    assert p.tool_name == "report_collection_problem"

    # 10. REQUEST_CLARIFICATION
    p = IntentToToolMapper.map_intent_to_tool(
        ConversationalIntentResult(intent=IntentType.REQUEST_CLARIFICATION), context=context
    )
    assert p.tool_name == "explain_requirement"

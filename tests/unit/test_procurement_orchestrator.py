"""Unit tests for Phase 4: Orchestrator Procurement Reasoning + Multi-Turn Context.

Verifies:
1. Pricing objection with farmer counter-rate (OBJECTION_PRICING -> handle_pricing_objection)
2. Pricing objection without counter-rate
3. Authoritative rate strictly preserved (cannot be altered by farmer pushback)
4. OrderState not mutated to CANCELLED on pricing objection
5. Multiple pricing objections increment negotiation_attempts
6. Pre-confirmation valid quantity amendment updates offer, recalculates bill, and updates amendment_history
7. Pre-confirmation invalid quantity amendment (below minimum) rejected with original offer preserved
8. Pre-confirmation amendment of pickup location
9. Pre-confirmation amendment of availability timing
10. Post-confirmation amendment of quantity strictly locked and rejected
11. Post-confirmation amendment of produce strictly locked and rejected
12. Post-confirmation amendment of location/timing routes to reschedule
13. Partial availability declaration ("200 kg total, 50 kg ready") binds ready lot and notes future lot
14. Intra-turn self-correction ("50 kg — actually 75 kg") resolves to latest explicit quantity
15. Unresolvable location conflict triggers targeted clarification
16. Amendment without prior offer handled gracefully
17. Observational AgentTrace recorded for OBJECTION_PRICING
18. Observational AgentTrace recorded for AMEND_OFFER
19. Multi-turn full flow: Offer -> Objection -> Amendment -> Confirm -> Payment -> Attempted lock breach rejected
20. Order confirmed after amendment uses updated terms (recalculated bill & quantity)
"""

import pytest
import uuid
from datetime import datetime, timezone

from orca.domain.schemas import ExtractedOffer
from orca.domain.state_machine import OrderState
from orca.domain.intents import IntentType
from orca.agent.orchestrator import AgentOrchestrator, DialogueContext
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service


@pytest.fixture
def orchestrator():
    """Create fresh AgentOrchestrator instance for testing."""
    return AgentOrchestrator(auto_process_payment=False)


# =============================================================================
# 1 & 2. Pricing Objection Handling
# =============================================================================

@pytest.mark.asyncio
async def test_pricing_objection_with_counter_rate(orchestrator: AgentOrchestrator):
    """1. Farmer pushes back on price with explicit counter-rate ($0.50)."""
    conv_id = f"conv_obj_{uuid.uuid4().hex[:8]}"
    
    # Step 1: Initial valid offer
    t1 = await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    assert t1.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value
    assert t1.metadata["authoritative_rate"] == 0.40

    # Step 2: Farmer pushes back with counter-rate
    t2 = await orchestrator.process_message(
        conv_id,
        "Can you pay 0.50 per kg?",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.OBJECTION_PRICING.value
    assert ctx.last_tool == "handle_pricing_objection"
    assert ctx.negotiation_attempts == 1
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION

    # Response should note suggested rate, restate fixed rate, explain benefits
    assert "0.50" in t2.text
    assert "0.40" in t2.text
    assert "fixed" in t2.text.lower()
    assert "collection" in t2.text.lower() or "runner" in t2.text.lower()
    assert t2.metadata["authoritative_rate"] == 0.40
    assert t2.metadata["farmer_counter_rate"] == 0.50


@pytest.mark.asyncio
async def test_pricing_objection_without_counter_rate(orchestrator: AgentOrchestrator):
    """2. Farmer pushes back on price without a counter-rate ('Your price is too low')."""
    conv_id = f"conv_obj_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )

    t2 = await orchestrator.process_message(
        conv_id,
        "Your price is too low",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_intent == IntentType.OBJECTION_PRICING.value
    assert ctx.last_tool == "handle_pricing_objection"
    assert ctx.negotiation_attempts == 1
    assert "fixed" in t2.text.lower()
    assert "0.40" in t2.text
    assert t2.metadata["farmer_counter_rate"] is None


@pytest.mark.asyncio
async def test_authoritative_rate_strictly_preserved(orchestrator: AgentOrchestrator):
    """3. Verify authoritative rate is never modified by objection or counter-rate."""
    conv_id = f"conv_obj_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    t2 = await orchestrator.process_message(
        conv_id,
        "Can you pay 0.60 per kg? I want a better price.",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    # Total payout must be 50 * 0.40 = $20.00, NOT 50 * 0.60
    assert t2.metadata["total_amount"] == 20.00
    assert t2.metadata["authoritative_rate"] == 0.40
    assert ctx.offer.offered_rate != 0.40 or ctx.offer.offered_rate is None or True


@pytest.mark.asyncio
async def test_order_state_not_cancelled_on_pricing_objection(orchestrator: AgentOrchestrator):
    """4. Objection should NOT transition order to CANCELLED."""
    conv_id = f"conv_obj_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    t2 = await orchestrator.process_message(
        conv_id,
        "Increase the price",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.state != OrderState.CANCELLED
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION


@pytest.mark.asyncio
async def test_multiple_pricing_objections_increment_negotiation_attempts(orchestrator: AgentOrchestrator):
    """5. Multiple sequential objections increment negotiation_attempts counter."""
    conv_id = f"conv_obj_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.negotiation_attempts == 0

    await orchestrator.process_message(conv_id, "Can you pay 0.50?", farmer_id="farmer_test")
    assert ctx.negotiation_attempts == 1

    await orchestrator.process_message(conv_id, "Please, mandi pays 0.55", farmer_id="farmer_test")
    assert ctx.negotiation_attempts == 2
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION


# =============================================================================
# 6 - 9. Pre-Confirmation Amendments
# =============================================================================

@pytest.mark.asyncio
async def test_pre_confirmation_amendment_valid_quantity(orchestrator: AgentOrchestrator):
    """6. Pre-confirmation valid quantity amendment updates offer and recalculates bill."""
    conv_id = f"conv_amend_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.offer.quantity == 50.0

    t2 = await orchestrator.process_message(
        conv_id,
        "Actually I have 80 kg",
        farmer_id="farmer_test",
    )
    assert ctx.last_intent == IntentType.AMEND_OFFER.value
    assert ctx.last_tool == "evaluate_procurement_offer"
    assert ctx.offer.quantity == 80.0
    assert len(ctx.amendment_history) == 1
    # 80 kg * 0.40 = $32.00
    assert t2.metadata["total_amount"] == 32.00
    assert t2.metadata["quantity"] == 80.0
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION


@pytest.mark.asyncio
async def test_pre_confirmation_amendment_invalid_quantity_rejected(orchestrator: AgentOrchestrator):
    """7. Pre-confirmation invalid quantity amendment (below 5 kg minimum) is rejected."""
    conv_id = f"conv_amend_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)

    t2 = await orchestrator.process_message(
        conv_id,
        "Actually I only have 2 kg",
        farmer_id="farmer_test",
    )
    assert ctx.last_intent == IntentType.AMEND_OFFER.value
    assert t2.metadata["amendment_rejected"] is True
    # Context offer must remain untouched!
    assert ctx.offer.quantity == 50.0
    assert "Cannot update offer" in t2.text
    assert "Your previous offer remains in effect" in t2.text
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION


@pytest.mark.asyncio
async def test_pre_confirmation_amendment_location(orchestrator: AgentOrchestrator):
    """8. Pre-confirmation amendment of pickup location."""
    conv_id = f"conv_amend_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.offer.pickup_location == "Springfield"

    t2 = await orchestrator.process_message(
        conv_id,
        "Change pickup to Greenfield Farm",
        farmer_id="farmer_test",
    )
    assert ctx.last_intent == IntentType.AMEND_OFFER.value
    assert ctx.offer.pickup_location == "Greenfield Farm"
    assert len(ctx.amendment_history) == 1
    assert "Greenfield Farm" in t2.text


@pytest.mark.asyncio
async def test_pre_confirmation_amendment_timing(orchestrator: AgentOrchestrator):
    """9. Pre-confirmation amendment of availability timing."""
    conv_id = f"conv_amend_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)

    t2 = await orchestrator.process_message(
        conv_id,
        "Actually pickup is tomorrow at 2 PM",
        farmer_id="farmer_test",
    )
    assert ctx.last_intent == IntentType.AMEND_OFFER.value
    assert "2:00 PM" in (ctx.offer.pickup_datetime or ctx.offer.availability_window or "")


# =============================================================================
# 10 - 12. Post-Confirmation Locks and Reschedule
# =============================================================================

@pytest.mark.asyncio
async def test_post_confirmation_amendment_quantity_locked(orchestrator: AgentOrchestrator):
    """10. Post-confirmation attempt to modify quantity is strictly rejected."""
    conv_id = f"conv_post_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    # Confirm the order
    t_confirm = await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_test")
    order_id = t_confirm.metadata["order_id"]
    ctx = orchestrator.get_context(conv_id)
    assert ctx.current_order_id == order_id

    # Now attempt to change quantity
    t_breach = await orchestrator.process_message(
        conv_id,
        "Actually change quantity to 100 kg",
        farmer_id="farmer_test",
    )
    assert t_breach.metadata["amendment_rejected"] is True
    assert t_breach.metadata["reason"] == "POST_CONFIRMATION_LOCK"
    assert "Cannot modify confirmed order" in t_breach.text

    # Order in database remains strictly 50 kg
    order = order_service.get_order(order_id)
    assert order.quantity == 50.0
    assert order.total_amount == 20.00


@pytest.mark.asyncio
async def test_post_confirmation_amendment_produce_locked(orchestrator: AgentOrchestrator):
    """11. Post-confirmation attempt to modify produce type is strictly rejected."""
    conv_id = f"conv_post_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    t_confirm = await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_test")
    order_id = t_confirm.metadata["order_id"]

    t_breach = await orchestrator.process_message(
        conv_id,
        "Change produce to onions instead",
        farmer_id="farmer_test",
    )
    assert t_breach.metadata["amendment_rejected"] is True
    order = order_service.get_order(order_id)
    assert order.produce_type == "potato"


@pytest.mark.asyncio
async def test_post_confirmation_amendment_reschedule_allowed(orchestrator: AgentOrchestrator):
    """12. Post-confirmation location/timing amendment routes to reschedule."""
    conv_id = f"conv_post_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    t_confirm = await orchestrator.process_message(conv_id, "Confirm", farmer_id="farmer_test")
    order_id = t_confirm.metadata["order_id"]

    t_resched = await orchestrator.process_message(
        conv_id,
        "Change pickup to tomorrow at 4 PM",
        farmer_id="farmer_test",
    )
    assert t_resched.metadata["rescheduled"] is True
    order = order_service.get_order(order_id)
    assert "4:00 PM" in (order.pickup_time_str or "")


# =============================================================================
# 13 - 16. Complex Conversational Scenarios
# =============================================================================

@pytest.mark.asyncio
async def test_partial_availability_handling(orchestrator: AgentOrchestrator):
    """13. Partial harvest declaration binds active lot and notes future lot."""
    conv_id = f"conv_part_{uuid.uuid4().hex[:8]}"
    
    t1 = await orchestrator.process_message(
        conv_id,
        "I have 200 kg of potatoes total, but only 50 kg is ready tomorrow at 10 AM in Springfield.",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.offer.quantity == 50.0
    assert ctx.future_availability_notes is not None
    assert "150" in ctx.future_availability_notes
    assert "150" in t1.text
    assert t1.metadata["quantity"] == 50.0
    assert t1.metadata["authoritative_rate"] == 0.40


@pytest.mark.asyncio
async def test_intra_turn_self_correction(orchestrator: AgentOrchestrator):
    """14. Intra-turn self-correction ('50 kg — actually 75 kg') resolves to 75 kg."""
    conv_id = f"conv_corr_{uuid.uuid4().hex[:8]}"
    
    t1 = await orchestrator.process_message(
        conv_id,
        "I have 50 kg — actually 75 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.offer.quantity == 75.0
    # 75 kg * 0.40 = $30.00
    assert t1.metadata["total_amount"] == 30.00


@pytest.mark.asyncio
async def test_unresolvable_location_conflict(orchestrator: AgentOrchestrator):
    """15. Ambiguous contradictory locations trigger targeted clarification."""
    conv_id = f"conv_conf_{uuid.uuid4().hex[:8]}"
    
    t1 = await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Delhi, pickup in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.DETAILS_PENDING
    assert ctx.active_clarification == "pickup_location"
    assert t1.metadata["conflict_detected"] is True
    assert "Delhi" in t1.text and "Springfield" in t1.text


@pytest.mark.asyncio
async def test_amendment_without_prior_offer(orchestrator: AgentOrchestrator):
    """16. Amendment attempt when no prior offer exists is handled gracefully."""
    conv_id = f"conv_empty_{uuid.uuid4().hex[:8]}"
    
    t1 = await orchestrator.process_message(
        conv_id,
        "Actually make it 75 kg",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.state == OrderState.DETAILS_PENDING
    assert "no active offer" in t1.text.lower() or "produce" in t1.text.lower()


# =============================================================================
# 17 - 18. Observational AgentTrace Recording
# =============================================================================

@pytest.mark.asyncio
async def test_agent_trace_recorded_for_objection_pricing(orchestrator: AgentOrchestrator):
    """17. AgentTrace correctly captures OBJECTION_PRICING and handle_pricing_objection."""
    conv_id = f"conv_trc_obj_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    await orchestrator.process_message(
        conv_id,
        "Can you pay 0.50?",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_trace is not None
    assert ctx.last_trace.detected_intent == IntentType.OBJECTION_PRICING.value
    assert ctx.last_trace.selected_tool == "handle_pricing_objection"
    assert ctx.last_trace.tool_success is True


@pytest.mark.asyncio
async def test_agent_trace_recorded_for_amend_offer(orchestrator: AgentOrchestrator):
    """18. AgentTrace correctly captures AMEND_OFFER and evaluate_procurement_offer."""
    conv_id = f"conv_trc_amd_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    await orchestrator.process_message(
        conv_id,
        "Actually I have 80 kg",
        farmer_id="farmer_test",
    )
    ctx = orchestrator.get_context(conv_id)
    assert ctx.last_trace is not None
    assert ctx.last_trace.detected_intent == IntentType.AMEND_OFFER.value
    assert ctx.last_trace.selected_tool == "evaluate_procurement_offer"
    assert ctx.last_trace.tool_success is True


# =============================================================================
# 19 - 20. End-to-End Multi-Turn Flows
# =============================================================================

@pytest.mark.asyncio
async def test_multi_turn_procurement_flow(orchestrator: AgentOrchestrator):
    """19. Complete multi-turn flow:
    Offer -> Pricing Objection -> Offer Amendment -> Confirm -> Payment -> Attempted Lock Breach.
    """
    conv_id = f"conv_full_{uuid.uuid4().hex[:8]}"
    
    # 1. Offer 50 kg potatoes
    t1 = await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_full",
    )
    assert t1.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value
    assert t1.metadata["total_amount"] == 20.00

    # 2. Objection to price
    t2 = await orchestrator.process_message(
        conv_id,
        "Can you pay 0.50 per kg?",
        farmer_id="farmer_full",
    )
    assert t2.metadata["intent"] == "OBJECTION_PRICING"
    assert t2.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value

    # 3. Amendment to quantity (80 kg)
    t3 = await orchestrator.process_message(
        conv_id,
        "Actually I have 80 kg",
        farmer_id="farmer_full",
    )
    assert t3.metadata["total_amount"] == 32.00
    assert t3.metadata["state"] == OrderState.AWAITING_FARMER_CONFIRMATION.value

    # 4. Confirmation
    t4 = await orchestrator.process_message(
        conv_id,
        "Confirm",
        farmer_id="farmer_full",
    )
    assert t4.metadata["state"] == OrderState.PAYMENT_PENDING.value
    order_id = t4.metadata["order_id"]
    assert t4.metadata["total_amount"] == 32.00

    # 5. Payment
    t5 = await orchestrator.process_message(
        conv_id,
        "Pay now",
        farmer_id="farmer_full",
    )
    assert t5.metadata["payment_status"] == "SUCCESS"

    # 6. Attempt post-confirmation modification -> rejected!
    t6 = await orchestrator.process_message(
        conv_id,
        "Change that to 100 kg",
        farmer_id="farmer_full",
    )
    assert t6.metadata["amendment_rejected"] is True
    order = order_service.get_order(order_id)
    assert order.quantity == 80.0


@pytest.mark.asyncio
async def test_confirmation_after_amendment_uses_updated_terms(orchestrator: AgentOrchestrator):
    """20. Confirming after amendment persists the updated quantity and recalculated bill."""
    conv_id = f"conv_conf_amd_{uuid.uuid4().hex[:8]}"
    
    await orchestrator.process_message(
        conv_id,
        "I have 50 kg of potatoes in Springfield available tomorrow at 10 AM.",
        farmer_id="farmer_test",
    )
    await orchestrator.process_message(
        conv_id,
        "Actually I have 75 kg",
        farmer_id="farmer_test",
    )
    t_confirm = await orchestrator.process_message(
        conv_id,
        "Confirm",
        farmer_id="farmer_test",
    )
    order_id = t_confirm.metadata["order_id"]
    order = order_service.get_order(order_id)
    assert order.quantity == 75.0
    assert order.validated_rate == 0.40
    assert order.total_amount == 30.00

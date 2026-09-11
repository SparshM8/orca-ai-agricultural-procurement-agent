"""Unit tests for the conversational extraction and dialogue state orchestrator.

Covers the 6 required scenarios:
1. Complete offer extraction
2. Missing quantity
3. Missing pickup location
4. Multi-turn conversation completion
5. Farmer-provided price vs authoritative rate
6. Invalid / ambiguous input
"""

import pytest
from orca.agent.orchestrator import AgentOrchestrator, RuleBasedPatternExtractor, parse_pickup_timing
from orca.domain.state_machine import OrderState
from orca.services.order import order_service
from orca.services.collection import collection_service


@pytest.fixture
def orchestrator():
    """Create fresh orchestrator instance for testing."""
    return AgentOrchestrator()


@pytest.mark.asyncio
async def test_complete_offer_extraction(orchestrator: AgentOrchestrator):
    """Scenario 1: Complete offer extraction in a single turn."""
    conv_id = "test_conv_complete_001"
    message = "I have 50 kg of potatoes in Springfield available this Friday at 10 AM."

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=message,
        farmer_id="farmer_complete",
    )

    context = orchestrator._conversations[conv_id]
    assert context.offer.produce_type == "potato"
    assert context.offer.quantity == 50.0
    assert context.offer.unit == "kg"
    assert context.offer.pickup_location == "Springfield"
    assert context.offer.availability_window is not None
    assert context.state == OrderState.AWAITING_FARMER_CONFIRMATION

    # Authoritative rate for potato is 0.40 USD/kg -> total = 20.00 USD
    assert outbound.metadata["authoritative_rate"] == 0.40
    assert outbound.metadata["total_amount"] == 20.00
    assert outbound.metadata["currency"] == "USD"
    assert "50 kg of potato" in outbound.text
    assert "Total Payout: USD 20.00" in outbound.text
    assert "Confirm" in outbound.text


@pytest.mark.asyncio
async def test_missing_quantity(orchestrator: AgentOrchestrator):
    """Scenario 2: Farmer offers produce and location but misses quantity."""
    conv_id = "test_conv_missing_qty"
    message = "I have potatoes in Springfield ready this week."

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=message,
        farmer_id="farmer_qty",
    )

    context = orchestrator._conversations[conv_id]
    assert context.offer.produce_type == "potato"
    assert context.offer.quantity is None
    assert context.offer.pickup_location == "Springfield"
    assert context.state == OrderState.DETAILS_PENDING

    assert "missing_fields" in outbound.metadata
    assert "quantity" in outbound.metadata["missing_fields"]
    assert "How much potato" in outbound.text or "quantity" in outbound.text.lower()


@pytest.mark.asyncio
async def test_missing_pickup_location(orchestrator: AgentOrchestrator):
    """Scenario 3: Farmer provides produce, quantity, and date but no pickup location."""
    conv_id = "test_conv_missing_loc"
    message = "I have 20 kg of tomatoes ready tomorrow."

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=message,
        farmer_id="farmer_loc",
    )

    context = orchestrator._conversations[conv_id]
    assert context.offer.produce_type == "tomato"
    assert context.offer.quantity == 20.0
    assert context.offer.unit == "kg"
    assert context.offer.pickup_location is None
    assert context.state == OrderState.DETAILS_PENDING

    assert "pickup_location" in outbound.metadata["missing_fields"]
    assert "Where is the 20 kg of tomato located" in outbound.text or "pickup location" in outbound.text.lower()


@pytest.mark.asyncio
async def test_multiturn_conversation_completion(orchestrator: AgentOrchestrator):
    """Scenario 4: Context is preserved across multiple turns until information is complete."""
    conv_id = "test_conv_multiturn"

    # Turn 1: Farmer mentions produce and quantity only
    t1 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="I have 10 kg of potatoes available.",
        farmer_id="farmer_multi",
    )
    ctx = orchestrator._conversations[conv_id]
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 10.0
    assert ctx.offer.unit == "kg"
    assert ctx.offer.pickup_location is None
    assert ctx.state == OrderState.DETAILS_PENDING
    assert "pickup_location" in t1.metadata["missing_fields"]

    # Turn 2: Farmer supplies pickup location
    t2 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="The pickup location is Springfield.",
        farmer_id="farmer_multi",
    )
    # Context must have preserved potato and 10 kg
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 10.0
    assert ctx.offer.pickup_location == "Springfield"
    assert ctx.state == OrderState.DETAILS_PENDING
    assert "availability" in t2.metadata["missing_fields"]

    # Turn 3: Farmer supplies availability date
    t3 = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Available tomorrow at 2 PM.",
        farmer_id="farmer_multi",
    )
    # Now all required info is complete!
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION
    assert t3.metadata["state"] == "AWAITING_FARMER_CONFIRMATION"
    assert t3.metadata["total_amount"] == 4.00  # 10 kg * $0.40
    assert "Total Payout: USD 4.00" in t3.text
    assert "Springfield" in t3.text


@pytest.mark.asyncio
async def test_farmer_provided_price_vs_authoritative_rate(orchestrator: AgentOrchestrator):
    """Scenario 5: Farmer proposes a price, but backend enforces authoritative rate."""
    conv_id = "test_conv_farmer_price"
    # Farmer requests $0.50 per kg, but authoritative rate for onion is $0.35/kg
    message = "I have 10 kg of onions in Springfield ready today. I want $0.50 per kg."

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=message,
        farmer_id="farmer_price_negotiator",
    )

    ctx = orchestrator._conversations[conv_id]
    assert ctx.offer.produce_type == "onion"
    assert ctx.offer.quantity == 10.0
    assert ctx.offer.offered_rate == 0.50

    # Backend authoritative rate enforced
    assert outbound.metadata["authoritative_rate"] == 0.35
    # Total payout is 10 * 0.35 = 3.50 (NOT 10 * 0.50 = 5.00)
    assert outbound.metadata["total_amount"] == 3.50
    assert "Total Payout: USD 3.50" in outbound.text

    # Response clearly mentions the authoritative rate and addresses farmer's requested price
    assert "You suggested a price of USD 0.50" in outbound.text
    assert "authoritative fixed procurement rate is USD 0.35" in outbound.text


@pytest.mark.asyncio
async def test_invalid_and_ambiguous_inputs(orchestrator: AgentOrchestrator):
    """Scenario 6: Handling ambiguous inputs, greetings, and unsupported crops."""
    # 6a. Pure greeting
    out1 = await orchestrator.process_message("conv_greet", "Hello, good morning!")
    assert "I am ORCA" in out1.text
    assert "onion, potato, tomato" in out1.text

    # 6b. Unsupported produce
    out2 = await orchestrator.process_message(
        "conv_unsupported",
        "I have 50 kg of pineapples in Springfield ready tomorrow."
    )
    assert "only procure: onion, potato, tomato" in out2.text
    assert "do not currently purchase 'pineapple'" in out2.text

    # 6c. Ambiguous quantity without unit
    out3 = await orchestrator.process_message(
        "conv_ambiguous_unit",
        "I have 10 potatoes in Springfield ready tomorrow."
    )
    ctx3 = orchestrator._conversations["conv_ambiguous_unit"]
    assert ctx3.offer.produce_type == "potato"
    assert ctx3.offer.quantity == 10.0
    assert ctx3.offer.unit is None
    assert "unit" in out3.metadata["missing_fields"]
    assert "specify the measurement unit" in out3.text


@pytest.mark.asyncio
async def test_pickup_timing_tomorrow_at_10_am(orchestrator: AgentOrchestrator):
    """Regression test: Farmer offers produce with 'tomorrow at 10 AM'.
    
    Verifies:
    1. '10 AM' is extracted and normalized to 'tomorrow at 10:00 AM'.
    2. Stored in dialogue context/schema.
    3. Included in transaction summary ('Pickup Availability: tomorrow at 10:00 AM').
    4. Persisted to order pickup datetime/time fields.
    5. Passed into collection task ('Scheduled Window: tomorrow at 10:00 AM').
    """
    conv_id = "test_conv_time_tomorrow_10am"
    message = "I have 50 kg of potatoes in Nairobi available tomorrow at 10 AM."

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=message,
        farmer_id="farmer_time_01",
    )

    ctx = orchestrator._conversations[conv_id]
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 50.0
    assert ctx.offer.unit == "kg"
    assert ctx.offer.pickup_location == "Nairobi"
    assert ctx.offer.availability_window == "tomorrow at 10:00 AM"
    assert ctx.offer.pickup_datetime == "tomorrow at 10:00 AM"
    assert "- Pickup Availability: tomorrow at 10:00 AM" in outbound.text

    # Confirm order
    confirm_outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Confirm",
        farmer_id="farmer_time_01",
    )
    order_id = ctx.current_order_id
    assert order_id is not None
    order = order_service.get_order(order_id)
    assert order is not None
    assert order.pickup_time_str == "tomorrow at 10:00 AM"
    assert order.pickup_datetime is not None
    assert "- Scheduled Window: tomorrow at 10:00 AM" in confirm_outbound.text

    # Pay order -> triggers collection task
    pay_outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Pay",
        farmer_id="farmer_time_01",
    )
    task = collection_service.get_task_by_order(order_id)
    assert task is not None
    assert task.scheduled_time_str == "tomorrow at 10:00 AM"
    assert task.scheduled_datetime is not None
    assert "- Scheduled Window: tomorrow at 10:00 AM" in pay_outbound.text


@pytest.mark.asyncio
async def test_pickup_timing_friday_at_2_pm(orchestrator: AgentOrchestrator):
    """Regression test: Farmer offers produce with 'Friday at 2 PM'.
    
    Verifies:
    1. Capitalized and normalized to 'Friday at 2:00 PM'.
    2. Stored in dialogue context/schema.
    3. Included in transaction summary ('Pickup Availability: Friday at 2:00 PM').
    4. Persisted to order and passed into collection task.
    """
    conv_id = "test_conv_time_friday_2pm"
    message = "I have 50 kg of potatoes in Nairobi available Friday at 2 PM."

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=message,
        farmer_id="farmer_time_02",
    )

    ctx = orchestrator._conversations[conv_id]
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.pickup_location == "Nairobi"
    assert ctx.offer.availability_window == "Friday at 2:00 PM"
    assert ctx.offer.pickup_datetime == "Friday at 2:00 PM"
    assert "- Pickup Availability: Friday at 2:00 PM" in outbound.text

    # Confirm order
    await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Confirm",
        farmer_id="farmer_time_02",
    )
    order = order_service.get_order(ctx.current_order_id)
    assert order.pickup_time_str == "Friday at 2:00 PM"
    assert order.pickup_datetime is not None

    # Pay order -> collection task
    await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Pay",
        farmer_id="farmer_time_02",
    )
    task = collection_service.get_task_by_order(order.id)
    assert task.scheduled_time_str == "Friday at 2:00 PM"
    assert task.scheduled_datetime is not None


@pytest.mark.asyncio
async def test_pickup_timing_tomorrow_without_time(orchestrator: AgentOrchestrator):
    """Regression test: Date-only input 'tomorrow' without time.
    
    Verifies that date-only inputs continue to work correctly without errors,
    preserving 'tomorrow' as the availability window.
    """
    conv_id = "test_conv_time_tomorrow_date_only"
    message = "I have 50 kg of potatoes in Nairobi available tomorrow."

    outbound = await orchestrator.process_message(
        conversation_id=conv_id,
        message_text=message,
        farmer_id="farmer_time_03",
    )

    ctx = orchestrator._conversations[conv_id]
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.pickup_location == "Nairobi"
    assert ctx.offer.availability_window == "tomorrow"
    assert ctx.offer.pickup_datetime is None
    assert "- Pickup Availability: tomorrow" in outbound.text

    # Confirm order
    await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Confirm",
        farmer_id="farmer_time_03",
    )
    order = order_service.get_order(ctx.current_order_id)
    assert order.pickup_time_str == "tomorrow"
    assert order.pickup_datetime is not None  # Target datetime calculated (default 9am)

    # Pay order -> collection task
    await orchestrator.process_message(
        conversation_id=conv_id,
        message_text="Pay",
        farmer_id="farmer_time_03",
    )
    task = collection_service.get_task_by_order(order.id)
    assert task.scheduled_time_str == "tomorrow"
    assert task.scheduled_datetime is not None


def test_rule_based_extractor_pickup_timing_variations():
    """Unit test parser timing variations directly."""
    dt1, str1 = parse_pickup_timing("tomorrow at 10 AM")
    assert str1 == "tomorrow at 10:00 AM"
    assert dt1 is not None

    dt2, str2 = parse_pickup_timing("Friday at 2 PM")
    assert str2 == "Friday at 2:00 PM"
    assert dt2 is not None

    dt3, str3 = parse_pickup_timing("tomorrow")
    assert str3 == "tomorrow"
    assert dt3 is not None

    dt4, str4 = parse_pickup_timing("this Friday at 2 PM")
    assert str4 == "this Friday at 2:00 PM"
    assert dt4 is not None


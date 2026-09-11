"""Tests for Procurement Defect Hardening.

Covers:
1. unsupported commodity amendment
2. arbitrary unsupported commodity extraction in amendment clause
3. supported commodity amendment
4. pickup location amendment remains location
5. no commodity/location cross-contamination
6. fresh-session self-correction becomes OFFER_PRODUCE
7. active-offer self-correction becomes AMEND_OFFER
8. latest explicit quantity wins
9. invalid amendment preserves original offer
10. no state mutation on failed amendment
11. live-style integration test for both original failures
"""

import pytest
from orca.agent.extractors.rule_based import RuleBasedPatternExtractor
from orca.agent.intent.rule_based_classifier import RuleBasedIntentClassifier
from orca.agent.orchestrator import AgentOrchestrator, DialogueContext
from orca.domain.intents import IntentType
from orca.domain.schemas import ExtractedOffer, InboundMessage
from orca.domain.state_machine import OrderState
from orca.services.procurement import procurement_service


@pytest.fixture
def extractor():
    return RuleBasedPatternExtractor()


@pytest.fixture
def classifier(extractor):
    return RuleBasedIntentClassifier(offer_extractor=extractor)


@pytest.fixture
def orchestrator_instance(classifier, extractor):
    return AgentOrchestrator(intent_classifier=classifier, extractor=extractor)


# =============================================================================
# Test 1 & 2: Unsupported Commodity Extraction in Amendment Clauses
# =============================================================================
@pytest.mark.asyncio
async def test_unsupported_commodity_amendment_extraction(extractor):
    """Verify that 'papaya' is extracted as produce_type, not pickup_location."""
    offer = await extractor.extract("Actually, change the potatoes to papaya.")
    assert offer.produce_type == "papaya"
    assert offer.pickup_location is None or "papaya" not in offer.pickup_location.lower()


@pytest.mark.asyncio
async def test_arbitrary_unsupported_commodity_extraction(extractor):
    """Verify arbitrary commodities in substitution clauses are extracted as produce."""
    # Test arbitrary unsupported crops
    offer1 = await extractor.extract("switch onions to mangoes")
    assert offer1.produce_type == "mango"
    assert offer1.pickup_location is None or "mango" not in offer1.pickup_location.lower()

    offer2 = await extractor.extract("change potatoes to guava")
    assert offer2.produce_type == "guava"
    assert offer2.pickup_location is None or "guava" not in offer2.pickup_location.lower()

    offer3 = await extractor.extract("make this dragonfruit instead of potatoes")
    assert offer3.produce_type == "dragonfruit"
    assert offer3.pickup_location is None or "dragonfruit" not in offer3.pickup_location.lower()


# =============================================================================
# Test 3: Supported Commodity Amendment
# =============================================================================
@pytest.mark.asyncio
async def test_supported_commodity_amendment(extractor):
    """Verify supported commodity substitution extracts the new supported crop."""
    offer = await extractor.extract("change potato to tomato")
    assert offer.produce_type == "tomato"
    assert offer.pickup_location is None or "tomato" not in offer.pickup_location.lower()


# =============================================================================
# Test 4 & 5: Location Amendment & Cross-Contamination Prevention
# =============================================================================
@pytest.mark.asyncio
async def test_pickup_location_amendment_remains_location(extractor):
    """Verify explicit location amendments extract location without altering commodity."""
    offer1 = await extractor.extract("change pickup to Springfield")
    assert offer1.pickup_location == "Springfield"
    assert offer1.produce_type is None

    offer2 = await extractor.extract("move collection to Farm 2")
    assert offer2.pickup_location == "Farm 2"
    assert offer2.produce_type is None


@pytest.mark.asyncio
async def test_no_commodity_location_cross_contamination(extractor, classifier):
    """Verify produce words never become pickup_location and location words never become produce."""
    # "Actually, change the potatoes to papaya."
    ctx = DialogueContext(conversation_id="c1", farmer_id="f1")
    ctx.offer = ExtractedOffer(produce_type="potato", quantity=50.0, unit="kg", pickup_location="Springfield")
    ctx.state = OrderState.AWAITING_FARMER_CONFIRMATION

    res = await classifier.classify("Actually, change the potatoes to papaya.", context=ctx)
    assert res.intent == IntentType.AMEND_OFFER
    assert res.entities.offer.produce_type == "papaya"
    assert res.entities.new_pickup_location is None or "papaya" not in res.entities.new_pickup_location.lower()

    # "Actually, make that 80 kg."
    res2 = await classifier.classify("Actually, make that 80 kg.", context=ctx)
    assert res2.intent == IntentType.AMEND_OFFER
    assert res2.entities.offer.quantity == 80.0
    assert res2.entities.new_pickup_location is None or "80" not in res2.entities.new_pickup_location


# =============================================================================
# Test 6: Fresh-Session Self-Correction Becomes OFFER_PRODUCE
# =============================================================================
@pytest.mark.asyncio
async def test_fresh_session_self_correction_becomes_offer_produce(classifier):
    """On a fresh session with no active offer, self-correction must be OFFER_PRODUCE."""
    ctx = DialogueContext(conversation_id="c_fresh", farmer_id="f_fresh")
    # Fresh session: context.offer is None
    res = await classifier.classify("I have 50 kg of potatoes, actually 75 kg.", context=ctx)
    assert res.intent == IntentType.OFFER_PRODUCE
    assert res.entities.offer is not None
    assert res.entities.offer.quantity == 75.0
    assert res.entities.offer.produce_type == "potato"


# =============================================================================
# Test 7: Active-Offer Self-Correction Becomes AMEND_OFFER
# =============================================================================
@pytest.mark.asyncio
async def test_active_offer_modification_becomes_amend_offer(classifier):
    """On an active pre-confirmation offer, modification must be AMEND_OFFER."""
    ctx = DialogueContext(conversation_id="c_active", farmer_id="f_active")
    ctx.offer = ExtractedOffer(
        produce_type="potato", quantity=50.0, unit="kg", pickup_location="Springfield", availability_window="tomorrow"
    )
    ctx.state = OrderState.AWAITING_FARMER_CONFIRMATION

    res = await classifier.classify("Actually, make that 75 kg.", context=ctx)
    assert res.intent == IntentType.AMEND_OFFER
    assert res.entities.offer.quantity == 75.0


# =============================================================================
# Test 8: Latest Explicit Quantity Wins
# =============================================================================
@pytest.mark.asyncio
async def test_latest_explicit_quantity_wins(extractor):
    """Verify various forms of self-correction always select the latest explicit quantity."""
    # Form A: "50 kg of potatoes, actually 75 kg."
    o1 = await extractor.extract("I have 50 kg of potatoes, actually 75 kg.")
    assert o1.quantity == 75.0
    assert o1.unit == "kg"
    assert o1.produce_type == "potato"

    # Form B: "50 kilos, actually make that 75"
    o2 = await extractor.extract("I've got 50 kilos, actually make that 75.")
    assert o2.quantity == 75.0
    assert o2.unit == "kg"

    # Form C: "50 kg — actually 75 kg"
    o3 = await extractor.extract("50 kg — actually 75 kg.")
    assert o3.quantity == 75.0
    assert o3.unit == "kg"

    # Form D: "Actually, it's 75 kg, not 50 kg."
    o4 = await extractor.extract("Actually, it's 75 kg, not 50 kg.")
    assert o4.quantity == 75.0
    assert o4.unit == "kg"


# =============================================================================
# Test 9 & 10: Invalid Amendment Preserves Original Offer & No State Mutation
# =============================================================================
@pytest.mark.asyncio
async def test_unsupported_commodity_preserves_original_offer_and_state(orchestrator_instance):
    """When farmer proposes 'papaya', the amendment must fail and potato offer remain untouched."""
    farmer_id = "test_papaya_preserve"
    conv_id = f"conv_{farmer_id}"
    # Turn 1: Establish valid offer
    t1 = await orchestrator_instance.process_message(
        conversation_id=conv_id, farmer_id=farmer_id, message_text="I have 50 kg of potatoes in Springfield available tomorrow at 10 AM."
    )
    assert t1.metadata["state"] == "AWAITING_FARMER_CONFIRMATION"
    assert t1.metadata["produce"] == "potato"
    assert t1.metadata["total_amount"] == 20.0

    # Turn 2: Attempt unsupported commodity amendment
    t2 = await orchestrator_instance.process_message(
        conversation_id=conv_id, farmer_id=farmer_id, message_text="Actually, change the potatoes to papaya."
    )
    # Must reject amendment
    assert t2.metadata.get("amendment_rejected") is True
    assert "UNSUPPORTED_PRODUCE:papaya" in t2.text or "not procured" in t2.text.lower()
    assert "Your previous offer remains in effect" in t2.text
    assert "50 kg of potato" in t2.text

    # Verify context offer is intact
    ctx = orchestrator_instance.get_context(conv_id)
    assert ctx.offer.produce_type == "potato"
    assert ctx.offer.quantity == 50.0
    assert ctx.offer.pickup_location == "Springfield"
    assert ctx.state == OrderState.AWAITING_FARMER_CONFIRMATION


# =============================================================================
# Test 11: Live-style Integration Test for Both Original Failures
# =============================================================================
@pytest.mark.asyncio
async def test_live_style_original_failures_resolved(orchestrator_instance):
    """End-to-end multi-turn test checking both Scenario 5 and Scenario 7 behaviors."""
    # Part 1: Fresh-session self-correction (Scenario 7)
    s7_farmer = "s7_farmer"
    msg_s7 = await orchestrator_instance.process_message(
        conversation_id=f"conv_{s7_farmer}", farmer_id=s7_farmer, message_text="I have 50 kg of potatoes in Springfield available tomorrow at 10 AM, actually 75 kg."
    )
    # Must be accepted at 75 kg
    assert msg_s7.metadata["state"] == "AWAITING_FARMER_CONFIRMATION"
    assert msg_s7.metadata["produce"] == "potato"
    assert msg_s7.metadata["quantity"] == 75.0
    assert msg_s7.metadata["total_amount"] == 30.0  # 75 * 0.40
    assert "75 kg of potato" in msg_s7.text
    assert "30.00" in msg_s7.text

    # Part 2: Unsupported amendment (Scenario 5)
    s5_farmer = "s5_farmer"
    conv_s5 = f"conv_{s5_farmer}"
    # Establish valid potato offer
    await orchestrator_instance.process_message(
        conversation_id=conv_s5, farmer_id=s5_farmer, message_text="I have 80 kg of potatoes in Springfield available tomorrow at 10 AM."
    )
    # Attempt papaya amendment
    msg_s5 = await orchestrator_instance.process_message(
        conversation_id=conv_s5, farmer_id=s5_farmer, message_text="Actually, change the potatoes to papaya."
    )
    assert msg_s5.metadata.get("amendment_rejected") is True
    assert "80 kg of potato" in msg_s5.text
    assert "papaya" not in msg_s5.metadata.get("pickup_location", "")

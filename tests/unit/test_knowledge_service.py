"""Unit tests for BusinessKnowledgeService and Grounded Knowledge Contracts.

Validates:
1. Grounded lookups across all 7 knowledge categories
2. Category filtering
3. Case-insensitivity
4. Rejection of out-of-scope / off-topic queries
5. Rejection of weak / ambiguous queries
6. Zero hardcoded authoritative prices
7. Zero mutation of orders, payments, collection tasks, and OrderState
8. Complete absence of Cora / Springhill artifacts
9. Proper serialization of KnowledgeResult
"""

import pytest
import re
from orca.domain.knowledge import (
    KnowledgeCategory,
    KnowledgeArticle,
    KnowledgeResult,
)
from orca.domain.state_machine import OrderState
from orca.services.knowledge import BusinessKnowledgeService, knowledge_service
from orca.services.order import order_service
from orca.services.payment import payment_service
from orca.services.collection import collection_service


# -----------------------------------------------------------------------------
# 1. Company Information Lookup
# -----------------------------------------------------------------------------
def test_company_information_lookup():
    queries = [
        "What is ORCA?",
        "Tell me about the ORCA procurement platform",
        "What does ORCA do?",
        "Who are you and how do you work?",
    ]
    for q in queries:
        result = knowledge_service.lookup(q)
        assert result.matched is True, f"Failed for query: {q}"
        assert result.answerable is True
        assert result.category == KnowledgeCategory.COMPANY_INFO
        assert result.source_article_id == "kb_orca_overview"
        assert result.clarification_required is False
        assert result.confidence >= 0.50
        assert "direct digital agricultural procurement platform" in result.answer.lower()


# -----------------------------------------------------------------------------
# 2. Procurement Process Lookups
# -----------------------------------------------------------------------------
def test_procurement_process_start_offer():
    queries = [
        "How do I start an offer to sell produce?",
        "How to sell produce through ORCA?",
        "How do I submit an offer for my harvest?",
    ]
    for q in queries:
        result = knowledge_service.lookup(q)
        assert result.matched is True
        assert result.answerable is True
        assert result.category == KnowledgeCategory.PROCUREMENT_PROCESS
        assert result.source_article_id == "kb_start_offer"
        assert "produce type" in result.answer.lower()
        assert "availability window" in result.answer.lower()


def test_procurement_process_confirmation():
    result = knowledge_service.lookup("How does order confirmation work?")
    assert result.matched is True
    assert result.answerable is True
    assert result.category == KnowledgeCategory.PROCUREMENT_PROCESS
    assert result.source_article_id == "kb_confirmation_flow"
    assert "explicit" in result.answer.lower()


def test_procurement_process_amendment():
    result = knowledge_service.lookup("Can I amend or change my offer before confirming?")
    assert result.matched is True
    assert result.answerable is True
    assert result.category == KnowledgeCategory.PROCUREMENT_PROCESS
    assert result.source_article_id == "kb_offer_amendment"
    assert "freely amend" in result.answer.lower()


def test_procurement_process_post_confirmation():
    result = knowledge_service.lookup("What happens after I confirm the order?")
    assert result.matched is True
    assert result.answerable is True
    assert result.category == KnowledgeCategory.PROCUREMENT_PROCESS
    assert result.source_article_id == "kb_post_confirmation"
    assert "order id" in result.answer.lower()


def test_procurement_process_order_status():
    result = knowledge_service.lookup("How do I check the status of my order?")
    assert result.matched is True
    assert result.answerable is True
    assert result.category == KnowledgeCategory.PROCUREMENT_PROCESS
    assert result.source_article_id == "kb_checking_order_status"
    assert "order id" in result.answer.lower()


# -----------------------------------------------------------------------------
# 3. Pricing Policy Lookups
# -----------------------------------------------------------------------------
def test_pricing_policy_authoritative_rates():
    queries = [
        "Why is the rate fixed and can I negotiate?",
        "How are prices set in ORCA?",
        "Can you pay more per kg or bargain?",
    ]
    for q in queries:
        result = knowledge_service.lookup(q)
        assert result.matched is True
        assert result.answerable is True
        assert result.category == KnowledgeCategory.PRICING_POLICY
        assert result.source_article_id == "kb_authoritative_rates"
        assert "fixed-rate" in result.answer.lower()
        assert "pricingservice" in result.answer.lower()


def test_pricing_policy_billing_calculation():
    result = knowledge_service.lookup("How is total billing calculated?")
    assert result.matched is True
    assert result.answerable is True
    assert result.category == KnowledgeCategory.PRICING_POLICY
    assert result.source_article_id == "kb_billing_calculation"
    assert "billingservice" in result.answer.lower()


# -----------------------------------------------------------------------------
# 4. Logistics Policy Lookups
# -----------------------------------------------------------------------------
def test_logistics_policy_collection_flow():
    result = knowledge_service.lookup("Who picks up the produce and how does collection work?")
    assert result.matched is True
    assert result.answerable is True
    assert result.category == KnowledgeCategory.LOGISTICS_POLICY
    assert result.source_article_id == "kb_collection_runner_flow"
    assert "farm-gate" in result.answer.lower()
    assert "runner" in result.answer.lower()


def test_logistics_policy_pickup_status():
    result = knowledge_service.lookup("When will the runner arrive to pick up my produce?")
    assert result.matched is True
    assert result.answerable is True
    assert result.category == KnowledgeCategory.LOGISTICS_POLICY
    assert result.source_article_id == "kb_pickup_status"
    assert "runner" in result.answer.lower()


def test_logistics_policy_pickup_reschedule():
    result = knowledge_service.lookup("Can I reschedule my pickup time or location?")
    assert result.matched is True
    assert result.answerable is True
    assert result.category == KnowledgeCategory.LOGISTICS_POLICY
    assert result.source_article_id == "kb_pickup_reschedule"
    assert "reschedule" in result.answer.lower()


# -----------------------------------------------------------------------------
# 5. Payment Policy Lookups
# -----------------------------------------------------------------------------
def test_payment_policy_flow():
    queries = [
        "When do I get paid?",
        "How does payment and payout work?",
        "What is the payment method?",
    ]
    for q in queries:
        result = knowledge_service.lookup(q)
        assert result.matched is True
        assert result.answerable is True
        assert result.category == KnowledgeCategory.PAYMENT_POLICY
        assert result.source_article_id == "kb_payment_workflow"
        assert "electronic payment" in result.answer.lower()


# -----------------------------------------------------------------------------
# 6. Dispute Resolution Lookups
# -----------------------------------------------------------------------------
def test_dispute_resolution_lookup():
    queries = [
        "What happens if the runner does not show up?",
        "What if there is a failed pickup or problem?",
        "How are disputes and complaints handled?",
    ]
    for q in queries:
        result = knowledge_service.lookup(q)
        assert result.matched is True
        assert result.answerable is True
        assert result.category == KnowledgeCategory.DISPUTE_RESOLUTION
        assert result.source_article_id == "kb_dispute_exception_handling"
        assert "exception" in result.answer.lower()


# -----------------------------------------------------------------------------
# 7. System Capabilities Lookups
# -----------------------------------------------------------------------------
def test_system_capabilities_lookup():
    queries = [
        "What can ORCA do and what are its limitations?",
        "Can ORCA give agricultural loans or insurance?",
        "What are the capabilities of ORCA?",
    ]
    for q in queries:
        result = knowledge_service.lookup(q)
        assert result.matched is True
        assert result.answerable is True
        assert result.category == KnowledgeCategory.SYSTEM_CAPABILITIES
        assert result.source_article_id == "kb_system_scope_boundaries"
        assert "does not negotiate" in result.answer.lower()


# -----------------------------------------------------------------------------
# 8. Category Filtering
# -----------------------------------------------------------------------------
def test_category_filtering():
    logistics_res = knowledge_service.lookup("status", category=KnowledgeCategory.LOGISTICS_POLICY)
    assert logistics_res.matched is True
    assert logistics_res.category == KnowledgeCategory.LOGISTICS_POLICY
    assert logistics_res.source_article_id == "kb_pickup_status"

    procurement_res = knowledge_service.lookup("status", category=KnowledgeCategory.PROCUREMENT_PROCESS)
    assert procurement_res.matched is True
    assert procurement_res.category == KnowledgeCategory.PROCUREMENT_PROCESS
    assert procurement_res.source_article_id == "kb_checking_order_status"

    mismatch_res = knowledge_service.lookup(
        "Who is the runner picking up produce?",
        category=KnowledgeCategory.PAYMENT_POLICY,
    )
    assert mismatch_res.matched is False
    assert mismatch_res.answerable is False
    assert mismatch_res.clarification_required is True


# -----------------------------------------------------------------------------
# 9. Case-Insensitive Matching
# -----------------------------------------------------------------------------
def test_case_insensitive_matching():
    queries = [
        "WHAT IS ORCA?",
        "how does BILLING work",
        "WhEn Do I GeT PaId?",
        "DiSpUtE ReSoLuTiOn",
    ]
    for q in queries:
        result = knowledge_service.lookup(q)
        assert result.matched is True, f"Failed for case variation: {q}"
        assert result.answerable is True


# -----------------------------------------------------------------------------
# 10. No-Match Behavior
# -----------------------------------------------------------------------------
def test_no_match_behavior():
    off_topic_queries = [
        "Who won the World Cup in 1998?",
        "Explain quantum electrodynamics to me",
        "How do I bake a chocolate cake?",
        "Write a poem about the ocean",
        "What is the stock price of Tesla?",
    ]
    for q in off_topic_queries:
        result = knowledge_service.lookup(q)
        assert result.matched is False, f"Unexpected match for: {q}"
        assert result.answerable is False
        assert result.clarification_required is True
        assert result.source_article_id is None
        assert result.category is None
        assert result.answer is None


# -----------------------------------------------------------------------------
# 11. Weak / Ambiguous Query Behavior
# -----------------------------------------------------------------------------
def test_weak_ambiguous_query_behavior():
    weak_queries = [
        "",
        "   ",
        "a",
        "the",
        "hello",
        "ok",
        "yes",
        "what",
        "how",
        "stuff",
        "xyz123",
    ]
    for q in weak_queries:
        result = knowledge_service.lookup(q)
        assert result.matched is False, f"Weak query unexpectedly matched: {q}"
        assert result.answerable is False
        assert result.clarification_required is True
        assert result.answer is None


# -----------------------------------------------------------------------------
# 12. Knowledge Contains No Hardcoded Authoritative Prices
# -----------------------------------------------------------------------------
def test_knowledge_contains_no_hardcoded_authoritative_prices():
    price_patterns = [
        re.compile(r"\$\s*\d+(?:\.\d+)?"),
        re.compile(r"\b\d+(?:\.\d+)?\s*(?:usd|kes|inr|cents?|dollars?)\b", re.IGNORECASE),
        re.compile(r"\b(?:0\.40|0\.45|0\.50|0\.30|0\.35)\b"),
    ]

    for article in knowledge_service.articles:
        text_to_check = f"{article.title} {article.summary} {article.content} {' '.join(article.keywords)}"
        for pattern in price_patterns:
            matches = pattern.findall(text_to_check)
            assert not matches, (
                f"Article '{article.article_id}' contains hardcoded price: {matches}"
            )


# -----------------------------------------------------------------------------
# 13-16. Zero Mutation Invariants (Safety)
# -----------------------------------------------------------------------------
def test_zero_order_mutation():
    initial_orders = list(order_service._orders.keys())
    
    knowledge_service.lookup("What is ORCA?")
    knowledge_service.lookup("How do I sell produce?")
    knowledge_service.lookup("Confirm order")
    knowledge_service.lookup("Pay for order")
    
    after_orders = list(order_service._orders.keys())
    assert initial_orders == after_orders


def test_zero_payment_mutation():
    initial_payments = list(payment_service._payments.keys())
    
    knowledge_service.lookup("When do I get paid?")
    knowledge_service.lookup("Authorize payment now")
    
    after_payments = list(payment_service._payments.keys())
    assert initial_payments == after_payments


def test_zero_collection_mutation():
    initial_tasks = list(collection_service._tasks.keys())
    
    knowledge_service.lookup("Who picks up my produce?")
    knowledge_service.lookup("Runner failed pickup")
    
    after_tasks = list(collection_service._tasks.keys())
    assert initial_tasks == after_tasks


def test_zero_order_state_mutation():
    test_order = order_service.create_order(
        farmer_id="farmer_test_kb",
        produce_type="potato",
        quantity=50.0,
        unit="kg",
        pickup_location="North Farm",
        validated_rate=0.40,
        region_code="GLOBAL_DEFAULT",
    )
    initial_state = test_order.status
    
    knowledge_service.lookup("When do I get paid?")
    knowledge_service.lookup("Who is my runner?")
    knowledge_service.lookup("Can I negotiate?")
    
    refetched = order_service.get_order(test_order.id)
    assert refetched is not None
    assert refetched.status == initial_state


# -----------------------------------------------------------------------------
# 17. KnowledgeResult Serialization
# -----------------------------------------------------------------------------
def test_knowledge_result_serialization():
    res = knowledge_service.lookup("What is ORCA?")
    assert res.matched is True
    
    data_dict = res.model_dump()
    assert isinstance(data_dict, dict)
    assert data_dict["matched"] is True
    assert data_dict["category"] == "COMPANY_INFO"
    assert data_dict["source_article_id"] == "kb_orca_overview"
    assert data_dict["confidence"] > 0
    
    json_str = res.model_dump_json()
    assert isinstance(json_str, str)
    
    restored = KnowledgeResult.model_validate_json(json_str)
    assert restored.matched == res.matched
    assert restored.source_article_id == res.source_article_id
    assert restored.category == res.category
    assert restored.answer == res.answer


# -----------------------------------------------------------------------------
# 18. Absence of Cora / Springhill Artifacts
# -----------------------------------------------------------------------------
def test_no_cora_or_springhill_content():
    forbidden_terms = [
        "cora",
        "springhill",
        "bsfl",
        "black soldier fly",
        "carbon credit",
        "carbon-negative",
        "uco",
        "used cooking oil",
        "farm tour",
        "grade a",
        "grade b",
        "profitability guarantee",
        "insect protein",
    ]
    for article in knowledge_service.articles:
        text_to_check = f"{article.title} {article.summary} {article.content} {' '.join(article.keywords)}".lower()
        for term in forbidden_terms:
            assert term not in text_to_check, (
                f"Article '{article.article_id}' contains ungrounded/forbidden term: '{term}'"
            )

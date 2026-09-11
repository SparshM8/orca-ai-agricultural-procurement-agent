"""Authoritative Business Knowledge Service for ORCA Agricultural Procurement.

Provides grounded informational Q&A and operational FAQ guidance based strictly
on SRS v2.0, authoritative system behavior, and configured business policies.

CRITICAL INVARIANTS:
1. Zero state mutation: never creates orders, updates offers, creates payments,
   alters collection tasks, or mutates OrderState.
2. Zero price fabrication: never hardcodes or invents prices; defers dynamic rates
   to PricingService.
3. Zero unsupported operational promises or foreign entities (no Cora/Springhill).
4. Deterministic matching: returns structured KnowledgeResult; never fabricates
   an answer for ungrounded or ambiguous queries.
"""

import re
from typing import List, Optional, Dict, Any, Set
from orca.domain.knowledge import (
    KnowledgeCategory,
    KnowledgeArticle,
    KnowledgeResult,
)

# Common non-informative stop words filtered during tokenization
STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "to", "from", "in", "on", "at", "by", "for", "with", "about", "into",
    "of", "i", "you", "he", "she", "it", "we", "they", "my", "your", "our",
    "their", "this", "that", "these", "those", "can", "will", "would", "should",
    "could", "me", "us", "him", "her", "them", "some", "any", "please", "tell",
    "what", "who", "when", "where", "why", "how", "which", "there", "here",
}


def _tokenize(text: str) -> List[str]:
    """Tokenize and clean text into lowercase alphanumeric words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in cleaned.split() if len(w) > 1 and w not in STOP_WORDS]


# Authoritative, grounded knowledge articles pre-seeded for ORCA
DEFAULT_ORCA_KNOWLEDGE_ARTICLES: List[KnowledgeArticle] = [
    # -------------------------------------------------------------
    # 1. COMPANY_INFO
    # -------------------------------------------------------------
    KnowledgeArticle(
        article_id="kb_orca_overview",
        category=KnowledgeCategory.COMPANY_INFO,
        title="ORCA Agricultural Procurement Platform Overview",
        summary="Direct digital agricultural procurement platform connecting farmers to buyers with transparent rates, guaranteed runner pickup, and digital payouts.",
        content=(
            "ORCA (Open Regional Crop Aggregation / Procurement) is a direct digital agricultural procurement platform. "
            "We connect farmers directly with buyers and institutional demand. Farmers receive transparent, authoritative "
            "fixed market purchase rates, guaranteed farm-gate collection by vetted local runners, and rapid electronic payouts "
            "upon order confirmation, eliminating middleman broker fees and transit uncertainty."
        ),
        keywords=[
            "what is orca", "about orca", "who are you", "orca company",
            "tell me about orca", "orca platform", "orca mission",
            "direct procurement platform", "about the orca procurement platform",
            "what does orca do", "orca", "what orca does",
            "procurement model", "what is your procurement model", "how does orca work",
        ],
        related_topics=["procurement_process", "pricing_policy", "logistics_policy"],
        source_type="SRS_DOCUMENTATION",
        verified=True,
    ),

    # -------------------------------------------------------------
    # 2. PROCUREMENT_PROCESS
    # -------------------------------------------------------------
    KnowledgeArticle(
        article_id="kb_start_offer",
        category=KnowledgeCategory.PROCUREMENT_PROCESS,
        title="How to Start a Produce Offer",
        summary="Provide produce type, quantity, unit, pickup location, and availability window to submit an offer.",
        content=(
            "To sell produce through ORCA, provide your offer details in conversation: the produce type (e.g., potatoes, tomatoes), "
            "the quantity and measurement unit (e.g., 50 kg, 2 tons), your farm pickup location, and when your harvest will be ready "
            "for collection (availability window). ORCA validates your offer against supported regional produce and constraints "
            "before presenting a clear transaction summary."
        ),
        keywords=[
            "start offer", "sell produce", "make an offer", "how to sell", "submit offer",
            "offer produce", "new offer", "sell crop", "selling produce", "how do i sell",
            "how to sell produce", "submit an offer for my harvest", "submit harvest",
            "start an offer to sell produce", "how to sell produce through orca",
        ],
        related_topics=["procurement_confirmation", "pricing_policy"],
        source_type="SYSTEM_BEHAVIOR",
        verified=True,
    ),
    KnowledgeArticle(
        article_id="kb_confirmation_flow",
        category=KnowledgeCategory.PROCUREMENT_PROCESS,
        title="Order Confirmation Workflow",
        summary="Explicit farmer agreement is required before an authoritative order record is created.",
        content=(
            "Once valid offer details are provided and verified against backend rates, ORCA presents a transaction summary "
            "displaying the quantity, authoritative rate, and total payout. An order is only created when the farmer explicitly "
            "accepts the terms (e.g., 'Confirm', 'I agree', or 'Deal'). Until confirmed, no order record is generated and no "
            "obligations are locked."
        ),
        keywords=[
            "how does order confirmation work", "order confirmation", "confirm order",
            "confirmation workflow", "accept offer", "agree to terms",
            "explicit confirmation", "how confirmation works",
        ],
        related_topics=["procurement_start_offer", "procurement_amendment", "payment_flow"],
        source_type="SYSTEM_BEHAVIOR",
        verified=True,
    ),
    KnowledgeArticle(
        article_id="kb_offer_amendment",
        category=KnowledgeCategory.PROCUREMENT_PROCESS,
        title="Offer and Order Amendments",
        summary="Offers can be freely amended prior to confirmation; confirmed orders require structured rescheduling or exception flow.",
        content=(
            "Prior to explicit confirmation, farmers can freely amend any offer attribute, such as quantity, produce type, "
            "pickup location, or availability window (e.g., 'Actually, change that to 80 kg'). Once confirmed, order details "
            "are locked to protect the logistics schedule and billing records. Post-confirmation changes require pickup rescheduling "
            "or runner exception handling."
        ),
        keywords=[
            "amend", "amendment", "amend offer", "modify offer", "change offer",
            "change quantity", "update offer", "edit offer", "correct offer",
            "can i amend or change my offer", "amend or change my offer before confirming",
        ],
        related_topics=["procurement_confirmation", "logistics_reschedule"],
        source_type="SYSTEM_BEHAVIOR",
        verified=True,
    ),
    KnowledgeArticle(
        article_id="kb_post_confirmation",
        category=KnowledgeCategory.PROCUREMENT_PROCESS,
        title="What Happens After Order Confirmation",
        summary="Order creation, deterministic billing generation, payment initiation, and runner pickup scheduling.",
        content=(
            "After you confirm an order, ORCA creates a permanent order record with an order ID (e.g., ORD-XXXX). "
            "Deterministic billing is finalized, an electronic payment request is initiated, and a collection task is "
            "automatically generated in the runner dispatch pool for collection at your farm gate."
        ),
        keywords=[
            "what happens after i confirm the order", "what happens after confirmation",
            "after confirmation", "post confirmation", "what happens next after confirm",
            "after order is confirmed",
        ],
        related_topics=["payment_flow", "logistics_collection"],
        source_type="SYSTEM_BEHAVIOR",
        verified=True,
    ),
    KnowledgeArticle(
        article_id="kb_checking_order_status",
        category=KnowledgeCategory.PROCUREMENT_PROCESS,
        title="Checking Order Status",
        summary="Query current lifecycle state of an order using its order ID.",
        content=(
            "You can check the status of your order at any time by asking ORCA with your order ID (e.g., 'What is the status "
            "of ORD-1234?'). ORCA will report the active lifecycle stage, such as AWAITING_FARMER_CONFIRMATION, ORDER_CONFIRMED, "
            "PAYMENT_PENDING, COLLECTION_ASSIGNED, or COMPLETED."
        ),
        keywords=[
            "how do i check the status of my order", "order status", "check order",
            "track order", "status of my order", "order progress", "where is my order",
            "status", "order status inquiry",
        ],
        related_topics=["logistics_status", "payment_status"],
        source_type="SYSTEM_BEHAVIOR",
        verified=True,
    ),

    # -------------------------------------------------------------
    # 3. PRICING_POLICY
    # -------------------------------------------------------------
    KnowledgeArticle(
        article_id="kb_authoritative_rates",
        category=KnowledgeCategory.PRICING_POLICY,
        title="Authoritative Fixed-Rate Pricing Policy",
        summary="Rates are determined dynamically by PricingService from regional market benchmarks and are non-negotiable.",
        content=(
            "ORCA operates on a transparent fixed-rate procurement model. Purchase rates are retrieved dynamically from the "
            "authoritative backend PricingService based on established regional market benchmarks. To guarantee fairness, consistency, "
            "and zero middleman bias, rates cannot be altered or bargained during conversation. Current rates for supported crops "
            "are dynamically provided by PricingService rather than static chat text."
        ),
        keywords=[
            "pricing policy", "how are prices set", "how are rates determined",
            "why is the rate fixed", "can i negotiate", "negotiate price", "bargain",
            "can you pay more per kg", "can you pay more", "counter offer",
            "fixed rate model", "why is the rate fixed and can i negotiate",
            "how are prices set in orca", "can you pay more per kg or bargain",
        ],
        related_topics=["pricing_billing_calculation", "company_overview"],
        source_type="CONFIGURED_POLICY",
        verified=True,
    ),
    KnowledgeArticle(
        article_id="kb_billing_calculation",
        category=KnowledgeCategory.PRICING_POLICY,
        title="Deterministic Billing and Payout Calculation",
        summary="Total amount calculated deterministically as quantity multiplied by authoritative rate, plus applicable regional tax.",
        content=(
            "Billing is calculated deterministically by BillingService: subtotal equals quantity multiplied by the authoritative "
            "rate per unit. Regional standard tax percentages and unit normalizations are applied where configured. The conversational "
            "agent never calculates prices or totals independently; all arithmetic is executed by the backend billing engine."
        ),
        keywords=[
            "how is total billing calculated", "billing calculation", "how is total calculated",
            "how does billing work", "billing formula", "invoice calculation",
            "subtotal and tax calculation", "how billing is calculated",
        ],
        related_topics=["pricing_authoritative_rates", "payment_flow"],
        source_type="SYSTEM_BEHAVIOR",
        verified=True,
    ),

    # -------------------------------------------------------------
    # 4. LOGISTICS_POLICY
    # -------------------------------------------------------------
    KnowledgeArticle(
        article_id="kb_collection_runner_flow",
        category=KnowledgeCategory.LOGISTICS_POLICY,
        title="Farm-Gate Collection and Runner Logistics",
        summary="Vetted runners collect produce directly from the farmer's location during the agreed window.",
        content=(
            "Once an order is confirmed, ORCA schedules a collection task for a local vetted runner. The runner travels directly "
            "to the farmer's registered pickup location for guaranteed farm-gate collection within the agreed availability window. "
            "The farmer does not need to transport produce to a central distribution depot."
        ),
        keywords=[
            "who picks up the produce", "how does collection work", "collection flow",
            "runner pickup", "who collects produce", "farm gate collection",
            "who picks up the produce and how does collection work",
            "how runner collection works", "farm pickup logistics",
        ],
        related_topics=["logistics_status", "logistics_reschedule", "dispute_handling"],
        source_type="SRS_DOCUMENTATION",
        verified=True,
    ),
    KnowledgeArticle(
        article_id="kb_pickup_status",
        category=KnowledgeCategory.LOGISTICS_POLICY,
        title="Checking Pickup and Runner Status",
        summary="Track runner assignment and scheduled pickup timing for confirmed orders.",
        content=(
            "Farmers can query pickup and runner status by asking ORCA (e.g., 'When will the runner arrive?' or 'Who is picking "
            "up my order?'). ORCA reports whether a task is pending, assigned to a specific runner ID, in progress, or completed."
        ),
        keywords=[
            "when will the runner arrive", "when will the runner arrive to pick up my produce",
            "pickup status", "runner status", "who is my runner", "who is picking up my order",
            "track pickup", "driver status", "is the runner coming", "status", "collection status",
        ],
        related_topics=["logistics_collection_flow", "procurement_order_status"],
        source_type="SYSTEM_BEHAVIOR",
        verified=True,
    ),
    KnowledgeArticle(
        article_id="kb_pickup_reschedule",
        category=KnowledgeCategory.LOGISTICS_POLICY,
        title="Rescheduling a Pickup",
        summary="Requesting a new pickup time or location prior to collection.",
        content=(
            "If a farmer cannot fulfill the original pickup window or needs to change the farm location, they can request a "
            "reschedule with their order ID (e.g., 'Change pickup to tomorrow at 2 PM'). The system updates the scheduled "
            "collection window and notifies logistics."
        ),
        keywords=[
            "reschedule pickup", "can i reschedule my pickup time or location",
            "change pickup time", "change pickup date", "postpone pickup",
            "reschedule collection", "change pickup location", "move pickup",
        ],
        related_topics=["logistics_collection_flow", "procurement_amendment"],
        source_type="SYSTEM_BEHAVIOR",
        verified=True,
    ),

    # -------------------------------------------------------------
    # 5. PAYMENT_POLICY
    # -------------------------------------------------------------
    KnowledgeArticle(
        article_id="kb_payment_workflow",
        category=KnowledgeCategory.PAYMENT_POLICY,
        title="Payment Processing and Payout Lifecycle",
        summary="Digital payment initiated upon order confirmation and settled through configured regional payment provider.",
        content=(
            "Payment workflow begins immediately when an order is confirmed. A payment request is generated and processed through "
            "the configured regional payment adapter. Payout statuses (PAYMENT_PENDING, PAYMENT_CONFIRMED) can be queried at any time. "
            "Farmers receive transparent, electronic payment directly with zero hidden platform deductions."
        ),
        keywords=[
            "when do i get paid", "how does payment work", "how do payments work", "how does payout work",
            "payment policy", "payment workflow", "what is the payment method",
            "when do i get paid and how does payout work", "electronic payment",
            "payout process", "payment methods", "payments work", "how payments work",
        ],
        related_topics=["pricing_billing_calculation", "procurement_confirmation"],
        source_type="SRS_DOCUMENTATION",
        verified=True,
    ),

    # -------------------------------------------------------------
    # 6. DISPUTE_RESOLUTION
    # -------------------------------------------------------------
    KnowledgeArticle(
        article_id="kb_dispute_exception_handling",
        category=KnowledgeCategory.DISPUTE_RESOLUTION,
        title="Dispute and Collection Problem Resolution",
        summary="Protocols for reporting runner no-shows, failed pickups, and logistics exceptions.",
        content=(
            "If a runner fails to arrive, arrives significantly late, or encounters an issue at the farm gate, the farmer or "
            "runner can report a collection problem. The collection task transitions to an EXCEPTION or FAILED state with the "
            "recorded issue reason. This triggers operational review and enables immediate rescheduling or resolution without "
            "financial loss to the farmer."
        ),
        keywords=[
            "what happens if the runner does not show up", "what if the runner does not show up",
            "what if there is a failed pickup or problem", "how are disputes and complaints handled",
            "runner did not show", "runner does not show", "runner late", "failed pickup",
            "collection problem", "dispute resolution", "report problem with runner",
            "disputes and complaints", "complaints", "exception handling",
        ],
        related_topics=["logistics_collection_flow", "logistics_reschedule"],
        source_type="SYSTEM_BEHAVIOR",
        verified=True,
    ),

    # -------------------------------------------------------------
    # 7. SYSTEM_CAPABILITIES
    # -------------------------------------------------------------
    KnowledgeArticle(
        article_id="kb_system_scope_boundaries",
        category=KnowledgeCategory.SYSTEM_CAPABILITIES,
        title="ORCA Capabilities and Operational Boundaries",
        summary="What ORCA is designed to do and what falls outside system boundaries.",
        content=(
            "ORCA is authorized to evaluate produce offers, validate quantities and units against regional constraints, "
            "provide dynamic authoritative rates from PricingService, calculate deterministic bills, create orders upon explicit "
            "confirmation, initiate payments, schedule runner collections, track statuses, and reschedule pickups. "
            "ORCA does NOT negotiate custom prices, accept unsupported produce, provide agricultural loans, offer crop insurance, "
            "or bypass backend validation rules."
        ),
        keywords=[
            "what can orca do and what are its limitations", "what can orca do",
            "what are the capabilities of orca", "can orca give agricultural loans or insurance",
            "system capabilities", "orca capabilities", "system boundaries",
            "limitations of orca", "can orca provide loans", "agricultural loans",
        ],
        related_topics=["company_overview", "pricing_authoritative_rates"],
        source_type="SRS_DOCUMENTATION",
        verified=True,
    ),
]


class BusinessKnowledgeService:
    """Authoritative business knowledge service for grounded Q&A and FAQ lookups.
    
    CRITICAL: Strictly read-only. Never mutates order state, creates transactions,
    or hardcodes pricing numbers.
    """

    def __init__(self, articles: Optional[List[KnowledgeArticle]] = None):
        self.articles: List[KnowledgeArticle] = (
            articles if articles is not None else list(DEFAULT_ORCA_KNOWLEDGE_ARTICLES)
        )

    def lookup(
        self,
        query: str,
        category: Optional[KnowledgeCategory] = None,
        min_confidence: float = 0.45,
    ) -> KnowledgeResult:
        """Deterministically search the structured knowledge catalog for grounded answers.
        
        Args:
            query: The natural language question or keyword string from the farmer.
            category: Optional KnowledgeCategory to restrict the search.
            min_confidence: Threshold required to return an answerable result (default 0.45).
            
        Returns:
            KnowledgeResult containing matched flag, confidence, and answer (or clarification).
        """
        if not query or not query.strip():
            return KnowledgeResult(
                matched=False,
                answerable=False,
                confidence=0.0,
                clarification_required=True,
            )

        query_clean = re.sub(r"[^\w\s]", " ", query.lower()).strip()
        query_tokens = _tokenize(query_clean)

        # Reject empty or purely stop-word queries
        if not query_tokens:
            return KnowledgeResult(
                matched=False,
                answerable=False,
                confidence=0.0,
                clarification_required=True,
            )

        # Filter candidate articles by category if specified
        candidates = self.articles
        if category is not None:
            candidates = [a for a in candidates if a.category == category]
            if not candidates:
                return KnowledgeResult(
                    matched=False,
                    answerable=False,
                    confidence=0.0,
                    category=category,
                    clarification_required=True,
                )

        best_article: Optional[KnowledgeArticle] = None
        best_score: float = 0.0

        for article in candidates:
            score = self._score_article(query_clean, query_tokens, article)
            if score > best_score:
                best_score = score
                best_article = article

        # Enforce strict confidence threshold
        if best_article and best_score >= min_confidence:
            return KnowledgeResult(
                matched=True,
                answerable=True,
                confidence=round(min(best_score, 1.0), 2),
                source_article_id=best_article.article_id,
                category=best_article.category,
                answer=best_article.content,
                clarification_required=False,
            )

        return KnowledgeResult(
            matched=False,
            answerable=False,
            confidence=round(min(best_score, 1.0), 2),
            source_article_id=None,
            category=None,
            answer=None,
            clarification_required=True,
        )

    def _score_article(
        self,
        query_clean: str,
        query_tokens: List[str],
        article: KnowledgeArticle,
    ) -> float:
        """Compute multi-factor match score between query and knowledge article."""
        score = 0.0
        query_set = set(query_tokens)
        if not query_set:
            return 0.0

        # 1. Exact phrase matching in keywords (highest precision)
        for kw in article.keywords:
            kw_clean = kw.lower().strip()
            if kw_clean == query_clean:
                return 1.0

            # If multi-word keyword appears as complete phrase in query
            if len(kw_clean.split()) >= 2:
                pattern = r"\b" + re.escape(kw_clean) + r"\b"
                if re.search(pattern, query_clean):
                    score = max(score, 0.90)

        # 2. Keyword token subset matching
        for kw in article.keywords:
            kw_tokens = set(_tokenize(kw))
            if kw_tokens and kw_tokens.issubset(query_set):
                if len(kw_tokens) >= 2:
                    score = max(score, 0.85)
                else:
                    score = max(score, 0.65)

        # 3. Overall keyword token overlap
        all_kw_tokens: Set[str] = set()
        for kw in article.keywords:
            all_kw_tokens.update(_tokenize(kw))

        overlap = len(query_set & all_kw_tokens)
        if overlap > 0:
            recall = overlap / len(query_set)
            score = max(score, 0.40 * recall + 0.35 * (overlap / len(all_kw_tokens)))

        # 4. Title token overlap
        title_tokens = set(_tokenize(article.title))
        t_overlap = len(query_set & title_tokens)
        if t_overlap > 0:
            t_recall = t_overlap / len(query_set)
            if t_recall >= 0.5:
                score = max(score, 0.60 * t_recall)

        # 5. Guard against single-word / generic queries
        if len(query_tokens) == 1 and score < 0.85:
            single_word = query_tokens[0]
            exact_single = any(single_word == kw.lower().strip() for kw in article.keywords)
            if not exact_single:
                score = 0.0

        return score


# Shared singleton instance
knowledge_service = BusinessKnowledgeService()

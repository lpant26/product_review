"""Classify review content into product and service feedback categories."""

from config import SERVICE_WEIGHT

PRODUCT_TOPICS = {
    "quality", "feature", "performance", "battery", "camera", "durability", "comfort",
    "sound", "display", "design", "screen", "build", "product", "device", "phone",
    "laptop", "headphone", "fit", "material", "working",
}
SERVICE_TOPICS = {
    "delivery", "shipping", "courier", "packaging", "seller", "refund", "return",
    "amazon", "dispatch", "delivery boy", "customer service", "support",
}


def assess_relevance(text: object) -> dict[str, object]:
    """Preserve service feedback while down-weighting its product-score influence."""
    content = str(text).lower()
    product_hits = [topic for topic in PRODUCT_TOPICS if topic in content]
    service_hits = [topic for topic in SERVICE_TOPICS if topic in content]
    relevant = bool(product_hits) or not bool(service_hits)
    categories = []
    if product_hits: categories.append("PRODUCT")
    category_map = {"delivery": "DELIVERY", "shipping": "DELIVERY", "courier": "DELIVERY", "packaging": "PACKAGING", "seller": "SELLER", "amazon": "SELLER", "refund": "CUSTOMER_SERVICE", "return": "CUSTOMER_SERVICE", "support": "CUSTOMER_SERVICE", "customer service": "CUSTOMER_SERVICE"}
    categories.extend(sorted({category_map[topic] for topic in service_hits if topic in category_map}))
    if not categories: categories.append("OTHER")
    return {
        "Is_Relevant": relevant,
        "Review_Category": " | ".join(categories),
        "Relevance_Weight": 1.0 if relevant else SERVICE_WEIGHT,
        "Product_Topics": ", ".join(sorted(product_hits)) if product_hits else "",
        "Service_Topics": ", ".join(sorted(service_hits)) if service_hits else "",
    }

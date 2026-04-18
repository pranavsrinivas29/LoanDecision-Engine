from prometheus_client import Counter, Gauge, Histogram


PREDICT_REQUESTS_TOTAL = Counter(
    "loan_predict_requests_total",
    "Total number of prediction requests",
)

EXPLAIN_REQUESTS_TOTAL = Counter(
    "loan_explain_requests_total",
    "Total number of explanation requests",
)

SUMMARIZE_REQUESTS_TOTAL = Counter(
    "loan_summarize_requests_total",
    "Total number of summary requests",
)

APPROVALS_TOTAL = Counter(
    "loan_approvals_total",
    "Total number of approved predictions",
)

REJECTIONS_TOTAL = Counter(
    "loan_rejections_total",
    "Total number of rejected predictions",
)

BACKEND_EXCEPTIONS_TOTAL = Counter(
    "loan_backend_exceptions_total",
    "Total number of backend exceptions",
    ["endpoint"],
)

MODEL_LOAD_FAILURES_TOTAL = Counter(
    "loan_model_load_failures_total",
    "Total number of model loading failures",
)

ACTIVE_MODEL_INFO = Gauge(
    "loan_active_model_info",
    "Active model metadata as labels",
    ["model_name", "model_version"],
)

PREDICTION_PROBABILITY = Histogram(
    "loan_prediction_probability",
    "Distribution of prediction probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

PREDICT_LATENCY_SECONDS = Histogram(
    "loan_predict_latency_seconds",
    "Latency for prediction requests in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

EXPLAIN_LATENCY_SECONDS = Histogram(
    "loan_explain_latency_seconds",
    "Latency for explanation requests in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)

SUMMARIZE_LATENCY_SECONDS = Histogram(
    "loan_summarize_latency_seconds",
    "Latency for summarization requests in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)
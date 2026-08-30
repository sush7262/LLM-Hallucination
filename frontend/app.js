/* ═══════════════════════════════════════════════════════════════════════
   RAG Hallucination Verifier — Frontend Logic
   Handles API calls, state management, and dynamic rendering.
   ═══════════════════════════════════════════════════════════════════════ */

// ── Configuration ──────────────────────────────────────────────────────
const API_BASE = "http://localhost:8000";

// ── State ──────────────────────────────────────────────────────────────
let currentMode = "rule_based";

// ── Preset Examples (from PROJECT_SPEC.md Section 5) ───────────────────
const EXAMPLES = {
    A: {
        name: "Revenue & Margins",
        answer:
            "The company announced quarterly revenue of 4.2 billion rupees, with profit margins increasing as a result of cutting costs in manufacturing.",
        passages: [
            "The company reported quarterly revenue of 4.2 billion rupees with a net profit margin of 11 percent.",
            "The company operates in the consumer electronics sector across South Asia.",
        ],
        expectedDecision: "PASS WITH FLAGS",
    },
    B: {
        name: "Eiffel Tower",
        answer: "The Eiffel Tower was built in 1889 and is located in Rome.",
        passages: [
            "The Eiffel Tower was completed in 1889 for the World's Fair. It is one of the most visited monuments in Paris, France.",
        ],
        expectedDecision: "REJECT / REGENERATE",
    },
};


// ═══════════════════════════════════════════════════════════════════════
//  Mode Selector
// ═══════════════════════════════════════════════════════════════════════

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll(".toggle-btn").forEach((btn) => {
        btn.classList.toggle("active", btn.dataset.mode === mode);
    });
}


// ═══════════════════════════════════════════════════════════════════════
//  Example Loader
// ═══════════════════════════════════════════════════════════════════════

function loadExample(id) {
    const ex = EXAMPLES[id];
    if (!ex) return;

    document.getElementById("answer-input").value = ex.answer;
    document.getElementById("passages-input").value = ex.passages.join("\n");

    // Brief visual feedback
    const btn = event.target.closest(".btn");
    if (btn) {
        btn.style.borderColor = "var(--accent)";
        setTimeout(() => (btn.style.borderColor = ""), 400);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  UI State Management
// ═══════════════════════════════════════════════════════════════════════

function showState(stateId) {
    const states = ["loading-state", "error-state", "empty-state", "results-container"];
    states.forEach((id) => {
        const el = document.getElementById(id);
        if (el) el.style.display = id === stateId ? "" : "none";
    });
}

function setLoading(loading) {
    const btn = document.getElementById("verify-btn");
    btn.disabled = loading;

    if (loading) {
        btn.innerHTML = `<span class="spinner" style="width:18px;height:18px;border-width:2px;margin:0;"></span> Verifying...`;
        showState("loading-state");
    } else {
        btn.innerHTML = `
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <polyline points="20 6 9 17 4 12"/>
            </svg>
            Verify`;
    }
}

function showError(message) {
    document.getElementById("error-message").textContent = message;
    showState("error-state");
}


// ═══════════════════════════════════════════════════════════════════════
//  API Call — Verify
// ═══════════════════════════════════════════════════════════════════════

async function verify() {
    const answer = document.getElementById("answer-input").value.trim();
    const passagesRaw = document.getElementById("passages-input").value.trim();

    // Validation
    if (!answer) {
        showError("Please enter a generated answer to verify.");
        return;
    }
    if (!passagesRaw) {
        showError("Please enter at least one retrieved passage.");
        return;
    }

    // Parse passages: one per line, skip empty lines
    const passages = passagesRaw
        .split("\n")
        .map((p) => p.trim())
        .filter((p) => p.length > 0);

    if (passages.length === 0) {
        showError("No valid passages found. Enter one passage per line.");
        return;
    }

    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/verify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                answer: answer,
                passages: passages,
                decomposition_mode: currentMode,
            }),
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(
                errorData.detail || `Server error: ${response.status} ${response.statusText}`
            );
        }

        const data = await response.json();
        renderResults(data);
    } catch (err) {
        if (err.name === "TypeError" && err.message.includes("fetch")) {
            showError(
                `Cannot reach backend at ${API_BASE}. Make sure the FastAPI server is running:\n` +
                `python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000`
            );
        } else {
            showError(err.message || "An unexpected error occurred.");
        }
    } finally {
        setLoading(false);
    }
}


// ═══════════════════════════════════════════════════════════════════════
//  Render Results
// ═══════════════════════════════════════════════════════════════════════

function renderResults(data) {
    showState("results-container");

    // ── Decision Banner ────────────────────────────────────────────────
    const banner = document.getElementById("decision-banner");
    const iconEl = document.getElementById("decision-icon");
    const labelEl = document.getElementById("decision-label");
    const metaEl = document.getElementById("decision-meta");

    // Clear previous classes
    banner.className = "decision-banner";

    if (data.decision.includes("REJECT")) {
        banner.classList.add("decision-reject");
        iconEl.textContent = "🚫";
        labelEl.textContent = "REJECT / REGENERATE";
        labelEl.style.color = "var(--color-contradicted)";
    } else if (data.decision.includes("FLAGS")) {
        banner.classList.add("decision-flags");
        iconEl.textContent = "⚠️";
        labelEl.textContent = "PASS WITH FLAGS";
        labelEl.style.color = "var(--color-unsupported)";
    } else {
        banner.classList.add("decision-pass");
        iconEl.textContent = "✅";
        labelEl.textContent = "PASS — Fully Supported";
        labelEl.style.color = "var(--color-supported)";
    }

    metaEl.textContent = `${data.claims.length} claim(s) verified in ${data.latency_ms}ms`;

    // ── Stats ──────────────────────────────────────────────────────────
    const counts = { supported: 0, unsupported: 0, contradicted: 0 };
    data.claims.forEach((c) => counts[c.label]++);

    document.getElementById("stat-total").textContent = data.claims.length;
    document.getElementById("stat-supported").textContent = counts.supported;
    document.getElementById("stat-unsupported").textContent = counts.unsupported;
    document.getElementById("stat-contradicted").textContent = counts.contradicted;

    // ── Donut Chart ────────────────────────────────────────────────────
    drawDonut(counts, data.claims.length);

    // ── Legend ──────────────────────────────────────────────────────────
    const legendEl = document.getElementById("chart-legend");
    legendEl.innerHTML = [
        { label: "Supported", color: "#22c55e", count: counts.supported },
        { label: "Unsupported", color: "#f59e0b", count: counts.unsupported },
        { label: "Contradicted", color: "#ef4444", count: counts.contradicted },
    ]
        .map(
            (item) => `
        <div class="legend-item">
            <span class="legend-dot" style="background: ${item.color}"></span>
            <span>${item.label}</span>
            <span class="legend-count">${item.count}</span>
        </div>`
        )
        .join("");

    // ── Claim Cards ────────────────────────────────────────────────────
    const claimsList = document.getElementById("claims-list");
    claimsList.innerHTML = data.claims
        .map(
            (claim, i) => `
        <div class="claim-card label-${claim.label}" style="animation-delay: ${i * 0.08}s">
            <div class="claim-header">
                <span class="claim-badge badge-${claim.label}">
                    ${getBadgeIcon(claim.label)} ${claim.label}
                </span>
                <span class="claim-confidence">
                    Confidence: ${(claim.confidence * 100).toFixed(1)}%
                </span>
            </div>
            <p class="claim-text">${escapeHtml(claim.claim)}</p>
            <div class="claim-evidence">
                <div class="claim-evidence-label">Aligned Evidence</div>
                <p class="claim-evidence-text">"${escapeHtml(claim.evidence)}"</p>
            </div>
        </div>`
        )
        .join("");
}


// ═══════════════════════════════════════════════════════════════════════
//  Donut Chart (Canvas)
// ═══════════════════════════════════════════════════════════════════════

function drawDonut(counts, total) {
    const canvas = document.getElementById("donut-chart");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;

    // High-DPI canvas
    canvas.width = 180 * dpr;
    canvas.height = 180 * dpr;
    canvas.style.width = "180px";
    canvas.style.height = "180px";
    ctx.scale(dpr, dpr);

    const cx = 90;
    const cy = 90;
    const outerR = 80;
    const innerR = 52;

    ctx.clearRect(0, 0, 180, 180);

    if (total === 0) {
        // Empty state
        ctx.beginPath();
        ctx.arc(cx, cy, outerR, 0, Math.PI * 2);
        ctx.arc(cx, cy, innerR, 0, Math.PI * 2, true);
        ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
        ctx.fill();
        return;
    }

    const segments = [
        { value: counts.supported, color: "#22c55e" },
        { value: counts.unsupported, color: "#f59e0b" },
        { value: counts.contradicted, color: "#ef4444" },
    ].filter((s) => s.value > 0);

    let startAngle = -Math.PI / 2; // start from top
    const gap = 0.04; // gap between segments in radians

    segments.forEach((seg) => {
        const sliceAngle = (seg.value / total) * (Math.PI * 2 - gap * segments.length);
        const endAngle = startAngle + sliceAngle;

        ctx.beginPath();
        ctx.arc(cx, cy, outerR, startAngle, endAngle);
        ctx.arc(cx, cy, innerR, endAngle, startAngle, true);
        ctx.closePath();
        ctx.fillStyle = seg.color;
        ctx.fill();

        startAngle = endAngle + gap;
    });

    // Center text
    ctx.fillStyle = "#f1f5f9";
    ctx.font = "bold 28px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(total, cx, cy - 6);

    ctx.fillStyle = "#64748b";
    ctx.font = "500 11px Inter, sans-serif";
    ctx.fillText("CLAIMS", cx, cy + 14);
}


// ═══════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════

function getBadgeIcon(label) {
    switch (label) {
        case "supported":
            return "✓";
        case "unsupported":
            return "?";
        case "contradicted":
            return "✗";
        default:
            return "•";
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Homework 6 — Statistical Independence:
 * Programmatic Contingency Table Construction, Marginal & Conditional Frequencies,
 * Expected Frequencies under Independence, Chi-Square Statistic & Exact p-Value,
 * and Statistical Association vs. Causality Interpretation.
 */

(function () {
  "use strict";

  // =========================================================================
  // 1. SPECIAL FUNCTIONS: LANCZOS LOG-GAMMA & REGULARIZED INCOMPLETE GAMMA
  // =========================================================================

  /**
   * Lanczos approximation for ln(Gamma(z)), accurate to ~15 significant digits.
   */
  function logGamma(z) {
    const g = 7;
    const p = [
      0.99999999999980993,
      676.5203681218851,
      -1259.1392167224028,
      771.32342877765313,
      -176.61502916214059,
      12.507343278686905,
      -0.13857109526572012,
      9.9843695780195716e-6,
      1.5056327351493116e-7
    ];
    if (z < 0.5) {
      return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * z)) - logGamma(1 - z);
    }
    z -= 1;
    let x = p[0];
    for (let i = 1; i < g + 2; i++) {
      x += p[i] / (z + i);
    }
    const t = z + g + 0.5;
    return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
  }

  /**
   * Lower regularized incomplete gamma P(a, x) via series expansion (used when x < a + 1).
   */
  function gammaSeriesP(a, x) {
    if (x <= 0) return 0.0;
    const EPS = 1e-14;
    const MAX_ITER = 500;
    let ap = a;
    let sum = 1.0 / a;
    let del = sum;
    for (let n = 1; n <= MAX_ITER; n++) {
      ap += 1.0;
      del *= x / ap;
      sum += del;
      if (Math.abs(del) < Math.abs(sum) * EPS) break;
    }
    const logPrefactor = -x + a * Math.log(x) - logGamma(a);
    if (logPrefactor < -700) return 0.0;
    return sum * Math.exp(logPrefactor);
  }

  /**
   * Upper regularized incomplete gamma Q(a, x) via Legendre continued fraction / Modified Lentz
   * (used when x >= a + 1).
   */
  function gammaContinuedFractionQ(a, x) {
    const EPS = 1e-14;
    const TINY = 1e-30;
    const MAX_ITER = 500;
    let b = x + 1.0 - a;
    let c = 1.0 / TINY;
    let d = 1.0 / b;
    let h = d;
    for (let i = 1; i <= MAX_ITER; i++) {
      const an = -i * (i - a);
      b += 2.0;
      d = an * d + b;
      if (Math.abs(d) < TINY) d = TINY;
      c = b + an / c;
      if (Math.abs(c) < TINY) c = TINY;
      d = 1.0 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1.0) < EPS) break;
    }
    const logPrefactor = -x + a * Math.log(x) - logGamma(a);
    if (logPrefactor < -700) return 0.0;
    return h * Math.exp(logPrefactor);
  }

  /**
   * Upper-tail p-value for Chi-Square distribution with df degrees of freedom:
   * P(Chi^2_{df} >= chi2) = Q(df / 2, chi2 / 2).
   */
  function chiSquarePValue(chi2, df) {
    if (df <= 0) return 1.0;
    if (chi2 <= 0) return 1.0;
    const a = 0.5 * df;
    const x = 0.5 * chi2;
    if (x < a + 1.0) {
      return Math.max(0.0, Math.min(1.0, 1.0 - gammaSeriesP(a, x)));
    } else {
      return Math.max(0.0, Math.min(1.0, gammaContinuedFractionQ(a, x)));
    }
  }

  /**
   * Chi-square probability density function f(x; df).
   */
  function chiSquarePDF(x, df) {
    if (x <= 0) {
      if (df === 2) return 0.5;
      return 0.0;
    }
    const halfDf = 0.5 * df;
    const logPdf = (halfDf - 1) * Math.log(x) - 0.5 * x - halfDf * Math.LN2 - logGamma(halfDf);
    if (logPdf < -700) return 0.0;
    return Math.exp(logPdf);
  }

  /**
   * Inverse Chi-square critical value Chi^2_{1 - alpha, df} via Wilson-Hilferty initial guess + Newton-bisection.
   */
  function chiSquareCriticalValue(alpha, df) {
    if (df <= 0) return 0;
    let lo = 0.0;
    let hi = Math.max(df + 10 * Math.sqrt(2 * df), 50.0);
    while (chiSquarePValue(hi, df) > alpha && hi < 1e6) {
      hi *= 2.0;
    }
    for (let iter = 0; iter < 65; iter++) {
      const mid = 0.5 * (lo + hi);
      const p = chiSquarePValue(mid, df);
      if (p > alpha) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return 0.5 * (lo + hi);
  }

  // =========================================================================
  // 2. PRESET BIVARIATE DATASETS & MONTE CARLO GENERATORS
  // =========================================================================

  const PRESET_SCENARIOS = {
    mfa_breach: {
      title: "MFA Policy (X) vs. Intrusion Outcome (Y)",
      xLabel: "MFA Policy (X)",
      yLabel: "Intrusion Outcome (Y)",
      causalNote:
        "Direct protective mechanism (X -> Y) plus potential organizational maturity confounding: accounts protected by Hardware FIDO2 keys exhibit a sharply lower full-compromise rate (3.3%) than accounts with No MFA (42.9%). Since p << alpha, we strongly reject H0 (X and Y are statistically dependent).",
      tuples: [
        ["No MFA", "Compromised", 60],
        ["No MFA", "Phished Only", 45],
        ["No MFA", "Blocked", 35],
        ["SMS / TOTP 2FA", "Compromised", 28],
        ["SMS / TOTP 2FA", "Phished Only", 62],
        ["SMS / TOTP 2FA", "Blocked", 120],
        ["Hardware FIDO2", "Compromised", 5],
        ["Hardware FIDO2", "Phished Only", 18],
        ["Hardware FIDO2", "Blocked", 127]
      ]
    },
    os_severity: {
      title: "Server OS Family (X) vs. IDS Alert Severity (Y)",
      xLabel: "Server OS (X)",
      yLabel: "Alert Severity (Y)",
      causalNote:
        "Genuine statistical independence (X _||_ Y): the conditional distribution of alert severities ( ~50% Low, ~35% Medium, ~15% Critical ) is nearly identical across Linux, Windows Server, and FreeBSD. Small deviations (O_ij - E_ij) are pure random sampling noise (p > alpha), so we fail to reject H0.",
      tuples: [
        ["Linux", "Low", 92],
        ["Linux", "Medium", 61],
        ["Linux", "Critical", 27],
        ["Windows Server", "Low", 68],
        ["Windows Server", "Medium", 50],
        ["Windows Server", "Critical", 22],
        ["FreeBSD", "Low", 40],
        ["FreeBSD", "Medium", 29],
        ["FreeBSD", "Critical", 11]
      ]
    },
    icecream_drowning: {
      title: "Ice Cream Sales Tier (X) vs. Drowning Incidents (Y)",
      xLabel: "Ice Cream Sales (X)",
      yLabel: "Daily Drowning Reports (Y)",
      causalNote:
        "Classic Spurious Association driven by a Common Confounder Z (Ambient Summer Temperature: X <- Z -> Y). The Chi-square test correctly detects strong statistical dependence (p < 10^-15) between Ice Cream Sales and Drowning Incidents, yet manipulating X (banning ice cream, do(X = Low)) has zero causal effect on Y!",
      tuples: [
        ["Low Sales", "0 Incidents", 165],
        ["Low Sales", "1-2 Incidents", 45],
        ["Low Sales", "3+ Incidents", 10],
        ["Moderate Sales", "0 Incidents", 110],
        ["Moderate Sales", "1-2 Incidents", 78],
        ["Moderate Sales", "3+ Incidents", 32],
        ["High Sales", "0 Incidents", 35],
        ["High Sales", "1-2 Incidents", 67],
        ["High Sales", "3+ Incidents", 58]
      ]
    },
    training_phishing: {
      title: "Randomized Security Training (X) vs. Simulated Phishing Response (Y)",
      xLabel: "Assigned Group (X)",
      yLabel: "Employee Action (Y)",
      causalNote:
        "Randomized Controlled Trial (RCT) where treatment X is randomly assigned: because randomization severs all back-door confounding paths into X, P(Y | do(X = x)) = P(Y | X = x). Here the statistically significant Chi-square dependence (p < 10^-10) genuinely reflects a causal effect of interactive training on phishing resilience.",
      tuples: [
        ["Control (No Training)", "Clicked Link", 64],
        ["Control (No Training)", "Ignored Email", 58],
        ["Control (No Training)", "Reported to SOC", 28],
        ["Passive Video Module", "Clicked Link", 41],
        ["Passive Video Module", "Ignored Email", 65],
        ["Passive Video Module", "Reported to SOC", 44],
        ["Interactive Drill", "Clicked Link", 14],
        ["Interactive Drill", "Ignored Email", 52],
        ["Interactive Drill", "Reported to SOC", 84]
      ]
    }
  };

  /**
   * Generate a synthetic raw bivariate sample of size N = 500 under either:
   * - mode === "indep": H0 true, P(X=x_i, Y=y_j) = p_{i.} * p_{.j}
   * - mode === "dep":   H1 true, strong diagonal coupling between X and Y
   */
  function generateMonteCarloTuples(mode) {
    const xCats = ["Tier_A", "Tier_B", "Tier_C"];
    const yCats = ["Outcome_1", "Outcome_2", "Outcome_3"];
    const pX = [0.40, 0.35, 0.25];
    const pY = [0.50, 0.30, 0.20];
    const N = 500;

    const counts = [
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]
    ];

    for (let k = 0; k < N; k++) {
      // Sample X from pX
      const u1 = Math.random();
      const i = u1 < pX[0] ? 0 : u1 < pX[0] + pX[1] ? 1 : 2;

      let j = 0;
      if (mode === "indep") {
        // Sample Y independently from pY
        const u2 = Math.random();
        j = u2 < pY[0] ? 0 : u2 < pY[0] + pY[1] ? 1 : 2;
      } else {
        // Sample Y conditionally on X=i (strong association)
        const condProbs = [
          [0.72, 0.20, 0.08],
          [0.35, 0.48, 0.17],
          [0.18, 0.27, 0.55]
        ][i];
        const u2 = Math.random();
        j = u2 < condProbs[0] ? 0 : u2 < condProbs[0] + condProbs[1] ? 1 : 2;
      }
      counts[i][j] += 1;
    }

    const tuples = [];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        tuples.push([xCats[i], yCats[j], counts[i][j]]);
      }
    }
    return tuples;
  }

  /**
   * Convert an array of [x, y, count] tuples into formatted text for the raw dataset textarea.
   * Shows both weighted summary tuples and can also parse individual (X, Y) lines.
   */
  function formatTuplesToText(tuples, xLabel, yLabel) {
    const lines = [
      `# Format: ${xLabel || "X"}, ${yLabel || "Y"}, Count (or one raw "X, Y" pair per line)`
    ];
    for (const [x, y, w] of tuples) {
      lines.push(`${x}, ${y}, ${w}`);
    }
    return lines.join("\n");
  }

  /**
   * Parse the raw dataset textarea into an array of [x, y, weight] records.
   * Supports:
   *  - "X_val, Y_val" (each line counts as weight = 1)
   *  - "X_val, Y_val, count" (weighted tuple / grouped observations)
   */
  function parseRawDatasetText(rawText) {
    const lines = rawText.split(/\r?\n/);
    const records = [];
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#") || line.startsWith("//")) continue;
      const parts = line.split(/[,\t;]+/).map((s) => s.trim()).filter(Boolean);
      if (parts.length < 2) continue;
      // Skip optional CSV header row if third column is non-numeric word "count" or "freq"
      if (
        records.length === 0 &&
        parts.length >= 2 &&
        /^(x|row|var1)$/i.test(parts[0]) &&
        /^(y|col|var2|f)$/i.test(parts[1])
      ) {
        continue;
      }
      const xVal = parts[0];
      const yVal = parts[1];
      let weight = 1;
      if (parts.length >= 3) {
        const parsedW = Number(parts[2]);
        if (Number.isFinite(parsedW) && parsedW >= 0) {
          weight = parsedW;
        }
      }
      if (weight > 0) {
        records.push([xVal, yVal, weight]);
      }
    }
    return records;
  }

  // =========================================================================
  // 3. PROGRAMMATIC CONTINGENCY TABLE & CHI-SQUARE ENGINE
  // =========================================================================

  function computeContingencyAnalysis(records, alpha) {
    const rowSet = new Set();
    const colSet = new Set();
    for (const [x, y] of records) {
      rowSet.add(String(x));
      colSet.add(String(y));
    }
    const rows = Array.from(rowSet);
    const cols = Array.from(colSet);
    const R = rows.length;
    const C = cols.length;

    const rIdx = {};
    rows.forEach((r, i) => { rIdx[r] = i; });
    const cIdx = {};
    cols.forEach((c, j) => { cIdx[c] = j; });

    const O = Array.from({ length: R }, () => new Float64Array(C));
    const rowTotals = new Float64Array(R);
    const colTotals = new Float64Array(C);
    let N = 0;

    for (const [x, y, w = 1] of records) {
      const i = rIdx[String(x)];
      const j = cIdx[String(y)];
      O[i][j] += w;
      rowTotals[i] += w;
      colTotals[j] += w;
      N += w;
    }

    const rowMarginals = Array.from(rowTotals, (r) => (N > 0 ? r / N : 0));
    const colMarginals = Array.from(colTotals, (c) => (N > 0 ? c / N : 0));

    const E = Array.from({ length: R }, () => new Float64Array(C));
    const jointRel = Array.from({ length: R }, () => new Float64Array(C));
    const condGivenRow = Array.from({ length: R }, () => new Float64Array(C));
    const condGivenCol = Array.from({ length: R }, () => new Float64Array(C));
    const stdResiduals = Array.from({ length: R }, () => new Float64Array(C));
    const adjResiduals = Array.from({ length: R }, () => new Float64Array(C));
    const chiContrib = Array.from({ length: R }, () => new Float64Array(C));

    let chi2 = 0.0;
    let g2 = 0.0;
    let cellsBelow5 = 0;
    let minExpected = Infinity;

    for (let i = 0; i < R; i++) {
      for (let j = 0; j < C; j++) {
        const expIJ = N > 0 ? (rowTotals[i] * colTotals[j]) / N : 0;
        E[i][j] = expIJ;
        if (expIJ < minExpected) minExpected = expIJ;
        if (expIJ < 5.0) cellsBelow5++;

        jointRel[i][j] = N > 0 ? O[i][j] / N : 0;
        condGivenRow[i][j] = rowTotals[i] > 0 ? O[i][j] / rowTotals[i] : 0;
        condGivenCol[i][j] = colTotals[j] > 0 ? O[i][j] / colTotals[j] : 0;

        if (expIJ > 0) {
          const diff = O[i][j] - expIJ;
          const contrib = (diff * diff) / expIJ;
          chiContrib[i][j] = contrib;
          stdResiduals[i][j] = diff / Math.sqrt(expIJ);
          const denomAdj = Math.sqrt(
            expIJ * Math.max(1e-12, (1 - rowMarginals[i]) * (1 - colMarginals[j]))
          );
          adjResiduals[i][j] = diff / denomAdj;
          chi2 += contrib;

          if (O[i][j] > 0) {
            g2 += 2.0 * O[i][j] * Math.log(O[i][j] / expIJ);
          }
        }
      }
    }

    const df = Math.max(0, (R - 1) * (C - 1));
    const pValue = df > 0 ? chiSquarePValue(chi2, df) : 1.0;
    const critVal = df > 0 ? chiSquareCriticalValue(alpha, df) : 0.0;
    const minDim = Math.min(R - 1, C - 1);
    const cramerV = N > 0 && minDim > 0 ? Math.sqrt(chi2 / (N * minDim)) : 0.0;
    const pearsonC = N > 0 ? Math.sqrt(chi2 / (chi2 + N)) : 0.0;
    const totalCells = R * C;
    const cochranValid = minExpected >= 1.0 && cellsBelow5 <= 0.2 * totalCells;

    return {
      rows,
      cols,
      R,
      C,
      N,
      O,
      E,
      rowTotals,
      colTotals,
      rowMarginals,
      colMarginals,
      jointRel,
      condGivenRow,
      condGivenCol,
      stdResiduals,
      adjResiduals,
      chiContrib,
      chi2,
      g2,
      df,
      pValue,
      critVal,
      cramerV,
      pearsonC,
      cellsBelow5,
      minExpected,
      cochranValid
    };
  }

  // =========================================================================
  // 4. DOM RENDERING: STAT CARDS, CONTINGENCY TABLES & INTERPRETATION
  // =========================================================================

  function formatPValue(p) {
    if (p < 1e-16) return "< 1.00e-16";
    if (p < 0.0001) return p.toExponential(3);
    return p.toFixed(6);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function renderSummaryCards(res, alpha) {
    document.getElementById("statNDim").textContent =
      `N = ${res.N} (${res.R} \u00D7 ${res.C})`;
    document.getElementById("statDfSub").textContent =
      `df = (${res.R}\u22121)(${res.C}\u22121) = ${res.df}`;

    document.getElementById("statChi2").textContent = res.chi2.toFixed(4);
    document.getElementById("statCritSub").textContent =
      `Crit \u03C7\u00B2_{${(1 - alpha).toFixed(3)}, ${res.df}} = ${res.critVal.toFixed(4)}`;

    document.getElementById("statPValue").textContent = formatPValue(res.pValue);
    document.getElementById("statG2Sub").textContent =
      `G\u00B2 (Log-Likelihood) = ${res.g2.toFixed(4)}`;

    document.getElementById("statCramerV").textContent = res.cramerV.toFixed(4);
    document.getElementById("statPearsonCSub").textContent =
      `Pearson C = ${res.pearsonC.toFixed(4)}`;

    const isDependent = res.pValue < alpha;
    const decEl = document.getElementById("statDecision");
    decEl.textContent = isDependent ? "DEPENDENT (Reject H\u2080)" : "INDEPENDENT (Fail to Reject H\u2080)";
    decEl.style.color = isDependent ? "#0055FF" : "#111111";

    const cochranEl = document.getElementById("statCochranSub");
    cochranEl.textContent = res.cochranValid
      ? `Cochran OK (min E_ij = ${res.minExpected.toFixed(1)} \u2265 5)`
      : `Low E_ij warning (${res.cellsBelow5} cells < 5)`;
  }

  function renderContingencyTable(res, viewMode, meta) {
    const wrapper = document.getElementById("contingencyTableWrapper");
    const xLabel = (meta && meta.xLabel) || "X";
    const yLabel = (meta && meta.yLabel) || "Y";

    let html = `<table class="data-table"><thead><tr>`;
    html += `<th>${escapeHtml(xLabel)} \\ ${escapeHtml(yLabel)}</th>`;
    for (let j = 0; j < res.C; j++) {
      html += `<th class="num">${escapeHtml(res.cols[j])}</th>`;
    }
    html += `<th class="num">Row Marginal n<sub>i&bull;</sub> (f<sub>i&bull;</sub>)</th></tr></thead><tbody>`;

    for (let i = 0; i < res.R; i++) {
      html += `<tr><td><strong>${escapeHtml(res.rows[i])}</strong></td>`;
      for (let j = 0; j < res.C; j++) {
        const o = res.O[i][j];
        const e = res.E[i][j];
        const jRel = (res.jointRel[i][j] * 100).toFixed(2);
        const rCond = (res.condGivenRow[i][j] * 100).toFixed(2);
        const cCond = (res.condGivenCol[i][j] * 100).toFixed(2);
        const rStd = res.stdResiduals[i][j];
        const contrib = res.chiContrib[i][j];

        let cellContent = "";
        if (viewMode === "comprehensive") {
          const sign = rStd >= 0 ? "+" : "";
          cellContent =
            `<div><strong>O<sub>ij</sub> = ${o}</strong> &nbsp;|&nbsp; <span style="color:#0055FF;">E<sub>ij</sub> = ${e.toFixed(2)}</span></div>` +
            `<div style="font-size:11px; color:#555555;">Joint: ${jRel}% &bull; Row: ${rCond}% &bull; Col: ${cCond}%</div>` +
            `<div style="font-size:11px; color:#555555;">r<sub>ij</sub> = ${sign}${rStd.toFixed(2)} &bull; &chi;&sup2;<sub>ij</sub> = ${contrib.toFixed(3)}</div>`;
        } else if (viewMode === "counts_expected") {
          cellContent = `<strong>O = ${o}</strong><br><span style="color:#0055FF;">E = ${e.toFixed(2)}</span>`;
        } else if (viewMode === "joint_rel") {
          cellContent =
            `<strong>f<sub>ij</sub> = ${res.jointRel[i][j].toFixed(4)}</strong> (${jRel}%)<br>` +
            `<span style="font-size:11px; color:#555555;">Indep f<sub>i&bull;</sub>f<sub>&bull;j</sub> = ${(res.rowMarginals[i] * res.colMarginals[j]).toFixed(4)}</span>`;
        } else if (viewMode === "cond_row") {
          cellContent =
            `<strong>P(Y|X) = ${res.condGivenRow[i][j].toFixed(4)}</strong> (${rCond}%)<br>` +
            `<span style="font-size:11px; color:#0055FF;">Marginal P(Y) = ${res.colMarginals[j].toFixed(4)}</span>`;
        } else if (viewMode === "cond_col") {
          cellContent =
            `<strong>P(X|Y) = ${res.condGivenCol[i][j].toFixed(4)}</strong> (${cCond}%)<br>` +
            `<span style="font-size:11px; color:#0055FF;">Marginal P(X) = ${res.rowMarginals[i].toFixed(4)}</span>`;
        } else if (viewMode === "chi_contrib") {
          const sign = rStd >= 0 ? "+" : "";
          cellContent =
            `<strong>&chi;&sup2;<sub>ij</sub> = ${contrib.toFixed(4)}</strong><br>` +
            `<span style="font-size:11px; color:#555555;">r<sub>ij</sub> = ${sign}${rStd.toFixed(3)} (O&minus;E = ${(o - e).toFixed(2)})</span>`;
        }
        html += `<td class="num">${cellContent}</td>`;
      }

      html += `<td class="num" style="background:#F8F8F8;"><strong>${res.rowTotals[i]}</strong><br><span style="font-size:11px; color:#555555;">f<sub>${i + 1}&bull;</sub> = ${res.rowMarginals[i].toFixed(4)} (${(res.rowMarginals[i] * 100).toFixed(1)}%)</span></td></tr>`;
    }

    // Footer row for Column Marginals
    html += `<tr style="background:#F8F8F8; border-top: 2px solid #111111;"><td><strong>Col Marginal n<sub>&bull;j</sub> (f<sub>&bull;j</sub>)</strong></td>`;
    for (let j = 0; j < res.C; j++) {
      html += `<td class="num"><strong>${res.colTotals[j]}</strong><br><span style="font-size:11px; color:#555555;">f<sub>&bull;${j + 1}</sub> = ${res.colMarginals[j].toFixed(4)} (${(res.colMarginals[j] * 100).toFixed(1)}%)</span></td>`;
    }
    html += `<td class="num"><strong>N = ${res.N}</strong><br><span style="font-size:11px; color:#0055FF;">&Sigma; &chi;&sup2;<sub>ij</sub> = ${res.chi2.toFixed(4)}</span></td></tr>`;
    html += `</tbody></table>`;

    wrapper.innerHTML = html;
  }

  function renderConditionalTables(res) {
    // 1. Row Conditional Table P(Y = y_j | X = x_i)
    let rHtml = `<table class="data-table"><thead><tr><th>X \\ P(Y = y<sub>j</sub> | X = x<sub>i</sub>)</th>`;
    for (let j = 0; j < res.C; j++) {
      rHtml += `<th class="num">${escapeHtml(res.cols[j])}</th>`;
    }
    rHtml += `<th class="num">Row Sum</th></tr></thead><tbody>`;

    for (let i = 0; i < res.R; i++) {
      rHtml += `<tr><td><strong>${escapeHtml(res.rows[i])}</strong></td>`;
      for (let j = 0; j < res.C; j++) {
        const val = res.condGivenRow[i][j];
        const diff = val - res.colMarginals[j];
        const sign = diff >= 0 ? "+" : "";
        rHtml += `<td class="num">${val.toFixed(4)} <span style="font-size:11px; color:#555555;">(${sign}${(diff * 100).toFixed(1)}%)</span></td>`;
      }
      rHtml += `<td class="num">1.0000</td></tr>`;
    }
    rHtml += `<tr style="background:#F8F8F8; border-top:2px solid #111111;"><td><strong style="color:#0055FF;">Marginal P(Y = y<sub>j</sub>)</strong></td>`;
    for (let j = 0; j < res.C; j++) {
      rHtml += `<td class="num"><strong style="color:#0055FF;">${res.colMarginals[j].toFixed(4)}</strong></td>`;
    }
    rHtml += `<td class="num"><strong>1.0000</strong></td></tr></tbody></table>`;
    document.getElementById("rowCondTableWrapper").innerHTML = rHtml;

    // 2. Column Conditional Table P(X = x_i | Y = y_j)
    let cHtml = `<table class="data-table"><thead><tr><th>X \\ P(X = x<sub>i</sub> | Y = y<sub>j</sub>)</th>`;
    for (let j = 0; j < res.C; j++) {
      cHtml += `<th class="num">${escapeHtml(res.cols[j])}</th>`;
    }
    cHtml += `<th class="num" style="color:#0055FF;">Marginal P(X = x<sub>i</sub>)</th></tr></thead><tbody>`;

    for (let i = 0; i < res.R; i++) {
      cHtml += `<tr><td><strong>${escapeHtml(res.rows[i])}</strong></td>`;
      for (let j = 0; j < res.C; j++) {
        const val = res.condGivenCol[i][j];
        const diff = val - res.rowMarginals[i];
        const sign = diff >= 0 ? "+" : "";
        cHtml += `<td class="num">${val.toFixed(4)} <span style="font-size:11px; color:#555555;">(${sign}${(diff * 100).toFixed(1)}%)</span></td>`;
      }
      cHtml += `<td class="num" style="background:#F8F8F8;"><strong style="color:#0055FF;">${res.rowMarginals[i].toFixed(4)}</strong></td></tr>`;
    }
    cHtml += `<tr style="background:#F8F8F8; border-top:2px solid #111111;"><td><strong>Col Sum</strong></td>`;
    for (let j = 0; j < res.C; j++) {
      cHtml += `<td class="num">1.0000</td>`;
    }
    cHtml += `<td class="num"><strong>1.0000</strong></td></tr></tbody></table>`;
    document.getElementById("colCondTableWrapper").innerHTML = cHtml;
  }

  function renderDynamicInterpretation(res, alpha, meta) {
    const box = document.getElementById("dynamicInterpretationBox");
    const isDep = res.pValue < alpha;

    // Find cell with largest Chi-square contribution
    let maxI = 0, maxJ = 0, maxContrib = -1;
    for (let i = 0; i < res.R; i++) {
      for (let j = 0; j < res.C; j++) {
        if (res.chiContrib[i][j] > maxContrib) {
          maxContrib = res.chiContrib[i][j];
          maxI = i;
          maxJ = j;
        }
      }
    }

    const topCellName = `(${escapeHtml(res.rows[maxI])}, ${escapeHtml(res.cols[maxJ])})`;
    const topO = res.O[maxI][maxJ];
    const topE = res.E[maxI][maxJ].toFixed(2);
    const topR = res.stdResiduals[maxI][maxJ].toFixed(2);

    const vStrength =
      res.cramerV < 0.10
        ? "negligible"
        : res.cramerV < 0.30
        ? "moderate"
        : "strong";

    const causalNote =
      meta && meta.causalNote
        ? meta.causalNote
        : "Remember that Pearson's Chi-square tests symmetric observational association P(Y|X) != P(Y) and cannot by itself distinguish direct causation (X -> Y), reverse causation (Y -> X), or unobserved confounding (X <- Z -> Y) without experimental intervention or causal adjustment.";

    box.innerHTML =
      `<strong>Statistical Decision &amp; Residual Diagnosis:</strong> ` +
      `With <em>N</em> = ${res.N} observations across an ${res.R} &times; ${res.C} table (<em>df</em> = ${res.df}), ` +
      `Pearson's statistic is <strong>&chi;&sup2; = ${res.chi2.toFixed(4)}</strong> (critical threshold &chi;&sup2;<sub>${(1 - alpha).toFixed(3)}, ${res.df}</sub> = ${res.critVal.toFixed(4)}) ` +
      `with asymptotic <strong><em>p</em>-value = ${formatPValue(res.pValue)}</strong> and <strong>Cram&eacute;r's <em>V</em> = ${res.cramerV.toFixed(4)}</strong> (${vStrength} effect size). ` +
      (isDep
        ? `Because <em>p</em> &lt; &alpha; = ${alpha}, we <strong>reject <em>H</em><sub>0</sub></strong> and conclude that <em>X</em> and <em>Y</em> are <strong>statistically dependent</strong>. `
        : `Because <em>p</em> &ge; &alpha; = ${alpha}, we <strong>fail to reject <em>H</em><sub>0</sub></strong>; the data are consistent with <strong>statistical independence</strong> (conditional profiles <em>P</em>(<em>Y</em>|<em>X</em>) align closely with marginal <em>P</em>(<em>Y</em>)). `) +
      `The single largest cell discrepancy occurs at <strong>${topCellName}</strong> with observed <em>O<sub>ij</sub></em> = ${topO} vs. expected <em>E<sub>ij</sub></em> = ${topE} (standardized residual <em>r<sub>ij</sub></em> = ${topR}, contributing ${maxContrib.toFixed(3)} to &chi;&sup2;).<br><br>` +
      `<strong>Association vs. Causality Context:</strong> ${escapeHtml(causalNote)}`;
  }

  // =========================================================================
  // 5. CANVAS VISUALIZATIONS
  // =========================================================================

  /**
   * Left Canvas: Grouped Bar Chart of Row Conditional Profiles P(Y = y_j | X = x_i)
   * overlaid with horizontal dashed lines for Marginal P(Y = y_j).
   */
  function drawConditionalProfilesCanvas(res) {
    const canvas = document.getElementById("condProfileCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    const padL = 52, padR = 20, padT = 32, padB = 56;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    // Title
    ctx.fillStyle = "#111111";
    ctx.font = "bold 12px monospace";
    ctx.textAlign = "left";
    ctx.fillText("Conditional P(Y = y_j | X = x_i) vs. Marginal P(Y = y_j)", padL, 18);

    // Axes
    ctx.strokeStyle = "#CCCCCC";
    ctx.lineWidth = 1;
    for (let tick = 0; tick <= 1.0001; tick += 0.2) {
      const yPos = padT + plotH * (1 - tick);
      ctx.beginPath();
      ctx.moveTo(padL, yPos);
      ctx.lineTo(padL + plotW, yPos);
      ctx.stroke();

      ctx.fillStyle = "#555555";
      ctx.font = "11px monospace";
      ctx.textAlign = "right";
      ctx.fillText(tick.toFixed(1), padL - 6, yPos + 4);
    }

    ctx.strokeStyle = "#111111";
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    const C = res.C;
    const R = res.R;
    if (C === 0 || R === 0) return;

    const groupW = plotW / C;
    const innerW = groupW * 0.72;
    const barW = Math.min(28, innerW / R);
    const shades = ["#111111", "#555555", "#888888", "#AAAAAA", "#333333"];

    for (let j = 0; j < C; j++) {
      const groupStart = padL + j * groupW + (groupW - barW * R) / 2;

      // Draw bars for each row X = x_i
      for (let i = 0; i < R; i++) {
        const prob = res.condGivenRow[i][j];
        const barH = Math.max(0, prob * plotH);
        const x = groupStart + i * barW;
        const y = padT + plotH - barH;

        ctx.fillStyle = shades[i % shades.length];
        ctx.fillRect(x, y, Math.max(2, barW - 2), barH);

        // Value label on top of bar
        if (barW >= 16) {
          ctx.fillStyle = "#111111";
          ctx.font = "10px monospace";
          ctx.textAlign = "center";
          ctx.fillText((prob * 100).toFixed(0) + "%", x + (barW - 2) / 2, Math.max(padT + 10, y - 4));
        }
      }

      // Draw dashed blue horizontal line across the group for Marginal P(Y = y_j)
      const margY = padT + plotH * (1 - res.colMarginals[j]);
      ctx.save();
      ctx.strokeStyle = "#0055FF";
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(padL + j * groupW + 8, margY);
      ctx.lineTo(padL + (j + 1) * groupW - 8, margY);
      ctx.stroke();
      ctx.restore();

      // Category label on X-axis
      ctx.fillStyle = "#111111";
      ctx.font = "11px monospace";
      ctx.textAlign = "center";
      const label = res.cols[j].length > 14 ? res.cols[j].slice(0, 12) + ".." : res.cols[j];
      ctx.fillText(label, padL + (j + 0.5) * groupW, padT + plotH + 18);
    }

    // Mini legend for Row shades at bottom
    let legX = padL;
    const legY = H - 12;
    ctx.font = "10px monospace";
    ctx.textAlign = "left";
    for (let i = 0; i < Math.min(R, 4); i++) {
      ctx.fillStyle = shades[i % shades.length];
      ctx.fillRect(legX, legY - 8, 9, 9);
      ctx.fillStyle = "#333333";
      const rName = res.rows[i].length > 14 ? res.rows[i].slice(0, 12) + ".." : res.rows[i];
      ctx.fillText(`X=${rName}`, legX + 12, legY);
      legX += ctx.measureText(`X=${rName}`).width + 26;
    }
  }

  /**
   * Right Canvas: Top half shows Chi-square_{df} PDF, critical threshold, and observed Chi^2;
   * Bottom half shows horizontal lollipop/bar chart of standardized Pearson residuals r_{ij}.
   */
  function drawChiSquareAndResidualsCanvas(res, alpha) {
    const canvas = document.getElementById("chiDistCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Split canvas into Top (Chi-square PDF) and Bottom (Standardized Residuals r_ij)
    const padL = 52, padR = 20;
    const topT = 26, topH = 125;
    const botT = 195, botH = 115;
    const plotW = W - padL - padR;

    // --- TOP PANEL: Chi-Square PDF ---
    const df = Math.max(1, res.df);
    const xMax = Math.max(res.critVal * 1.6, Math.min(res.chi2 * 1.15, res.critVal * 3.5), df + 4 * Math.sqrt(2 * df), 12);
    const numSteps = 220;
    let maxPdf = 0;
    const pts = [];
    for (let k = 0; k <= numSteps; k++) {
      const x = (k / numSteps) * xMax;
      // Avoid x=0 singularity when df=1
      const safeX = Math.max(0.08, x);
      const y = chiSquarePDF(safeX, df);
      if (Number.isFinite(y) && y > maxPdf) maxPdf = y;
      pts.push({ x, y });
    }
    if (maxPdf <= 0) maxPdf = 0.5;

    ctx.fillStyle = "#111111";
    ctx.font = "bold 12px monospace";
    ctx.textAlign = "left";
    ctx.fillText(
      `\u03C7\u00B2(df=${res.df}) Density | Crit = ${res.critVal.toFixed(2)}, Obs \u03C7\u00B2 = ${res.chi2.toFixed(2)}`,
      padL,
      16
    );

    // Shade rejection region x >= critVal
    ctx.fillStyle = "rgba(136, 136, 136, 0.28)";
    ctx.beginPath();
    const critXCanvas = padL + Math.min(1, res.critVal / xMax) * plotW;
    ctx.moveTo(critXCanvas, topT + topH);
    for (const pt of pts) {
      if (pt.x >= res.critVal) {
        const cx = padL + (pt.x / xMax) * plotW;
        const cy = topT + topH - (pt.y / maxPdf) * (topH - 10);
        ctx.lineTo(cx, cy);
      }
    }
    ctx.lineTo(padL + plotW, topT + topH);
    ctx.closePath();
    ctx.fill();

    // Draw PDF curve
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    pts.forEach((pt, idx) => {
      const cx = padL + (pt.x / xMax) * plotW;
      const cy = topT + topH - (pt.y / maxPdf) * (topH - 10);
      if (idx === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    ctx.stroke();

    // Baseline
    ctx.strokeStyle = "#111111";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, topT + topH);
    ctx.lineTo(padL + plotW, topT + topH);
    ctx.stroke();

    // Critical value line
    ctx.save();
    ctx.strokeStyle = "#555555";
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(critXCanvas, topT);
    ctx.lineTo(critXCanvas, topT + topH);
    ctx.stroke();
    ctx.restore();

    ctx.fillStyle = "#555555";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`Crit=${res.critVal.toFixed(2)}`, critXCanvas, topT + topH + 13);

    // Observed Chi2 marker
    const obsClamped = Math.min(xMax, res.chi2);
    const obsXCanvas = padL + (obsClamped / xMax) * plotW;
    ctx.strokeStyle = "#0055FF";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(obsXCanvas, topT);
    ctx.lineTo(obsXCanvas, topT + topH);
    ctx.stroke();

    ctx.fillStyle = "#0055FF";
    ctx.font = "bold 10px monospace";
    ctx.textAlign = obsXCanvas > padL + plotW - 70 ? "right" : "left";
    const labelOffset = obsXCanvas > padL + plotW - 70 ? -4 : 4;
    ctx.fillText(
      res.chi2 > xMax ? `Obs=${res.chi2.toFixed(1)} \u2192` : `Obs=${res.chi2.toFixed(2)}`,
      obsXCanvas + labelOffset,
      topT + 12
    );

    // --- BOTTOM PANEL: Standardized Pearson Residuals r_ij = (O_ij - E_ij)/sqrt(E_ij) ---
    ctx.fillStyle = "#111111";
    ctx.font = "bold 12px monospace";
    ctx.textAlign = "left";
    ctx.fillText("Standardized Cell Residuals r_ij = (O_ij \u2212 E_ij) / \u221AE_ij", padL, botT - 8);

    const flatResiduals = [];
    let maxAbsR = 2.5;
    for (let i = 0; i < res.R; i++) {
      for (let j = 0; j < res.C; j++) {
        const rVal = res.stdResiduals[i][j];
        if (Math.abs(rVal) > maxAbsR) maxAbsR = Math.abs(rVal);
        flatResiduals.push({
          label: `(${i + 1},${j + 1})`,
          r: rVal
        });
      }
    }
    maxAbsR *= 1.12;

    const zeroY = botT + botH / 2;
    // Reference lines at r = 0 and r = +/- 1.96
    ctx.strokeStyle = "#CCCCCC";
    ctx.lineWidth = 1;
    [-1.96, 1.96].forEach((ref) => {
      if (Math.abs(ref) < maxAbsR) {
        const ry = zeroY - (ref / maxAbsR) * (botH / 2);
        ctx.save();
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(padL, ry);
        ctx.lineTo(padL + plotW, ry);
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = "#555555";
        ctx.font = "9px monospace";
        ctx.textAlign = "right";
        ctx.fillText((ref > 0 ? "+" : "") + ref.toFixed(2), padL - 4, ry + 3);
      }
    });

    ctx.strokeStyle = "#111111";
    ctx.beginPath();
    ctx.moveTo(padL, zeroY);
    ctx.lineTo(padL + plotW, zeroY);
    ctx.stroke();
    ctx.fillStyle = "#111111";
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.fillText("0.0", padL - 4, zeroY + 3);

    const nCells = flatResiduals.length;
    const slotW = plotW / Math.max(1, nCells);
    const barW = Math.min(22, slotW * 0.62);

    flatResiduals.forEach((item, idx) => {
      const cx = padL + (idx + 0.5) * slotW;
      const barH = (item.r / maxAbsR) * (botH / 2);
      const isSig = Math.abs(item.r) >= 1.96;

      ctx.fillStyle = isSig ? "#0055FF" : "#555555";
      if (barH >= 0) {
        ctx.fillRect(cx - barW / 2, zeroY - barH, barW, barH);
      } else {
        ctx.fillRect(cx - barW / 2, zeroY, barW, -barH);
      }

      ctx.fillStyle = "#111111";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(item.label, cx, botT + botH + 14);
    });
  }

  // =========================================================================
  // 6. EVENT WIRING & INITIALIZATION
  // =========================================================================

  let currentScenarioMeta = PRESET_SCENARIOS.mfa_breach;

  function loadScenarioIntoTextarea(presetKey) {
    const textarea = document.getElementById("rawDatasetInput");
    if (presetKey === "synth_indep") {
      const tuples = generateMonteCarloTuples("indep");
      currentScenarioMeta = {
        title: "Monte Carlo Independent Sample (H0 True: p_{ij} = p_{i.} p_{.j})",
        xLabel: "Variable X",
        yLabel: "Feature Y",
        causalNote:
          "Generated by independent random sampling X ~ p_X and Y ~ p_Y (true Cramér's V = 0). About 95% of random draws at alpha = 0.05 yield p >= 0.05 (Fail to Reject H0), while ~5% produce a Type I false positive purely by chance."
      };
      textarea.value = formatTuplesToText(tuples, "Variable X", "Feature Y");
    } else if (presetKey === "synth_dep") {
      const tuples = generateMonteCarloTuples("dep");
      currentScenarioMeta = {
        title: "Monte Carlo Dependent Sample (H1 True: Coupled Conditional Distribution)",
        xLabel: "Variable X",
        yLabel: "Feature Y",
        causalNote:
          "Generated with strong conditional coupling P(Y | X = x_i) != P(Y). The Chi-square statistic consistently exceeds the critical threshold with p << 0.001."
      };
      textarea.value = formatTuplesToText(tuples, "Variable X", "Feature Y");
    } else if (PRESET_SCENARIOS[presetKey]) {
      const sc = PRESET_SCENARIOS[presetKey];
      currentScenarioMeta = sc;
      textarea.value = formatTuplesToText(sc.tuples, sc.xLabel, sc.yLabel);
    } else {
      currentScenarioMeta = {
        title: "Custom Raw Bivariate Dataset (X, Y)",
        xLabel: "X",
        yLabel: "Y",
        causalNote:
          "Custom dataset evaluated programmatically. Inspect the standardized Pearson residuals r_ij to identify which cells drive the Chi-square statistic, and consider confounding before inferring causality."
      };
    }
  }

  function runFullAnalysis() {
    const rawText = document.getElementById("rawDatasetInput").value;
    const alpha = parseFloat(document.getElementById("alphaSelect").value) || 0.05;
    const viewMode = document.getElementById("tableViewSelect").value;

    const records = parseRawDatasetText(rawText);
    if (records.length === 0) return;

    const res = computeContingencyAnalysis(records, alpha);
    renderSummaryCards(res, alpha);
    renderContingencyTable(res, viewMode, currentScenarioMeta);
    renderConditionalTables(res);
    renderDynamicInterpretation(res, alpha, currentScenarioMeta);
    drawConditionalProfilesCanvas(res);
    drawChiSquareAndResidualsCanvas(res, alpha);
  }

  document.addEventListener("DOMContentLoaded", () => {
    const presetSelect = document.getElementById("presetSelect");
    const alphaSelect = document.getElementById("alphaSelect");
    const tableViewSelect = document.getElementById("tableViewSelect");
    const btnCompute = document.getElementById("btnCompute");
    const btnResampleIndep = document.getElementById("btnResampleIndep");
    const btnResampleDep = document.getElementById("btnResampleDep");

    loadScenarioIntoTextarea(presetSelect.value);
    runFullAnalysis();

    presetSelect.addEventListener("change", () => {
      if (presetSelect.value !== "custom_raw") {
        loadScenarioIntoTextarea(presetSelect.value);
      }
      runFullAnalysis();
    });

    alphaSelect.addEventListener("change", runFullAnalysis);
    tableViewSelect.addEventListener("change", runFullAnalysis);
    btnCompute.addEventListener("click", runFullAnalysis);

    btnResampleIndep.addEventListener("click", () => {
      presetSelect.value = "synth_indep";
      loadScenarioIntoTextarea("synth_indep");
      runFullAnalysis();
    });

    btnResampleDep.addEventListener("click", () => {
      presetSelect.value = "synth_dep";
      loadScenarioIntoTextarea("synth_dep");
      runFullAnalysis();
    });
  });
})();

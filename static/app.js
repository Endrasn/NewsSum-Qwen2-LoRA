const $ = (id) => document.getElementById(id);

const state = {
  scores: {
    textrank: null,
    qwen: null,
  },
  chartTimer: null,
};

const demoText = `新华社北京电  近年来，随着移动互联网和社交平台快速发展，中文新闻资讯呈现爆发式增长。用户在短时间内需要从大量文本中识别核心事实、事件脉络和重要结论，自动摘要技术因此成为自然语言处理领域的重要应用方向。
在抽取式方法中，TextRank 能够根据句子之间的相似度关系为句子排序，直接选取原文中的关键句组成摘要，具有实现简单、无需训练数据等优点。与此同时，基于 Transformer 的预训练语言模型在生成式摘要任务中表现突出，能够在理解全文语义的基础上组织更自然的摘要表达。
本项目围绕中文新闻文本，构建 TextRank 与 Qwen2.5-7B-Instruct LoRA 两条摘要流程，并通过 ROUGE-1、ROUGE-2、ROUGE-L 等指标评估生成摘要与人工参考摘要之间的匹配程度，为抽取式和生成式方法的对比实验提供可视化入口。`;

const demoReference = "项目构建中文新闻自动摘要对比流程，使用 TextRank 进行抽取式摘要，并结合 Qwen2.5-7B-Instruct LoRA 生成摘要。系统通过 ROUGE 指标评估两种方法与人工摘要的匹配程度。";

function popElement(target) {
  if (!target) return;
  target.classList.remove("pop");
  void target.offsetWidth;
  target.classList.add("pop");
}

function setLoading(target, message) {
  target.textContent = message;
  target.classList.remove("error");
  popElement(target);
}

function setError(target, message) {
  target.textContent = message;
  target.classList.add("error");
  popElement(target);
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error("后端服务未启动或接口返回异常，请使用 python app.py 启动完整服务。");
  }

  if (!response.ok) {
    throw new Error(data.error || "请求失败");
  }
  return data;
}

function switchPage(pageId) {
  document.querySelectorAll(".page").forEach((page) => page.classList.toggle("active", page.id === pageId));
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("active", tab.dataset.page === pageId));
  popElement($(pageId));
  if (pageId === "rougePage") {
    scheduleRougeChartRender();
  }
}

function updateCounts() {
  $("sourceCount").textContent = `${$("sourceText").value.trim().length} 字`;
}

function bindRange(inputId, outputId, digits = 2) {
  const input = $(inputId);
  const output = $(outputId);
  const sync = () => {
    output.value = Number(input.value).toFixed(digits);
    popElement(output);
  };
  input.addEventListener("input", sync);
  sync();
}

function syncRangeValue(inputId, outputId, digits = 2) {
  $(outputId).value = Number($(inputId).value).toFixed(digits);
  popElement($(outputId));
}

function getQwenParams() {
  return {
    text: $("sourceText").value,
    model_path: $("modelPath").value,
    adapter_path: $("adapterPath").value,
    max_new_tokens: Number($("maxTokens").value),
    temperature: Number($("temperature").value),
    top_p: Number($("topP").value),
    repetition_penalty: Number($("repetitionPenalty").value),
  };
}

async function runTextrank() {
  const result = $("textrankResult");
  setLoading(result, "TextRank 正在计算句子权重...");
  state.scores.textrank = null;
  clearRougeChart();

  try {
    const data = await postJson("/api/textrank", {
      text: $("sourceText").value,
      ratio: Number($("textrankRatio").value),
      max_sentences: Number($("sentenceCount").value),
    });
    result.textContent = data.summary || "无摘要结果";
    popElement(result);
  } catch (error) {
    setError(result, error.message);
  }
}

async function runQwen() {
  const result = $("qwenResult");
  setLoading(result, "Qwen LoRA 正在推理，首次加载模型可能较慢...");
  state.scores.qwen = null;
  clearRougeChart();

  try {
    const data = await postJson("/api/qwen", getQwenParams());
    result.textContent = data.summary || "无摘要结果";
    popElement(result);
  } catch (error) {
    setError(result, error.message);
  }
}

async function runAll() {
  await runTextrank();
  await runQwen();
}

function validSummary(text) {
  const invalid = ["等待运行", "正在", "无摘要结果", "后端服务未启动"];
  return text && !invalid.some((word) => text.includes(word));
}

async function scoreOne(hypothesis) {
  if (!validSummary(hypothesis)) return null;
  return postJson("/api/rouge", {
    reference: $("referenceText").value,
    hypothesis,
  });
}

async function scoreBoth() {
  const metrics = $("metrics");
  metrics.innerHTML = `<p class="empty">正在计算 ROUGE...</p>`;
  popElement(metrics);

  try {
    const [textrank, qwen] = await Promise.all([
      scoreOne($("textrankResult").textContent),
      scoreOne($("qwenResult").textContent),
    ]);
    state.scores.textrank = textrank;
    state.scores.qwen = qwen;
    renderMetrics();
    switchPage("rougePage");
  } catch (error) {
    metrics.innerHTML = `<p class="error">${error.message}</p>`;
    popElement(metrics);
  }
}

async function scoreSingle(kind) {
  const target = kind === "textrank" ? $("textrankResult") : $("qwenResult");
  try {
    state.scores[kind] = await scoreOne(target.textContent);
    renderMetrics();
    switchPage("rougePage");
  } catch (error) {
    $("metrics").innerHTML = `<p class="error">${error.message}</p>`;
    popElement($("metrics"));
  }
}

function metricRow(label, data) {
  if (!data) return "";
  return `
    <div class="metric-group">
      <h3>${label}</h3>
      ${Object.entries(data)
        .map(
          ([name, value]) => `<div class="metric">
            <strong>${name.toUpperCase()}</strong>
            <span>P ${value.precision}</span>
            <span>R ${value.recall}</span>
            <span>F1 ${value.f1}</span>
          </div>`,
        )
        .join("")}
    </div>`;
}

function renderMetrics() {
  const html = [metricRow("TextRank", state.scores.textrank), metricRow("Qwen LoRA", state.scores.qwen)]
    .filter(Boolean)
    .join("");
  $("metrics").innerHTML = html || `<p class="empty">暂无评分。请先运行摘要并填写参考摘要。</p>`;
  popElement($("metrics"));
}

function scoreValues(score) {
  if (!score) return [0, 0, 0];
  return ["rouge-1", "rouge-2", "rouge-l"].map((key) => Number(score[key]?.f1 || 0));
}

function polygonPoints(values, center, radius) {
  return values
    .map((value, index) => {
      const angle = -Math.PI / 2 + index * ((Math.PI * 2) / 3);
      const distance = radius * Math.max(0, Math.min(1, value));
      return `${center.x + Math.cos(angle) * distance},${center.y + Math.sin(angle) * distance}`;
    })
    .join(" ");
}

function polygonPointObjects(values, center, radius) {
  return values.map((value, index) => {
    const angle = -Math.PI / 2 + index * ((Math.PI * 2) / 3);
    const distance = radius * Math.max(0, Math.min(1, value));
    return {
      x: center.x + Math.cos(angle) * distance,
      y: center.y + Math.sin(angle) * distance,
    };
  });
}

function pointString(points) {
  return points.map((point) => `${point.x},${point.y}`).join(" ");
}

function dataPoints(points, className, radius = 5.5) {
  return points.map((point) => `<circle cx="${point.x}" cy="${point.y}" r="${radius}" class="${className} data-point"></circle>`).join("");
}

function clearRougeChart() {
  const svg = $("rougeChart");
  if (!svg) return;
  svg.classList.add("chart-pending");
  svg.innerHTML = `
    <g class="chart-placeholder">
      <circle cx="260" cy="224" r="8"></circle>
      <text x="260" y="260" text-anchor="middle">Chart loading</text>
    </g>`;
}

function scheduleRougeChartRender(delay = 560) {
  clearTimeout(state.chartTimer);
  clearRougeChart();
  state.chartTimer = setTimeout(() => {
    drawRougeChart();
  }, delay);
}

function drawRougeChart() {
  const svg = $("rougeChart");
  svg.classList.remove("chart-pending");
  const center = { x: 260, y: 224 };
  const radius = 150;
  const labels = ["ROUGE-1", "ROUGE-2", "ROUGE-L"];
  const grid = [0.25, 0.5, 0.75, 1]
    .map((value) => `<polygon points="${polygonPoints([value, value, value], center, radius)}" class="grid-line"></polygon>`)
    .join("");
  const gridOverlay = [0.25, 0.5, 0.75, 1]
    .map((value) => `<polygon points="${polygonPoints([value, value, value], center, radius)}" class="grid-line grid-overlay"></polygon>`)
    .join("");
  const axis = labels
    .map((label, index) => {
      const angle = -Math.PI / 2 + index * ((Math.PI * 2) / 3);
      const x = center.x + Math.cos(angle) * (radius + 42);
      const y = center.y + Math.sin(angle) * (radius + 42);
      const endX = center.x + Math.cos(angle) * radius;
      const endY = center.y + Math.sin(angle) * radius;
      return `<line x1="${center.x}" y1="${center.y}" x2="${endX}" y2="${endY}" class="axis-line"></line>
        <text x="${x}" y="${y}" text-anchor="middle" dominant-baseline="middle">${label}</text>`;
    })
    .join("");

  const textRankPoints = polygonPointObjects(scoreValues(state.scores.textrank), center, radius);
  const qwenPoints = polygonPointObjects(scoreValues(state.scores.qwen), center, radius);
  const textRank = pointString(textRankPoints);
  const qwen = pointString(qwenPoints);

  svg.innerHTML = `
    ${grid}
    <polygon points="${textRank}" class="series-fill fill-textrank" style="fill: rgba(47, 112, 166, 0.22);"></polygon>
    <polygon points="${qwen}" class="series-fill fill-qwen" style="fill: rgba(193, 95, 63, 0.28);"></polygon>
    ${gridOverlay}
    ${axis}
    <polyline points="${qwen} ${qwenPoints[0].x},${qwenPoints[0].y}" class="series-line line-qwen"></polyline>
    ${dataPoints(qwenPoints, "point-qwen", 5)}
    <polyline points="${textRank} ${textRankPoints[0].x},${textRankPoints[0].y}" class="series-line line-textrank"></polyline>
    ${dataPoints(textRankPoints, "point-textrank", 6.5)}
    <circle cx="${center.x}" cy="${center.y}" r="3" class="center-dot"></circle>
    <g class="legend">
      <circle cx="178" cy="420" r="6" class="legend-textrank"></circle>
      <text x="192" y="421">TextRank</text>
      <circle cx="300" cy="420" r="6" class="legend-qwen"></circle>
      <text x="314" y="421">Qwen LoRA</text>
    </g>`;
  popElement(svg);
}

function resetSettings() {
  $("sentenceCount").value = 4;
  $("textrankRatio").value = 0.25;
  $("modelPath").value = "models/Qwen2.5-7B-Instruct";
  $("adapterPath").value = ".";
  $("maxTokens").value = 220;
  $("temperature").value = 0.2;
  $("topP").value = 0.9;
  $("repetitionPenalty").value = 1.05;
  syncRangeValue("textrankRatio", "textrankRatioValue", 2);
  syncRangeValue("temperature", "temperatureValue", 2);
  syncRangeValue("topP", "topPValue", 2);
  syncRangeValue("repetitionPenalty", "repetitionPenaltyValue", 2);
  document.querySelectorAll(".settings-card").forEach(popElement);
}

function loadDemoContent() {
  $("sourceText").value = demoText;
  $("referenceText").value = demoReference;
  $("textrankResult").textContent = "等待运行";
  $("qwenResult").textContent = "等待运行";
  state.scores.textrank = null;
  state.scores.qwen = null;
  updateCounts();
  renderMetrics();
  clearRougeChart();
  [$("sourceText"), $("referenceText"), $("textrankResult"), $("qwenResult")].forEach(popElement);
}

function resetInitialContent() {
  $("sourceText").value = "";
  $("referenceText").value = "";
  $("textrankResult").textContent = "等待运行";
  $("qwenResult").textContent = "等待运行";
  state.scores.textrank = null;
  state.scores.qwen = null;
  updateCounts();
  renderMetrics();
  clearRougeChart();
}

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => switchPage(tab.dataset.page)));
$("loadDemo").addEventListener("click", loadDemoContent);
$("sourceText").addEventListener("input", updateCounts);
$("runTextrank").addEventListener("click", runTextrank);
$("runQwen").addEventListener("click", runQwen);
$("runAll").addEventListener("click", runAll);
$("scoreBoth").addEventListener("click", scoreBoth);
$("resetSettings").addEventListener("click", resetSettings);
document.querySelectorAll("[data-score]").forEach((button) => {
  button.addEventListener("click", () => scoreSingle(button.dataset.score));
});

bindRange("textrankRatio", "textrankRatioValue", 2);
bindRange("temperature", "temperatureValue", 2);
bindRange("topP", "topPValue", 2);
bindRange("repetitionPenalty", "repetitionPenaltyValue", 2);
resetInitialContent();

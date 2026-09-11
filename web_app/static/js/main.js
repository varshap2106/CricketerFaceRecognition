/**
 * CricVision AI - Indian Cricket Team Face Identification
 * Interactive Frontend JavaScript Engine
 */

document.addEventListener("DOMContentLoaded", () => {
  // Global State
  let currentFile = null;
  let allPlayers = {};

  // DOM Elements - Navigation Tabs
  const navTabs = document.querySelectorAll(".nav-tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  // DOM Elements - Identifier
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const dropPrompt = document.getElementById("drop-zone-prompt");
  const previewContainer = document.getElementById("preview-container");
  const imagePreview = document.getElementById("image-preview");
  const btnClearFile = document.getElementById("btn-clear-file");
  const btnRunRecognition = document.getElementById("btn-run-recognition");
  const sampleChips = document.querySelectorAll(".sample-chip");

  // Results DOM
  const emptyState = document.getElementById("empty-state");
  const loadingState = document.getElementById("loading-state");
  const resultsContent = document.getElementById("results-content");
  const annotatedResultImg = document.getElementById("annotated-result-img");
  const cardPlayerName = document.getElementById("card-player-name");
  const cardPlayerNickname = document.getElementById("card-player-nickname");
  const cardJersey = document.getElementById("card-jersey");
  const cardConfidencePct = document.getElementById("card-confidence-pct");
  const cardConfBar = document.getElementById("card-conf-bar");
  const cardStatRole = document.getElementById("card-stat-role");
  const cardStatRuns = document.getElementById("card-stat-runs");
  const cardStatCenturies = document.getElementById("card-stat-centuries");
  const cardStatIcc = document.getElementById("card-stat-icc");
  const cardPlayerBio = document.getElementById("card-player-bio");
  const topRanksList = document.getElementById("top-ranks-list");
  const detectionCountBadge = document.getElementById("detection-count-badge");

  // DOM Elements - Roster
  const rosterGrid = document.getElementById("roster-grid");
  const rosterSearch = document.getElementById("roster-search");
  const filterChips = document.querySelectorAll(".filter-chip");

  // DOM Elements - Analytics & General
  const quickRetrainBtn = document.getElementById("quick-retrain-btn");
  const refreshMetricsBtn = document.getElementById("refresh-metrics-btn");
  const confusionMatrixImg = document.getElementById("confusion-matrix-img");

  /* ==========================================================
     1. TAB NAVIGATION
     ========================================================== */
  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetTab = tab.getAttribute("data-tab");
      navTabs.forEach((t) => t.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const activePanel = document.getElementById(targetTab);
      if (activePanel) activePanel.classList.add("active");
    });
  });

  /* ==========================================================
     2. FILE DRAG & DROP AND PREVIEW
     ========================================================== */
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (JPG, PNG, WEBP)");
      return;
    }
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      dropPrompt.classList.add("hidden");
      previewContainer.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
  }

  btnClearFile.addEventListener("click", (e) => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = "";
    imagePreview.src = "";
    dropPrompt.classList.remove("hidden");
    previewContainer.classList.add("hidden");
  });

  /* ==========================================================
     3. QUICK SAMPLES SELECTOR
     ========================================================== */
  sampleChips.forEach((chip) => {
    chip.addEventListener("click", async () => {
      const sampleSlug = chip.getAttribute("data-sample");
      const sampleUrl = `/static/samples/${sampleSlug}.jpg`;

      try {
        const resp = await fetch(sampleUrl);
        if (resp.ok) {
          const blob = await resp.blob();
          const sampleFile = new File([blob], `${sampleSlug}_real.jpg`, { type: "image/jpeg" });
          handleSelectedFile(sampleFile);
          triggerPrediction(sampleFile);
        } else {
          alert(`Loading sample for ${sampleSlug}...`);
        }
      } catch (err) {
        console.error("Error loading sample image:", err);
      }
    });
  });

  /* ==========================================================
     4. RUN PREDICTION PIPELINE
     ========================================================== */
  btnRunRecognition.addEventListener("click", () => {
    if (!currentFile) {
      alert("Please upload an image first or click one of the quick player demo samples.");
      return;
    }
    triggerPrediction(currentFile);
  });

  async function triggerPrediction(file) {
    showLoading();

    const formData = new FormData();
    formData.append("file", file);

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      const data = await resp.json();
      if (data.status === "success") {
        renderPredictionResults(data);
      } else {
        alert(`Prediction Notice: ${data.message || "Could not process image"}`);
        hideLoading();
      }
    } catch (err) {
      console.error(err);
      alert("Error contacting prediction server.");
      hideLoading();
    }
  }

  function renderPredictionResults(data) {
    hideLoading();
    emptyState.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    if (data.annotated_image_base64) {
      annotatedResultImg.src = data.annotated_image_base64;
    }

    detectionCountBadge.innerText = `${data.faces_detected} Face(s) Detected`;

    if (data.detections && data.detections.length > 0) {
      const topDet = data.detections[0];
      const profile = topDet.profile || {};
      const isUnknown = topDet.player_slug === "unknown";

      cardPlayerName.innerText = topDet.player_name || "Unknown Person";
      cardPlayerNickname.innerText = profile.nickname ? `"${profile.nickname}"` : (isUnknown ? "Non-indexed Person" : "");
      cardJersey.innerText = profile.jersey_no ? (profile.jersey_no === "-" ? "-" : `No. ${profile.jersey_no}`) : "Team India";

      const confPct = Math.round((topDet.confidence || 0) * 100);
      cardConfidencePct.innerText = `${confPct}%`;
      cardConfBar.style.width = `${confPct}%`;

      if (isUnknown) {
        cardConfBar.style.background = "linear-gradient(90deg, #ff4757, #ffa502)";
      } else {
        cardConfBar.style.background = "linear-gradient(90deg, var(--accent-blue), var(--accent-emerald))";
      }

      cardStatRole.innerText = profile.role || "Non-indexed";
      cardStatRuns.innerText = profile.runs ? `${profile.runs.toLocaleString()}+` : "N/A";
      cardStatCenturies.innerText = profile.centuries !== undefined ? profile.centuries : "N/A";
      cardStatIcc.innerText = profile.icc_rank || "N/A";
      cardPlayerBio.innerText = profile.bio || "Input does not match indexed Indian cricketers.";

      topRanksList.innerHTML = "";
      if (topDet.top_predictions && topDet.top_predictions.length > 0) {
        topDet.top_predictions.forEach((pred) => {
          const pct = Math.round(pred.confidence * 100);
          const item = document.createElement("div");
          item.className = "rank-item";
          item.innerHTML = `
            <span class="rank-name">${pred.name}</span>
            <div class="rank-meter-box">
              <div class="mini-bar"><div class="mini-fill" style="width: ${pct}%;"></div></div>
              <strong>${pct}%</strong>
            </div>
          `;
          topRanksList.appendChild(item);
        });
      }
    }
  }

  function showLoading() {
    emptyState.classList.add("hidden");
    resultsContent.classList.add("hidden");
    loadingState.classList.remove("hidden");
  }

  function hideLoading() {
    loadingState.classList.add("hidden");
  }

  /* ==========================================================
     5. TEAM ROSTER & SEARCH / FILTERING
     ========================================================== */
  async function loadTeamRoster() {
    try {
      const resp = await fetch("/api/players");
      const data = await resp.json();
      if (data.status === "success") {
        allPlayers = data.players;
        renderRosterCards(Object.values(allPlayers));
      }
    } catch (e) {
      console.error("Error loading roster:", e);
    }
  }

  function renderRosterCards(playersList) {
    if (!rosterGrid) return;
    rosterGrid.innerHTML = "";

    if (!playersList || playersList.length === 0) {
      rosterGrid.innerHTML = '<p style="color:#a4b0be;text-align:center;padding:2rem;grid-column:1/-1;">No players found.</p>';
      return;
    }

    playersList.forEach((player) => {
      const runs = (player.runs || 0).toLocaleString();
      const card = document.createElement("div");
      card.className = "roster-card";
      card.innerHTML = `
        <div class="roster-card-header">
          <div class="jersey-circle">${player.jersey_no || '#'}</div>
          <span class="role-badge">${player.role || 'Player'}</span>
        </div>
        <h3 class="roster-player-name">${player.name || 'Unknown'}</h3>
        <div class="roster-nickname">"${player.nickname || ''}"</div>
        <div class="roster-stats-row">
          <div class="roster-stat-col"><span>Runs</span><strong>${runs}</strong></div>
          <div class="roster-stat-col"><span>100s</span><strong>${player.centuries || 0}</strong></div>
          <div class="roster-stat-col"><span>Wickets</span><strong>${player.wickets || 0}</strong></div>
          <div class="roster-stat-col"><span>Matches</span><strong>${player.matches || 0}</strong></div>
        </div>
        <p class="roster-bio">${player.bio || ''}</p>
      `;
      rosterGrid.appendChild(card);
    });
  }

  if (rosterSearch) {
    rosterSearch.addEventListener("input", filterRoster);
  }

  filterChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      filterChips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      filterRoster();
    });
  });

  function filterRoster() {
    const query = rosterSearch ? rosterSearch.value.toLowerCase() : "";
    const activeChip = document.querySelector(".filter-chip.active");
    const activeFilter = activeChip ? activeChip.getAttribute("data-filter") : "all";

    const filtered = Object.values(allPlayers).filter((player) => {
      const matchesQuery =
        !query ||
        (player.name || "").toLowerCase().includes(query) ||
        (player.nickname || "").toLowerCase().includes(query) ||
        (player.role || "").toLowerCase().includes(query);

      const matchesRole =
        activeFilter === "all" || (player.role || "").toLowerCase().includes(activeFilter.toLowerCase());

      return matchesQuery && matchesRole;
    });

    renderRosterCards(filtered);
  }

  /* ==========================================================
     6. MODEL RETRAINING & ANALYTICS
     ========================================================== */
  if (quickRetrainBtn) {
    quickRetrainBtn.addEventListener("click", async () => {
      quickRetrainBtn.disabled = true;
      quickRetrainBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';

      try {
        const resp = await fetch("/api/retrain", { method: "POST" });
        const data = await resp.json();
        if (data.status === "success") {
          alert(`Model retrained successfully! Best model: ${data.benchmark.best_model_name} (Acc: ${Math.round(data.benchmark.best_accuracy * 100)}%)`);
          refreshMetrics();
        } else {
          alert(`Retraining Error: ${data.message}`);
        }
      } catch (err) {
        alert("Retrain request failed. Please check server logs.");
      } finally {
        quickRetrainBtn.disabled = false;
        quickRetrainBtn.innerHTML = '<i class="fa-solid fa-arrows-rotate"></i> Retrain';
      }
    });
  }

  if (refreshMetricsBtn) {
    refreshMetricsBtn.addEventListener("click", refreshMetrics);
  }

  function refreshMetrics() {
    if (confusionMatrixImg) {
      confusionMatrixImg.src = `/outputs/confusion_matrix.png?t=${new Date().getTime()}`;
    }
  }

  // Initial Data Load
  loadTeamRoster();
});
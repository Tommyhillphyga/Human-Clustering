const state = {
    currentJobId: null,
    pollHandle: null,
    snapshot: null,
};

const refs = {
    form: document.getElementById("uploadForm"),
    videoInput: document.getElementById("videoInput"),
    submitButton: document.getElementById("submitButton"),
    pauseButton: document.getElementById("pauseButton"),
    resumeButton: document.getElementById("resumeButton"),
    downloadButton: document.getElementById("downloadButton"),
    videoStream: document.getElementById("videoStream"),
    videoPlaceholder: document.getElementById("videoPlaceholder"),
    statusBadge: document.getElementById("statusBadge"),
    statusText: document.getElementById("statusText"),
    jobName: document.getElementById("jobName"),
    personCount: document.getElementById("personCount"),
    clusterCount: document.getElementById("clusterCount"),
    stationaryCount: document.getElementById("stationaryCount"),
    fpsValue: document.getElementById("fpsValue"),
    progressText: document.getElementById("progressText"),
    frameProgressLabel: document.getElementById("frameProgressLabel"),
    exportProgressLabel: document.getElementById("exportProgressLabel"),
    processingProgress: document.getElementById("processingProgress"),
    exportProgress: document.getElementById("exportProgress"),
    clusterList: document.getElementById("clusterList"),
    clusterMeta: document.getElementById("clusterMeta"),
    chart: document.getElementById("trendChart"),
};

refs.form.addEventListener("submit", handleSubmit);
refs.pauseButton.addEventListener("click", () => postJobAction("pause"));
refs.resumeButton.addEventListener("click", () => postJobAction("resume"));
refs.downloadButton.addEventListener("click", () => {
    if (state.snapshot?.download_url) {
        window.location.href = state.snapshot.download_url;
    }
});

drawChart({ people: [], clusters: [], stationary: [] });

async function handleSubmit(event) {
    event.preventDefault();
    if (!refs.videoInput.files.length) {
        setStatus("failed", "Select an MP4, AVI, or MOV file to start analysis.");
        return;
    }

    refs.submitButton.disabled = true;
    setStatus("queued", "Uploading video and preparing the processing job...");

    const formData = new FormData(refs.form);
    try {
        const response = await fetch("/api/jobs", {
            method: "POST",
            body: formData,
        });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || "Unable to create processing job.");
        }
        bootJob(payload);
    } catch (error) {
        refs.submitButton.disabled = false;
        setStatus("failed", error.message);
    }
}

function bootJob(snapshot) {
    state.currentJobId = snapshot.job_id;
    state.snapshot = snapshot;
    refs.videoPlaceholder.style.display = "none";
    refs.videoStream.style.display = "block";
    refs.videoStream.src = `${snapshot.stream_url}?ts=${Date.now()}`;
    refs.submitButton.disabled = false;
    updateSnapshot(snapshot);
    startPolling();
}

function startPolling() {
    if (state.pollHandle) {
        clearInterval(state.pollHandle);
    }
    fetchSnapshot();
    state.pollHandle = window.setInterval(fetchSnapshot, 1000);
}

async function fetchSnapshot() {
    if (!state.currentJobId) {
        return;
    }
    try {
        const response = await fetch(`/api/jobs/${state.currentJobId}`);
        const snapshot = await response.json();
        if (!response.ok) {
            throw new Error(snapshot.detail || "Failed to read job state.");
        }
        updateSnapshot(snapshot);
        if (["completed", "failed"].includes(snapshot.status) && state.pollHandle) {
            clearInterval(state.pollHandle);
            state.pollHandle = null;
        }
    } catch (error) {
        setStatus("failed", error.message);
    }
}

async function postJobAction(action) {
    if (!state.currentJobId) {
        return;
    }
    try {
        const response = await fetch(`/api/jobs/${state.currentJobId}/${action}`, {
            method: "POST",
        });
        const snapshot = await response.json();
        if (!response.ok) {
            throw new Error(snapshot.detail || `Unable to ${action} this job.`);
        }
        updateSnapshot(snapshot);
    } catch (error) {
        setStatus("failed", error.message);
    }
}

function updateSnapshot(snapshot) {
    state.snapshot = snapshot;
    const progress = Math.round((snapshot.progress || 0) * 100);
    const exportProgress = Math.round((snapshot.export_progress || 0) * 100);

    refs.jobName.textContent = snapshot.filename || "No active job";
    refs.personCount.textContent = snapshot.person_count;
    refs.clusterCount.textContent = snapshot.cluster_count;
    refs.stationaryCount.textContent = snapshot.stationary_count;
    refs.fpsValue.textContent = Number(snapshot.processing_fps || 0).toFixed(1);

    refs.progressText.textContent = `${progress}%`;
    refs.frameProgressLabel.textContent = `${snapshot.frame_index} / ${snapshot.total_frames || 0}`;
    refs.exportProgressLabel.textContent = `${exportProgress}%`;
    refs.processingProgress.style.width = `${progress}%`;
    refs.exportProgress.style.width = `${exportProgress}%`;

    updateClusters(snapshot.active_cluster_ids || []);
    updateButtons(snapshot.status, snapshot.download_url);
    setStatus(snapshot.status, buildStatusCopy(snapshot));
    drawChart(snapshot.history || { people: [], clusters: [], stationary: [] });
}

function updateClusters(clusterIds) {
    refs.clusterMeta.textContent = clusterIds.length ? `${clusterIds.length} active` : "None";
    refs.clusterList.innerHTML = "";

    if (!clusterIds.length) {
        refs.clusterList.innerHTML = '<p class="empty-note">Stationary groups will appear here once they satisfy the clustering rules.</p>';
        return;
    }

    clusterIds.forEach((clusterId) => {
        const pill = document.createElement("span");
        pill.className = "cluster-pill";
        pill.textContent = `Cluster ${clusterId}`;
        refs.clusterList.appendChild(pill);
    });
}

function updateButtons(status, downloadUrl) {
    refs.pauseButton.disabled = status !== "running";
    refs.resumeButton.disabled = status !== "paused";
    refs.downloadButton.disabled = !(status === "completed" && downloadUrl);
}

function setStatus(status, copy) {
    refs.statusBadge.textContent = capitalize(status);
    refs.statusBadge.className = `status-badge ${status}`;
    refs.statusText.textContent = copy;
}

function buildStatusCopy(snapshot) {
    if (snapshot.status === "running") {
        return `Processing frame ${snapshot.frame_index} of ${snapshot.total_frames || "?"} at ${Number(snapshot.processing_fps || 0).toFixed(1)} FPS.`;
    }
    if (snapshot.status === "paused") {
        return `Job paused at frame ${snapshot.frame_index}. Resume whenever you are ready.`;
    }
    if (snapshot.status === "completed") {
        return "Processing complete. The annotated MP4 is ready to download.";
    }
    if (snapshot.status === "failed") {
        return snapshot.error || "Processing failed.";
    }
    if (snapshot.status === "queued") {
        return "Job queued. The processing worker is warming up the detector and tracker.";
    }
    return "Waiting for a video job.";
}

function drawChart(history) {
    const canvas = refs.chart;
    const context = canvas.getContext("2d");
    const width = canvas.width;
    const height = canvas.height;
    const padding = 20;

    context.clearRect(0, 0, width, height);
    context.fillStyle = "rgba(255,255,255,0.02)";
    context.fillRect(0, 0, width, height);

    const series = [
        { values: history.people || [], color: "#6bd7ff", label: "People" },
        { values: history.stationary || [], color: "#ffb54a", label: "Stationary" },
        { values: history.clusters || [], color: "#ff5f57", label: "Clusters" },
    ];

    const maxLength = Math.max(...series.map((entry) => entry.values.length), 0);
    const flatValues = series.flatMap((entry) => entry.values);
    const maxValue = Math.max(...flatValues, 1);

    context.strokeStyle = "rgba(255,255,255,0.08)";
    context.lineWidth = 1;
    for (let index = 0; index < 4; index += 1) {
        const y = padding + ((height - padding * 2) / 3) * index;
        context.beginPath();
        context.moveTo(padding, y);
        context.lineTo(width - padding, y);
        context.stroke();
    }

    series.forEach((entry) => {
        if (!entry.values.length) {
            return;
        }
        context.beginPath();
        entry.values.forEach((value, index) => {
            const x = padding + ((width - padding * 2) * index) / Math.max(maxLength - 1, 1);
            const y = height - padding - ((height - padding * 2) * value) / maxValue;
            if (index === 0) {
                context.moveTo(x, y);
            } else {
                context.lineTo(x, y);
            }
        });
        context.strokeStyle = entry.color;
        context.lineWidth = 3;
        context.stroke();
    });

    const legend = [
        { text: "People", color: "#6bd7ff" },
        { text: "Stationary", color: "#ffb54a" },
        { text: "Clusters", color: "#ff5f57" },
    ];
    legend.forEach((item, index) => {
        const x = padding + index * 118;
        const y = 18;
        context.fillStyle = item.color;
        context.fillRect(x, y - 8, 12, 12);
        context.fillStyle = "#cbd7e4";
        context.font = '12px "Space Grotesk", sans-serif';
        context.fillText(item.text, x + 18, y + 2);
    });
}

function capitalize(value) {
    return (value || "idle").charAt(0).toUpperCase() + (value || "idle").slice(1);
}

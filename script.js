let datasetChart = null;

function activateTab(tabId, btn) {
  document.querySelectorAll('.tab-content').forEach(tc => tc.style.display = 'none');
  document.getElementById(tabId).style.display = "block";

  document.querySelectorAll('.tabs button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
}

// Load dataset prediction pie chart with percentages
async function loadDataset() {
  const details = document.getElementById('histogramDetail');
  details.innerHTML = "Loading predictions...";
  try {
    const resp = await fetch("http://127.0.0.1:5000/predict_dataset");
    const data = await resp.json();
    const labels = Object.keys(data.percentages);
    const percentages = Object.values(data.percentages);

    if(datasetChart) datasetChart.destroy();

    datasetChart = new Chart(document.getElementById("datasetChart"), {
      type: 'pie',
      data: {
        labels: labels,
        datasets: [{
          data: percentages,
          backgroundColor: [
            '#cd4d51ff','#744c31ff','#834865ff','#4bd4dbff',
            '#747074ff','#f2c150ff','#00a950'
          ],
          borderColor: '#ffffff6f',
          borderWidth: 1
        }]
      },
      options: {
        onClick: (evt, elems) => {
          if(elems.length > 0){
            const idx = elems[0].index;
            details.innerHTML = `<b>Emotion:</b> ${labels[idx]}<br><b>Percentage:</b> ${percentages[idx]}%`;
          }
        },
        responsive: true,
        plugins: {
          legend: {
            position: 'right'
          },
          tooltip: {
            callbacks: {
              label: (context) => `${context.label}: ${context.parsed}%`
            }
          }
        }
      }
    });
    details.innerHTML = "Click on any slice to view details.";
  } catch {
    details.innerHTML = "Failed to load dataset prediction.";
  }
}

// User input prediction form submission
document.getElementById("emotionForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const audioFile = document.getElementById("audioInput").files[0];
  const imageFile = document.getElementById("imageInput").files[0];
  const userResult = document.getElementById("userResult");

  if (!audioFile || !imageFile) {
    alert("Please upload audio and image files.");
    return;
  }

  userResult.textContent = "Predicting... please wait.";

  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("image", imageFile);

  try {
    const resp = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    if (!resp.ok) {
      userResult.textContent = "Prediction failed. Please try again.";
      return;
    }

    const data = await resp.json();

    let probList = "<ul class='probability-list'>";
    for (const [emotion, prob] of Object.entries(data.probability_distribution)) {
      probList += `<li><strong>${emotion}:</strong> ${prob}%</li>`;
    }
    probList += "</ul>";

    userResult.innerHTML = `
      <h3>${data.emotion} (${data.confidence_percent}%)</h3>
      <p><strong>Explanation:</strong> ${data.llm_explanation}</p>
      <h4>Probability Distribution:</h4>
      ${probList}
    `;
  } catch (err) {
    userResult.textContent = "Network error or backend not reachable.";
  }
});

// Initialize page view
activateTab('datasetTab', document.querySelector('.tabs button'));

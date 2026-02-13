// Handle News Prediction Form
document.getElementById('unified-form').addEventListener('submit', function (event) {
    event.preventDefault();

    const userInput = document.getElementById('user_input').value.trim();
    const resultDiv = document.getElementById('unified-result');
    resultDiv.innerHTML = "🔎 Analyzing news...";

    if (!userInput) {
        alert("Please enter some text or a URL!");
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ 'user_input': userInput })
    })
        .then(response => response.json())
        .then(data => {
            if (data.prediction) {
                const accuracy = data.accuracy || 0;
                const linksHtml = data.links && data.links.length > 0
                    ? `<br><strong>Sources:</strong><ul>${data.links.map(link => `<li><a href="${link[1]}" target="_blank">${link[0]}</a></li>`).join('')}</ul>`
                    : '<br><em>No credible sources found.</em>';

                const accuracyBar = `
                <div class="accuracy-bar-wrapper" style="margin-top:10px;">
                    <div class="accuracy-bar" style="width: ${accuracy}%; background-color: ${getAccuracyColor(accuracy)};">
                        ${accuracy}%
                    </div>
                </div>`;

                resultDiv.innerHTML = `
                <strong>Prediction:</strong> ${data.prediction}
                ${accuracyBar}
                ${linksHtml}`;
            } else {
                resultDiv.innerHTML = `<strong>Error:</strong> ${data.error || 'No result'}`;
            }
        })
        .catch(error => {
            resultDiv.innerText = 'Error: ' + error.message;
        });
});

// Handle Image Upload Form
document.getElementById('image-form').addEventListener('submit', function (event) {
    event.preventDefault();

    const imageFile = document.getElementById('image_file').files[0];
    const resultDiv = document.getElementById('image-result');
    resultDiv.innerHTML = "🔎 Analyzing image...";

    if (!imageFile) {
        alert("Please upload an image!");
        return;
    }

    const formData = new FormData();
    formData.append('image_file', imageFile);

    fetch('/predict_image', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.prediction) {
                const accuracy = data.accuracy || 0;
                const accuracyBar = `
                <div class="accuracy-bar-wrapper" style="margin-top:10px;">
                    <div class="accuracy-bar" style="width: ${accuracy}%; background-color: ${getAccuracyColor(accuracy)};">
                        ${accuracy}%
                    </div>
                </div>`;

                let imageHTML = `<strong>Prediction:</strong> ${data.prediction}${accuracyBar}`;
                imageHTML += `<br><strong>Uploaded Image:</strong> <a href="${data.image_path}" target="_blank">View</a>`;
                resultDiv.innerHTML = imageHTML;
            } else {
                resultDiv.innerHTML = `<strong>Error:</strong> ${data.error || 'No result'}`;
            }
        })
        .catch(error => {
            resultDiv.innerText = 'Error: ' + error.message;
        });
});

// Utility function to get color based on accuracy
function getAccuracyColor(accuracy) {
    if (accuracy >= 85) return '#27ae60';     // green
    if (accuracy >= 60) return '#f39c12';     // orange
    return '#c0392b';                         // red
}
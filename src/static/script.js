const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const uploadContent = document.getElementById('upload-content');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');

const loading = document.getElementById('loading');
const resultSection = document.getElementById('result-section');
const emotionEmoji = document.getElementById('emotion-emoji');
const emotionName = document.getElementById('emotion-name');
const confidenceFill = document.getElementById('confidence-fill');
const confidenceText = document.getElementById('confidence-text');
const resetBtn = document.getElementById('reset-btn');

const emotionToEmoji = {
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠',
    'Surprise': '😲',
    'Fear': '😨',
    'Disgust': '🤢',
    'Neutral': '😐',
    'No face detected': '❓'
};

// Handle Drag and Drop
dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});

dropzone.addEventListener('click', () => {
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        handleFile(fileInput.files[0]);
    }
});

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file.');
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadContent.classList.add('hidden');
        previewContainer.classList.remove('hidden');
    };
    reader.readAsDataURL(file);

    // Send to API
    uploadAndPredict(file);
}

async function uploadAndPredict(file) {
    // UI Reset
    resultSection.classList.add('hidden');
    loading.classList.remove('hidden');
    dropzone.style.pointerEvents = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            displayResult(data.emotion, data.confidence);
        } else {
            alert('Error: ' + data.error);
            resetUI();
        }
    } catch (error) {
        alert('Network error. Is the server running?');
        resetUI();
    } finally {
        loading.classList.add('hidden');
        dropzone.style.pointerEvents = 'auto';
    }
}

function displayResult(emotion, confidence) {
    resultSection.classList.remove('hidden');
    
    emotionName.textContent = emotion;
    emotionEmoji.textContent = emotionToEmoji[emotion] || '🤔';
    
    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);
    
    confidenceText.textContent = `${confidence.toFixed(1)}% Confidence`;
}

function resetUI() {
    fileInput.value = '';
    uploadContent.classList.remove('hidden');
    previewContainer.classList.add('hidden');
    resultSection.classList.add('hidden');
    confidenceFill.style.width = '0%';
}

resetBtn.addEventListener('click', resetUI);

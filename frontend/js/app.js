document.addEventListener("DOMContentLoaded", function() {
  const uploadArea = document.getElementById('upload-area');
  const fileInput = document.getElementById('imageFile');
  const selectButton = document.getElementById('selectButton');
  const progressDiv = document.getElementById('progress');
  const resultDiv = document.getElementById('result');
  const downloadButton = document.getElementById('downloadButton');

  // When the select button is clicked, trigger file input
  selectButton.addEventListener('click', () => fileInput.click());

  // Handle drag and drop
  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('hover');
  });
  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('hover');
  });
  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('hover');
    const files = e.dataTransfer.files;
    if (files.length) {
      fileInput.files = files;
      processImage();
    }
  });

  // Handle file selection
  fileInput.addEventListener('change', processImage);

  async function processImage() {
    const file = fileInput.files[0];
    if (!file) {
      resultDiv.innerHTML = '<div class="alert alert-danger">No file selected.</div>';
      return;
    }
    
    progressDiv.style.display = 'block';
    resultDiv.innerHTML = '';
    downloadButton.style.display = 'none';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:5100/predict', {
        method: 'POST',
        body: formData
      });
      const data = await response.json();
      progressDiv.style.display = 'none';
      
      if (data.error) {
        resultDiv.innerHTML = `<div class="alert alert-danger">${data.error}</div>`;
        return;
      }
      
      // Create image element to display the annotated image
      const img = document.createElement('img');
      img.src = 'data:image/jpeg;base64,' + data.result_image;
      img.className = "img-fluid";
      resultDiv.innerHTML = '';
      resultDiv.appendChild(img);
      
      // Display object counts in a table
      let countsHTML = `<h4>Object Counts:</h4><table class="table table-bordered"><thead><tr><th>Class</th><th>Count</th></tr></thead><tbody>`;
      for (let key in data.object_counts) {
        countsHTML += `<tr><td>${key}</td><td>${data.object_counts[key]}</td></tr>`;
      }
      countsHTML += `</tbody></table>`;
      resultDiv.innerHTML += countsHTML;
      
      // Display summary text
      if (data.summary) {
        resultDiv.innerHTML += `<h5>${data.summary}</h5>`;
      }
      
      // Setup download button
      downloadButton.style.display = 'block';
      downloadButton.onclick = () => {
        downloadImage(data.result_image, 'annotated_image.jpg');
      };
      
    } catch (error) {
      console.error(error);
      progressDiv.style.display = 'none';
      resultDiv.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
    }
  }

  function downloadImage(base64Data, fileName) {
    const link = document.createElement('a');
    link.href = 'data:image/jpeg;base64,' + base64Data;
    link.download = fileName;
    link.click();
  }
});
async function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];

    if (!file) {
        alert("Please select an image");
        return;
    }

    const loader = document.getElementById("loader");
    loader.style.display = "block";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error("Server error");
        }

        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);

        const resultImage = document.getElementById("resultImage");
        resultImage.src = imageUrl;

        document.getElementById("downloadBtn").style.display = "inline-block";

    } catch (error) {
        alert("Error: " + error.message);
    }

    loader.style.display = "none";
}

function downloadImage() {
    const img = document.getElementById("resultImage");

    const link = document.createElement("a");
    link.href = img.src;
    link.download = "detected.png";
    link.click();
}
import React, { useState } from "react";

// CIFAR-10 Class Labels
const CIFAR10_LABELS = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck"
];

const UploadForm = () => {
  const [imageFile, setImageFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    setImageFile(e.target.files[0]);
  };

  const convertToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result.split(",")[1]); // Only the Base64 string
      reader.onerror = (error) => reject(error);
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!imageFile) {
      alert("Please select an image to upload");
      return;
    }

    try {
      setLoading(true);
      const base64Image = await convertToBase64(imageFile);

      const response = await fetch("https://fmbqf81mde.execute-api.us-east-1.amazonaws.com/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: base64Image, // Sending Base64 string
        }),
      });

      const result = await response.json();
      const label = CIFAR10_LABELS[result.prediction]; // Convert index to label
      console.log("Result:", result);

      alert(`Prediction Result: ${label} (Class Index: ${result.prediction})`);
    } catch (error) {
      console.error("Error during submission:", error);
      alert("Error uploading the image.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>Upload Image for Prediction</h2>
      <input type="file" accept="image/*" onChange={handleFileChange} />
      <button type="submit" disabled={loading}>
        {loading ? "Uploading..." : "Submit"}
      </button>
    </form>
  );
};

export default UploadForm;

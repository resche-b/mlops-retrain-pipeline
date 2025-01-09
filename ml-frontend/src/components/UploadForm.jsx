import React, { useState } from "react";

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
      reader.onload = () => resolve(reader.result.split(",")[1]); 
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

      const response = await fetch("YOUR_API_ENDPOINT_URL", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          image: base64Image, 
        }),
      });

      const result = await response.json();
      console.log("Result:", result);
      alert(`Prediction Result: ${result.prediction}`);
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

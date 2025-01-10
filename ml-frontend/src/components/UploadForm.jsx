import React, { useState } from "react";

const UploadForm = () => {
  const [imageFile, setImageFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [predictedClass, setPredictedClass] = useState(null);

  // CIFAR-10 possible classes
  const possibleClasses = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
  ];

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

      const response = await fetch("https://fmbqf81mde.execute-api.us-east-1.amazonaws.com/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: base64Image }),
      });

      if (!response.ok) {
        throw new Error(`HTTP Error: ${response.status}`);
      }

      const result = await response.json();
      const classIndex = result.prediction; 
      const className = possibleClasses[classIndex]; 
      setPredictedClass(className); 
    } catch (error) {
      console.error("Error during submission:", error);
      alert(`Error uploading the image: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.container}>
      <form onSubmit={handleSubmit} style={styles.form}>
        <h2 style={styles.heading}>Upload Image for Prediction</h2>
        <input type="file" accept="image/*" onChange={handleFileChange} style={styles.input} />
        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? "Uploading..." : "Submit"}
        </button>

        {predictedClass && (
          <div style={{ marginTop: "20px" }}>
            <h3 style={styles.result}>Prediction Result: {predictedClass}</h3>
          </div>
        )}

        <div style={{ marginTop: "20px" }}>
          <h4 style={styles.subHeading}>Possible Classes:</h4>
          <div style={styles.classesGrid}>
            {possibleClasses.map((cls, index) => (
              <span key={index} style={styles.classItem}>
                {cls}
              </span>
            ))}
          </div>
        </div>
      </form>
    </div>
  );
};

const styles = {
  container: {
    backgroundColor: "#121212",
    minHeight: "60vh",
    display: "flex",
    justifyContent: "center",
    alignItems: "center",
  },
  form: {
    backgroundColor: "#1e1e1e",
    padding: "30px",
    borderRadius: "10px",
    boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.5)",
    maxWidth: "500px",
    width: "100%",
  },
  heading: {
    color: "#ffffff",
    textAlign: "center",
    marginBottom: "20px",
  },
  input: {
    display: "block",
    width: "100%",
    padding: "10px",
    marginBottom: "20px",
    borderRadius: "5px",
    border: "1px solid #666",
    backgroundColor: "#2a2a2a",
    color: "#fff",
  },
  button: {
    backgroundColor: "#6200ea",
    color: "#fff",
    border: "none",
    padding: "10px 20px",
    borderRadius: "5px",
    cursor: "pointer",
    width: "100%",
  },
  result: {
    color: "#00e676",
    textAlign: "center",
  },
  subHeading: {
    color: "#ffffff",
    marginBottom: "10px",
  },
  classesGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
    gap: "10px",
    justifyItems: "center", 
  },
  classItem: {
    padding: "10px",
    textAlign: "center",
    borderRadius: "8px",
    backgroundColor: "#333",
    color: "#ffffff",
    border: "1px solid #555",
    width: "80%",
    maxWidth: "200px",
  },
};

export default UploadForm;

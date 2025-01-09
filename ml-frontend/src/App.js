import React from "react";
import UploadForm from "./components/UploadForm";
import "./styles.css";

function App() {
  return (
    <div className="App">
      <h1>Welcome to CIFAR-10 Classifier</h1>
      <UploadForm />
    </div>
  );
}

export default App;

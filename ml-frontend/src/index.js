import React from "react";
import ReactDOM from "react-dom/client";  // Updated import
import App from "./App";

const rootElement = document.getElementById("root");
const root = ReactDOM.createRoot(rootElement);  // Create a root for React 18+
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

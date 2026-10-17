import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";

// apply persisted theme / colour convention before the first paint
try {
  document.documentElement.dataset.theme = localStorage.getItem("defaco.theme") || "dark";
  document.documentElement.dataset.updown = localStorage.getItem("defaco.updown") || "red";
} catch {
  document.documentElement.dataset.theme = "dark";
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

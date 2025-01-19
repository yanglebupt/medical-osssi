import React from "react";
import ReactDOM from "react-dom/client";
import { SpeedInsights } from "@vercel/speed-insights/react";
import App from "./App";


ReactDOM.createRoot(document.getElementById("app")!).render(
  <React.StrictMode>
    <App />
    <SpeedInsights />
  </React.StrictMode>
);

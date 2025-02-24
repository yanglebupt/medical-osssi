import React from "react";
import ReactDOM from "react-dom/client";
import { SpeedInsights } from "@vercel/speed-insights/react";
import App from "./App";

console.log(import.meta.env.MODE != "production")

ReactDOM.createRoot(document.getElementById("app")!).render(
  <React.StrictMode>
    <App />
    {import.meta.env.MODE != "production" && <SpeedInsights />}
  </React.StrictMode>
);

import React from "react";
import ReactDOM from "react-dom/client";
import { SpeedInsights } from "@vercel/speed-insights/react";
import { isMobile } from "react-device-detect";
import App from "./app";

(function loadGlobalCss() {
  import(isMobile ? "./global.h5.css" : "./global.css");
})();

ReactDOM.createRoot(document.getElementById("app")!).render(
  <React.StrictMode>
    <App />
    <SpeedInsights />
  </React.StrictMode>
);

import { defineConfig, loadEnv } from "vite";
import { viteStaticCopy } from "vite-plugin-static-copy";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default ({mode}) => {
  const env = loadEnv(mode, process.cwd())
  return defineConfig({
    base: env["VITE_BASE"],
    assetsInclude: ["**/*.xlsx", "**/*.onnx"],
    plugins: [
      react(),
      viteStaticCopy({
        targets: [
          {
            src: "node_modules/onnxruntime-web/dist/*.wasm",
            dest: ".",
          },
        ],
      }),
    ],
  });
}

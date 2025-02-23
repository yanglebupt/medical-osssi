import { defineConfig, loadEnv } from "vite";
import viteCompression from 'vite-plugin-compression'
import { viteStaticCopy } from "vite-plugin-static-copy";
import generateVercelJson from "./generate-vercel-json";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default ({mode}) => {
  const env = loadEnv(mode, process.cwd())
  const plugins = [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/wasm-opts/*.wasm",
          dest: ".",
        },
      ],
    }),
    viteCompression({
      filter: /.(js|mjs|json|css|html|wasm|onnx)$/i,
      threshold: 100*1024, // 100 KB
      deleteOriginFile: true
    }),
  ]

  if (mode=="pre-production") plugins.push(generateVercelJson())

  return defineConfig({
    base: env["VITE_BASE"],
    assetsInclude: ["**/*.xlsx", "**/*.onnx"],
    plugins,
  });
}

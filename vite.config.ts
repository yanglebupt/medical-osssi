import { defineConfig, loadEnv, Plugin, PluginOption } from "vite";
import viteCompression from 'vite-plugin-compression'
import { viteStaticCopy } from "vite-plugin-static-copy";
import { generateVercelJson } from './generate-vercel-json'
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default ({mode}) => {
  const env = loadEnv(mode, process.cwd())
  const plugins: Array<Plugin|PluginOption> = [
    react(),
    viteStaticCopy({
      targets: [
        {
          src: `node_modules/onnxruntime-web/dist/${mode == "production" ? "wasm-opts/" : ""}*.wasm`,
          dest: ".",
        },
      ],
    })
  ]

  if (mode == "production" || mode == "pre-production")
    plugins.push(viteCompression({
      filter: /.(js|mjs|json|css|html|wasm|onnx)$/i,
      threshold: 100*1024, // 100 KB
      deleteOriginFile: true,
      success: mode == "pre-production" ? () => generateVercelJson("dist") : undefined
    }))

  return defineConfig({
    base: env["VITE_BASE"],
    assetsInclude: ["**/*.xlsx", "**/*.onnx"],
    plugins,
  });
}

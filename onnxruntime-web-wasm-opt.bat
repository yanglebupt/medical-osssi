@echo off

md .\node_modules\onnxruntime-web\dist\wasm-opts

@REM C:/Users/27356/Downloads/binaryen-version_122-x86_64-windows/binaryen-version_122/bin/wasm-opt.exe ./node_modules/onnxruntime-web/dist/ort-wasm.wasm -Oz -o ./node_modules/onnxruntime-web/dist/wasm-opts/ort-wasm.wasm
@REM C:/Users/27356/Downloads/binaryen-version_122-x86_64-windows/binaryen-version_122/bin/wasm-opt.exe ./node_modules/onnxruntime-web/dist/ort-wasm-simd.wasm -Oz -o ./node_modules/onnxruntime-web/dist/wasm-opts/ort-wasm-simd.wasm --enable-simd
@REM C:/Users/27356/Downloads/binaryen-version_122-x86_64-windows/binaryen-version_122/bin/wasm-opt.exe ./node_modules/onnxruntime-web/dist/ort-wasm-threaded.wasm -Oz -o ./node_modules/onnxruntime-web/dist/wasm-opts/ort-wasm-threaded.wasm --enable-threads --enable-bulk-memory-opt --enable-bulk-memory
@REM C:/Users/27356/Downloads/binaryen-version_122-x86_64-windows/binaryen-version_122/bin/wasm-opt.exe ./node_modules/onnxruntime-web/dist/ort-training-wasm-simd.wasm -Oz -o ./node_modules/onnxruntime-web/dist/wasm-opts/ort-training-wasm-simd.wasm --enable-simd

@REM C:/Users/27356/Downloads/binaryen-version_122-x86_64-windows/binaryen-version_122/bin/wasm-opt.exe ./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.wasm -Oz -o ./node_modules/onnxruntime-web/dist/wasm-opts/ort-wasm-simd-threaded.wasm --enable-simd --enable-threads --enable-bulk-memory-opt --enable-bulk-memory

@REM C:/Users/27356/Downloads/binaryen-version_122-x86_64-windows/binaryen-version_122/bin/wasm-opt.exe ./node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.jsep.wasm -Oz -o ./node_modules/onnxruntime-web/dist/wasm-opts/ort-wasm-simd-threaded.jsep.wasm --enable-simd --enable-threads --enable-bulk-memory-opt --enable-bulk-memory
@REM C:/Users/27356/Downloads/binaryen-version_122-x86_64-windows/binaryen-version_122/bin/wasm-opt.exe ./node_modules/onnxruntime-web/dist/ort-wasm-simd.jsep.wasm -Oz -o ./node_modules/onnxruntime-web/dist/wasm-opts/ort-wasm-simd.jsep.wasm --enable-simd
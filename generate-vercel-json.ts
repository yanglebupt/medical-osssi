import { Plugin } from 'vite'
import { readdirSync, statSync, createWriteStream } from "fs"
import { join, extname } from "path"

function findGzFiles(directory: string) {
  const gzFiles: string[] = [];

  function traverse(dir: string) {
    const files = readdirSync(dir);

    files.forEach(file => {
      const filePath = join(dir, file);
      const fileStat = statSync(filePath);

      if (fileStat.isDirectory()) {
        traverse(filePath);
      } else if (extname(file).toLowerCase() === '.gz') {
        gzFiles.push(filePath);
      }
    });
  }

  traverse(directory);
  return gzFiles;
}

function writeJSONFile(jsonData: any, saveFilename: string) {
	const writeStream = createWriteStream(saveFilename, {
		flags: 'w',
		mode: 0o666,
		encoding: 'utf8'
	});
	writeStream.write(JSON.stringify(jsonData, null, 2), 'utf8');
	writeStream.close()
}

const ContentTypeMap = {
	js: "application/javascript",
	css: "text/css",
	html: "text/html",
	onnx: "application/octet-stream",
	wasm: "application/wasm",
}
const MustGZExs = ["wasm", "onnx"]

export default () => {
	let outDir;
	return {
		name: 'generate-vercel-json',
		enforce: 'post',
		apply: 'build',
		configResolved(config) {
			outDir = config.build.outDir
		},
		buildEnd() {
			const vercelJson: { rewrites: Array<{source: string, destination: string}>, 
													headers: Array<{source: string, headers: Array<{key: string, value: string}>}>
												} = {
				rewrites: [],
				headers: [],
			}
			MustGZExs.forEach((ex)=>{
				vercelJson.rewrites.push({
					source: `/(.*)\\.${ex}`,
					destination: `/$1\\.${ex}.gz`
				})
				vercelJson.headers.push({
					source: `/(.*)\\.${ex}`,
					headers: [
						{
							key: "Content-Type",
							value: ContentTypeMap[ex]
						},
						{
							key: "Content-Encoding",
							value: "gzip"
						},
						{
							key: "vary",
							value: "Accept-Encoding"
						}
					]
				})
			})
			findGzFiles(outDir).forEach(gzFile => {
				const relGZFile = gzFile.split('.gz')[0].split(outDir)[1].replaceAll("\\", "/")
				const ext = extname(relGZFile).toLowerCase().slice(1)
				if (!(ext=="onnx" || ext=="wasm")) {
					vercelJson.rewrites.push({
						source: relGZFile,
						destination: relGZFile + ".gz"
					})
					vercelJson.headers.push({
						source: relGZFile,
						headers: [
							{
								key: "Content-Type",
								value: ContentTypeMap[ext]
							},
							{
								key: "Content-Encoding",
								value: "gzip"
							},
							{
								key: "vary",
								value: "Accept-Encoding"
							}
						]
					})
				}
			})
			writeJSONFile(vercelJson, join(__dirname, "vercel.json"))
		},
	} as Plugin
}
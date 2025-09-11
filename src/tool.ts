import * as ort from "onnxruntime-web";
import { string_headers } from "./app/headers";
ort.env.wasm.wasmPaths = import.meta.env.VITE_BASE

export function range(start:number, end:number) {
  const array: number[] = [];
  for (let i = start; i < end; i++) {
    array.push(i);
  }
  return array;
}

export async function requestBuffer(filename: string) {
  const res = await fetch(filename);
  return await res.arrayBuffer();
}

export function tensor2array(tensor: ort.Tensor) {
  const [_, feas] = tensor.dims;
  const array = [];
  for (let i = 0; i < tensor.data.length; i += feas) {
    const res = [];
    for (let j = 0; j < feas; j++) {
      res.push(tensor.data[i + j]);
    }
    array.push(res);
  }
  return array;
}

export const method_mapping: Record<string, string> = {
  "xgb": "XGBoost"
}
const cvs = 10
export const total_calc_count = Object.keys(method_mapping).length * cvs
const modelpath_mapping = import.meta.glob("./models/**/*.onnx", {
  eager: true,
  import: "default",
}) as Record<string, string>;

export function is_empty(form:Record<string, number>, h:string) {
  const v = form[h]
  return v === undefined || (string_headers.includes(h) ? v === "" :  (v === -1 || isNaN(v)))
}

export async function predict(
  datas: Array<Record<string, number>>,
  usedHeaders: string[],
  name: string,
  dims: [number, number],
  progress?: (message:string, progress: number)=>void
){
  const [feature_nums, features] = dims;
  const feature_list = datas.map((record: any) => {
    return usedHeaders.map((k) => is_empty(record, k) ? undefined : record[k]);
  });
  const featuresF32 = new Float32Array(feature_list.flat());
  let calc_count = 0

  return Promise.all(Object.keys(method_mapping).map((method) => 
    new Promise<{method:string, probs_list: Array<Array<number>>}>(async (resolve, reject)=>{
      const probs_list = []
      try {
        for (let cv_index = 0; cv_index < cvs; cv_index++) {
          const model_path = modelpath_mapping[`./models/${method}/${name}-cv${cv_index}.onnx`]
          const session = await ort.InferenceSession.create(model_path);
          const model_ipt = {
            [session.inputNames[0]]: new ort.Tensor("float32", featuresF32, [
              feature_nums,
              features,
            ]),
          };
          const results = await session.run(model_ipt);
          const probs = tensor2array(results[session.outputNames[1]]).map(
            ([_, p1]) => p1
          );
          probs_list.push(probs as number[])
          calc_count += 1
          progress && progress(model_path.split("/").pop()!, calc_count/total_calc_count)
        }
      } catch (error) {
        reject(error)
      }
      resolve({method, probs_list})
  })))
}

export function mean_std(array: Array<number>, fractionDigits=3): [string, string]{
  const sum = (x:number,y:number)=> x+y;
  const round = (x:number)=>x.toFixed(fractionDigits).padEnd(fractionDigits+2, "0")
  const mean = array.reduce(sum) / array.length
  const std = Math.sqrt(array.map(x=>Math.pow(x-mean, 2)).reduce(sum) / array.length)
  return [round(mean), round(std)]
}

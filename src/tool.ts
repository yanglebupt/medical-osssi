import * as ort from "onnxruntime-web";

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

export function convertDictNumber(datas: Array<Record<string, string>>) {
  return datas.map((data) => {
    const newData: Record<string, number> = {};
    Object.keys(data).forEach((k) => (newData[k] = parseFloat(data[k])));
    return newData;
  });
}

export async function predict(
  datas: Array<Record<string, number>>,
  usedHeaders: string[],
  path: string,
  dims: [number, number]
) {
  const [feature_nums, features] = dims;
  const feature_list = datas.map((record: any) => {
    return usedHeaders.map((k) => {
      if (!(k in record)) throw new Error(`${k} header is not in input datas`);
      return record[k];
    });
  });
  const featuresF32 = new Float32Array(feature_list.flat());
  const session = await ort.InferenceSession.create(path);
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
  return probs;
}

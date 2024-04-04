import * as XLSX from "xlsx";
import { label_name, usedHeaders_list } from "./headers";
import { predict, requestBuffer } from "./tool";
import filepath from "./data0130/0126validation2.xlsx";

const excelBuffer = await requestBuffer(filepath);
const workbook = XLSX.read(excelBuffer, {
  type: "buffer",
});
const worksheetJson = XLSX.utils.sheet_to_json(workbook.Sheets["Sheet1"]);

usedHeaders_list.slice(0, 1).forEach(async ([name, usedHeaders, _, path]) => {
  console.log(name);
  const probs = await predict(worksheetJson as any, usedHeaders, path, [
    worksheetJson.length,
    usedHeaders.length,
  ]);
  const labels = worksheetJson.map((record: any) => record[label_name]);
  console.log(labels);
  console.log(probs);
});

/*================= 未筛选前 ================= */
// 术前变量
const pre_headers = [
  "sex",
  "time.surg",
  "age",
  "height",
  "weight",
  "smoke",
  "hp",
  "dm",
  "chd",
  "arrhy",
  "copd",
  "pad",
  "rf",
  "chemo",
  "radio",
  "pn",
  "plt.pre",
  "hb.pre",
  "alb.pre",
  "tbil.pre",
  "wbc.pre",
  "alt.pre",
  "scr.pre",
  "asa",
  "class.surg.t1",
];
// 术中变量
const mid_headers = ["plasma", "rbc", "stoma", "bleed", "aa"];
// 术后变量
const post_headers = [
  "plt.post",
  "hb.post",
  "alb.post",
  "tbil.post",
  "wbc.post",
  "alt.post",
  "scr.post",
  "icu",
  "los.icu",
];

/*================= 筛选S ================= */
const pre_headers_S = [
  "time.surg",
  "weight",
  "copd",
  "pad",
  "radio",
  "class.surg.t1",
  "pn",
  "alt.pre",
];

const pre_mid_headers_S = [
  "time.surg",
  "weight",
  "copd",
  "pad",
  "class.surg.t1",
  "pn",
  "alt.pre",
  "bleed",
  "plasma",
  "rbc",
  "stoma",
  "aa",
];

const pre_mid_post_headers_S = [
  "time.surg",
  "weight",
  "copd",
  "pad",
  "rf",
  "class.surg.t1",
  "pn",
  "alb.pre",
  "tbil.pre",
  "rbc",
  "stoma",
  "aa",
  "los.icu",
  "alb.post",
  "hb.post",
  "plt.pre",
  "plt.post",
  "tbil.post",
  "scr.post",
  "wbc.post",
];

/*================= 筛选A ================= */
const pre_headers_A = [
  "time.surg",
  "weight",
  "copd",
  "pad",
  "radio",
  "class.surg.t1",
  "pn",
];

const pre_mid_headers_A = [
  "weight",
  "copd",
  "pad",
  "class.surg.t1",
  "pn",
  "bleed",
  "plasma",
  "rbc",
  "stoma",
  "aa",
];

const pre_mid_post_headers_A = [
  "weight",
  "copd",
  "pad",
  "class.surg.t1",
  "pn",
  "alb.pre",
  "rbc",
  "stoma",
  "aa",
  "los.icu",
  "alb.post",
  "hb.post",
  "plt.pre",
  "plt.post",
  "tbil.post",
  "scr.post",
  "wbc.post",
];

const label_name = "ssi.bin";
const usedHeaders_list: Array<[string, string[], number, string]> = [
  ["1", pre_headers, 1, "/models/1.onnx"],
  ["1S", pre_headers_S, 4, "/models/1S.onnx"],
  ["1A", pre_headers_A, 7, "/models/1A.onnx"],
  ["2", pre_headers.concat(mid_headers), 2, "/models/2.onnx"],
  ["2S", pre_mid_headers_S, 5, "/models/2S.onnx"],
  ["2A", pre_mid_headers_A, 8, "/models/2A.onnx"],
  [
    "3",
    pre_headers.concat(mid_headers).concat(post_headers),
    3,
    "/models/3.onnx",
  ],
  ["3S", pre_mid_post_headers_S, 6, "/models/3S.onnx"],
  ["3A", pre_mid_post_headers_A, 9, "/models/3A.onnx"],
];

const header_mapping: Record<string, string> = {
  "time.surg": "Year of surgery",
  age: "Age",
  height: "Height",
  weight: "Weight",
  sex: "Sex",
  smoke: "Smoke",
  hp: "Hypertension",
  dm: "Diabetes mellitus",
  chd: "Coronary heart disease",
  arrhy: "Arrhythmia",
  copd: "COPD",
  pad: "Peripheral arterial disease",
  rf: "Chronic kidney disease",
  radio: "Radiotherapy",
  chemo: "Chemotherapy",
  asa: "ASA physical status",
  "class.surg.t1": "Procedure type",
  pn: "Parenteral nutrition",
  "hb.pre": "Preoperative Hb",
  "log.wbc.pre": "ln(Preoperative WBC)",
  "plt.pre": "Preoperative Plt",
  "alb.pre": "Preoperative Alb",
  "log.alt.pre": "ln(Preoperative Alt)",
  "log.tbil.pre": "ln(Preoperative Tbil)",
  "scr.pre": "Preoperative Scr",
  bleed: "Estimated blood loss",
  plasma: "Plasma transfusion",
  rbc: "RBC transfusion",
  stoma: "Stoma",
  aa: "Surgical approach",
  icu: "ICU admission",
  "los.icu": "ICU admission > 1 day",
  "alb.change": "change of Alb",
  "hb.post": "Postoperative Hb",
  "log.ratio.plt": "ln(post/pre of Plt)",
  "log.tbil.post": "ln(Postoperative Tbil)",
  "log.scr.post": "ln(Postoperative Scr)",
  "ratio.alt": "post/pre of Alt",
  "log.wbc.post": "ln(Postoperative WBC)",
  "wbc.pre": "Preoperative WBC",
  "wbc.post": "Postoperative WBC",
  "tbil.pre": "Preoperative Tbil",
  "alb.post": "Postoperative Alb",
  "alt.post": "Postoperative Alt",
  "alt.pre": "Preoperative Alt",
  "scr.post": "Postoperative Scr",
  "tbil.post": "Postoperative Tbil",
  "plt.post": "Postoperative Plt",
};

const now_methods = [
  {
    id: "1",
    text: "Preoperative",
  },
  {
    id: "2",
    text: "Preoperative+Intraoperative",
  },
  {
    id: "3",
    text: "Preoperative+Intraoperative+Postoperative",
  },
];

export { usedHeaders_list, label_name, header_mapping, now_methods };

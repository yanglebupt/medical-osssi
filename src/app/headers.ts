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
const mid_headers = [
  "plasma", "rbc", "stoma", "bleed","aa",
  "hr_lower","hr_upper","sbp_lower","sbp_upper","dbp_lower","dbp_upper"
];
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

const numerical_headers: Array<string> = [
  "age","time.surg","height","weight","plasma","rbc","bleed",
  "hr_lower","hr_upper","sbp_lower","sbp_upper","dbp_lower","dbp_upper",
  "plt.pre","plt.post","hb.pre","hb.post","alb.pre",
  "alb.post","tbil.pre","tbil.post","wbc.pre","wbc.post",
  "alt.pre","alt.post","scr.pre","scr.post"
]

const category_headers: Array<string> = [
  "sex", "smoke","hp","dm","chd","arrhy","copd","pad",
  "rf","chemo","radio","pn","asa","icu","los.icu",
  "stoma","class.surg.t1","aa"
]

const category_options: Record<string, Array<string>> = {
  "sex": ["male", "female"],
  "pn":["no","one day before surgery","more than one day before surgery"],
  "asa":["","I","II","III","IV"],
  "class.surg.t1":["upper gastrointestinal tract","lower gastrointestinal tract","pancreaticoduodenectomy","pancreatic body and tail resection surgery"],
  "aa":["manual fasten","medical instrument fasten","medical instrument plus manual fasten","mixture fasten","medical machine fasten"],
}

function get_options_by_header(header: string) {
  return category_options[header] ?? ["no", "yes"]
}
             

const label_name = "ssi.bin";
const usedHeaders_list: Array<[string, string[]]> = [
  ["1", pre_headers],
  ["2", pre_headers.concat(mid_headers)],
  [
    "3",
    pre_headers.concat(mid_headers).concat(post_headers),
  ],
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
  "hr_lower": "TWA-HR < 45 bpm",
  "hr_upper": "TWA-HR > 100 bpm",
  "sbp_lower": "TWA-SBp < 90 mmHg",
  "sbp_upper": "TWA-SBp > 160 mmHg",
  "dbp_lower": "TWA-DBp < 50 mmHg",
  "dbp_upper": "TWA-DBp > 100 mmHg"
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

export { usedHeaders_list, label_name, header_mapping, now_methods,numerical_headers, category_headers, get_options_by_header };

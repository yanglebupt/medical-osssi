/*================= RFE+手动特征筛选+特征变化的结果 ================= */

import { is_empty } from "../tool";

// 术前变量
const display_pre_headers = {
  "Basic Information": ['height', 'weight'],
  "Comorbidity": ['copd', 'pad', 'pn', 'radio', 'rf', 'arrhy'],
  "Surgical Information": ['time.surg', 'class.surg.t1', 'asa'],
};
// 术中变量
const display_mid_headers = {
  "$": ['rbc', 'stoma', 'bleed', 'plasma', 'aa', 'sbp_lower'],
  "Basic Information": ['height', 'weight'],
  "Comorbidity": ['copd', 'pad', 'pn', 'radio', 'arrhy'],
  "Surgical Information": ['time.surg', 'class.surg.t1', 'asa'],
};
// 术后变量
const display_post_headers = {
  "$": ['icu'],
  "Intraoperative Predictors": ['stoma', 'bleed', 'plasma', 'aa', 'sbp_lower', 'hr_lower'],
  "Preoperative Laboratory Values": ['scr.pre', 'alt.pre'],
  "Basic Information": ['height', 'weight'],
  "Comorbidity": ['copd', 'pad', 'pn', 'radio', 'arrhy'],
  "Surgical Information": ['time.surg', 'class.surg.t1', 'asa'],
  "Postoperative Laboratory Values": ['alb.post', 'wbc.post', 'alt.post', 'hb.post'],
};


const pre_headers = [
    'time.surg', 'bmi', 'copd', 'pad', 'pn', 'class.surg.t1',
    'asa', 'radio', 'rf', 'arrhy'
]
const mid_headers = [
    'time.surg', 'bmi', 'copd', 'pad', 'pn', 'class.surg.t1',
    'asa', 'radio', 'arrhy', 'rbc', 'stoma', 'bleed',
    'plasma', 'aa', 'sbp_lower'
]
const post_headers = [
    'time.surg', 'bmi', 'copd', 'pad', 'pn', 'class.surg.t1',
    'asa', 'radio', 'arrhy', 'stoma', 'bleed',
    'plasma', 'aa', 'sbp_lower', 'hr_lower', 'icu', 'alb.post', 'wbc.post', 'alt.((post-pre)/pre)',
    'scr.((post-pre)/pre)', 'hb.post'
]

// 可选特征
export const optional_features: Array<string> = [
  'sbp_lower', 'hr_lower'
]

// 数值特征
export const numerical_headers: Array<string> = [
  "age","time.surg","height","weight","bmi","plasma","rbc","bleed",
  "hr_lower","hr_upper","sbp_lower","sbp_upper","dbp_lower","dbp_upper",
  "plt.pre","plt.post","hb.pre","hb.post","alb.pre",
  "alb.post","tbil.pre","tbil.post","wbc.pre","wbc.post",
  "alt.pre","alt.post","scr.pre","scr.post"
]

// 选项特征
export const category_headers: Array<string> = [
  "sex", "smoke","hp","dm","chd","arrhy","copd","pad",
  "rf","chemo","radio","pn","asa","icu","los.icu",
  "stoma","class.surg.t1","aa"
]

// 特征选项
export const category_options: Record<string, Array<string>> = {
  "sex": ["Male", "Female"],
  "pn":["No","One day before surgery","More than one day before surgery"],
  "asa":["","I","II","III","IV"],
  "class.surg.t1":["Upper gastrointestinal tract","Lower gastrointestinal tract","Pancreaticoduodenectomy","Pancreatic body and tail resection surgery"],
  "aa":["Manual fasten","Medical instrument fasten","Medical instrument plus manual fasten","Mixture fasten","Medical machine fasten"],
}

// 数值单位
export const numerical_units: Record<string, string> = {
  "height":"m", "weight":"kg","plasma":"ml","rbc":"U","bleed":"ml",
  "plt.pre":"x10^9/L","plt.post":"x10^9/L","hb.pre":"g/L","hb.post":"g/L","alb.pre":"g/L",
  "alb.post":"g/L","tbil.pre":"umol/L","tbil.post":"umol/L","wbc.pre":"x10^9/L","wbc.post":"x10^9/L",
  "alt.pre":"U/L","alt.post":"U/L","scr.pre":"umol/L","scr.post":"umol/L"
}

// select 类型
export const category_select_type: Record<string, Array<string>> = {
  "select": ["pn", "class.surg.t1", "aa"],
}

export function get_category_select_type(header: string) {
  for (let i = 0, sts = Object.keys(category_select_type), n = sts.length; i < n; i++) {
    if(category_select_type[sts[i]].includes(header)) return sts[i] 
  }
  return ""
}

export function get_options_by_header(header: string) {
  return category_options[header] ?? ["No", "Yes"]
}


export const display_usedHeaders_list: Array<{name:string, title:string, headers: Record<string, string[]>, used_headers: string[]}> = [
  {name: "1", title: "Preoperative Predictors", headers: display_pre_headers, used_headers: pre_headers},
  {name: "2", title: "Intraoperative Predictors", headers: display_mid_headers, used_headers: mid_headers},
  {name: "3", title: "Postoperative Predictors", headers: display_post_headers, used_headers: post_headers},
];

function is_header_in_headers(h:string, headers: Record<string, string[]>){
  for (let i = 0, hss = Object.values(headers), n = hss.length; i < n; i++) {
    if(hss[i].includes(h)) return true
  }
  return false
}

export function filter_empty_in_headers(headers: Record<string, string[]>, form: Record<string, number>){
  let no_empty_headers: string[] = []
  for (let i = 0, hss = Object.values(headers), n = hss.length; i < n; i++) {
    no_empty_headers = no_empty_headers.concat(hss[i].filter((h) => is_empty(form, h)))
  }
  return no_empty_headers
}

export function filter_display_headers(idx: number, headers: Record<string, string[]>){
  const filter_headers: Record<string, string[]> = {}
  Object.keys(headers).forEach(k=>{
    const n_headers = headers[k].filter(h=>{
      for (let i = 0; i < idx; i++) {
        if(is_header_in_headers(h, display_usedHeaders_list[i].headers)) return false
      }
      return true
    })
    if(n_headers.length > 0) filter_headers[k] = n_headers
  })
  return filter_headers
}

export const header_mapping: Record<string, string> = {
  "time.surg": "Year of surgery",
  "age": "Age",
  "height": "Height",
  "weight": "Weight",
  "bmi": "BMI",
  "sex": "Sex",
  "smoke": "Smoke",
  "hp": "Hypertension",
  "dm": "Diabetes mellitus",
  "chd": "Coronary heart disease",
  "arrhy": "Arrhythmia",
  "copd": "COPD",
  "pad": "Peripheral arterial disease",
  "rf": "Chronic kidney disease",
  "radio": "Radiotherapy",
  "chemo": "Chemotherapy",
  "asa": "ASA physical status",
  "class.surg.t1": "Procedure type",
  "pn": "Parenteral nutrition",
  "hb.pre": "Preoperative Hb",
  "log.wbc.pre": "ln(Preoperative WBC)",
  "plt.pre": "Preoperative Plt",
  "alb.pre": "Preoperative Alb",
  "log.alt.pre": "ln(Preoperative Alt)",
  "log.tbil.pre": "ln(Preoperative Tbil)",
  "scr.pre": "Preoperative Scr",
  "bleed":  "Estimated blood loss",
  "plasma": "Plasma transfusion",
  "rbc": "RBC transfusion",
  "stoma": "Stoma",
  "aa": "Surgical approach",
  "icu": "ICU admission",
  "los.icu": "ICU admission > 1 day",
  "alb.change": "change of Alb",
  "hb.post": "Postoperative Hb",
  "log.ratio.plt": "ln(post/pre of Plt)",
  "log.tbil.post": "ln(Postoperative Tbil)",
  "log.scr.post": "ln(Postoperative Scr)",
  "ratio.alt": "post/pre of Alt",
  "log.wbc.post": "ln(Postoperative WBC)",
  "wbc.pre":"Preoperative WBC",
  "wbc.post":"Postoperative WBC",
  "tbil.pre":"Preoperative Tbil",
  "alb.post":"Postoperative Alb",
  "alt.post":"Postoperative Alt",
  "alt.pre":"Preoperative Alt",
  "scr.post":"Postoperative Scr",
  "tbil.post":"Postoperative Tbil",
  "plt.post":"Postoperative Plt",
  "hr_lower": "TWA(HR < 45 bpm)",
  "hr_upper": "TWA(HR > 100 bpm)",
  "sbp_lower": "TWA(SBp < 90 mmHg)",
  "sbp_upper": "TWA(SBp > 160 mmHg)",
  "dbp_lower": "TWA(DBp < 50 mmHg)",
  "dbp_upper": "TWA(DBp > 100 mmHg)",
  'alb.(post-pre)': "Postoperative ΔAlb", 
  'wbc.(post-pre)': "Postoperative ΔWBC", 
  'alt.(post-pre)': "Postoperative ΔAlt", 
  'scr.(post-pre)': "Postoperative ΔScr", 
  'hb.(post-pre)': "Postoperative ΔHb",
  'alb.((post-pre)/pre)': "Postoperative RelΔAlb", 
  'wbc.((post-pre)/pre)': "Postoperative RelΔWBC", 
  'alt.((post-pre)/pre)': "Postoperative RelΔAlt", 
  'scr.((post-pre)/pre)': "Postoperative RelΔScr", 
  'hb.((post-pre)/pre)': "Postoperative RelΔHb",
  'alb.(post/pre)': "Postoperative Ratio Alb", 
  'wbc.(post/pre)': "Postoperative Ratio WBC", 
  'alt.(post/pre)': "Postoperative Ratio Alt", 
  'scr.(post/pre)': "Postoperative Ratio Scr", 
  'hb.(post/pre)': "Postoperative Ratio Hb"
}
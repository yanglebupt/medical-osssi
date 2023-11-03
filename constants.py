dtype = "float64"

# 全部变量
usedHeaders = ["sex","age","height","weight","bmi","smoke",
               "hp","dm","chd","arrhy","copd","pad","rf","chemo",
               "radio","pn","picc","sbp.pre","dbp.pre","hr.pre",
               "plt.pre","plt.post","hb.pre","hb.post","alb.pre",
               "alb.post","tbil.pre","tbil.post","wbc.pre","wbc.post",
               "alt.pre","alt.post","scr.pre","scr.post","asa","bleed",
               "trans.surg","trans.post","icu","los.icu","scopy","open",
               "scopy.open","stoma","class.surg.t1"
              ]


# 筛选的连续变量
headers_1 = ["sex","age","height","weight","bmi","bleed","trans.surg","trans.post"]

# 筛选的分类变量
headers_2 = ["smoke","hp","dm","chd","arrhy","copd","pad",
             "rf","chemo","radio","pn","picc","asa","icu","los.icu",
             "scopy","open","scopy.open","stoma","class.surg.t1"
            ]
             
# 筛选的术前术后变量
headers_3 = ["sbp.pre","dbp.pre","hr.pre",
             "plt.pre","plt.post","hb.pre","hb.post","alb.pre",
             "alb.post","tbil.pre","tbil.post","wbc.pre","wbc.post",
             "alt.pre","alt.post","scr.pre","scr.post"
            ]


# fea16
var_one = [
    {"key":"weight","rename":"weight"},
    {"key":"copd","rename":"copd"},
    {"key":"pad", "rename":"pad"},
    {"key":"picc","rename":"picc"},
    {"key":"icu","rename":"icu"},
    {"key":"los.icu","rename":"icuG1"},
    {"key":"scopy.open","rename":"scopy2open"},
    {"key":"stoma","rename":"stoma"},
    {"key":"bleed","func":lambda fea: 1*(fea>500),"rename":"bleedG500"},
    {"key":"class.surg.t1","rename":"surgClass"}
]
var_pre_post = [
    {"pre":"wbc.pre","post":"wbc.post","ctype":"post","islog":True,"rename":"WBCPostLog"},
    {"pre":"plt.pre","post":"plt.post","ctype":"change","islog":False,"rename":"PltChange"},
    {"pre":"hb.pre","post":"hb.post","ctype":"change","islog":False,"rename":"HbChange"},
    {"pre":"alb.pre","post":"alb.post","ctype":"post","islog":False,"rename":"AlbPost"},
    {"pre":"alt.pre","post":"alt.post","ctype":"rate","islog":True,"rename":"AltRateLog"},
    {"pre":"scr.pre","post":"scr.post","ctype":"rate","islog":False,"rename":"ScrRate"},
]
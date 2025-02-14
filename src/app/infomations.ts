const disclaimer = "The risks provided by the model should not be used solely for clinical decision-making and must be evaluated by professional physicians. Patients should always consult professional physicians when deciding on their infection and treatment, and should not make clinical decisions based solely on the risks provided by the model, we are not responsible for these decisions."
export const side_infos = [
    {title:"Notes", text: (styles: CSSModuleClasses) => `Before calculating risk of preoperative/intraoperative/postoperative organ/space surgical site infection, all values except for TWA(*) for the corresponding and previous stages must be entered. If you want to calculate TWA(*), you can request script by contacting us xxx@qq.com. <br /><br />
After input, some predictors such as BMI and RelΔ in the paper will be further automatically calculated (but not displayed), and finally used for model calculation of infection risk. <br /><br />
Request webpage and models requires networking, but then all subsequent calculations are offline, so the data entered by you will not be uploaded. If you want to access the entire offline webpage and models, please go to <a href="https://github.com/yanglebupt/medical-osssi" target="_blank">https://github.com/yanglebupt/medical-osssi</a> <br /><br />
<b class=${styles["disclaimer-title"]}><i>Disclaimer</i></b>
<div class=${styles["disclaimer"]}>${disclaimer}</div>`},
    {title:"About", text: `The website and models were developed as a collaboration between xxxx and xxxx. <br /><br />
The model was trained and calibrated based on XGBoost, using grid search strategy to find the best hyperparameters, and ultimately achieved good performance in both discrimination and calibration <br /><br />
The feature selection process firstly used recursive feature elimination (RFE) for automatic selection, and then manually added or removed some features based on the experience of professional physicians <br /><br />
More results can be found in the paper <a href="#">xxx</a>`}
]
const disclaimer = "The infection risks provided by these models must be evaluated by professional physicians. Patients should always consult professional physicians when deciding on their infections and treatments, and should not make any clinical decisions based on the risks provided by these models solely. We are not responsible for these decisions."
export const side_infos = [
    {title:"Notes", text: (styles: CSSModuleClasses) => `Before calculating risk of pre/intra/post-operative organ/space surgical site infection, all values except for TWA(*) for the corresponding and previous stages must be entered. If you want to calculate TWA(*), you can contact us by yanglebupt@qq.com. <br /><br />
After input, some predictors such as BMI and RelΔ in the paper will be further automatically calculated (but not displayed), and finally used for model calculation of infection risk. <br /><br />
Request webpage and models requires networking, but then all subsequent calculations are offline, so the data entered by you will never be uploaded. If you want to access the entire offline webpage and models, please go to <a href="https://github.com/yanglebupt/medical-osssi" target="_blank">https://github.com/yanglebupt/medical-osssi</a>. <br /><br />
<b class=${styles["disclaimer-title"]}><i>Disclaimer</i></b>
<div class=${styles["disclaimer"]}>${disclaimer}</div>`},
    {title:"About", text: `The website and models were developed as a collaboration between Peking Union Medical College Hospital and Beijing University of Posts and Telecommunications. <br /><br />
These models were trained based on XGBoost with isotonic regression calibration, and using grid search strategy to find the best hyperparameters. These models have achieved good performance in both discrimination and calibration. <br /><br />
The feature selection process firstly used recursive feature elimination (RFE) for automatic selection, and then manually added or removed some features based on the experience of professional physicians. <br /><br />
More results can be found in the paper <a href="#"></a>.`}
]
import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import cs from "classnames";
import { header_mapping, get_options_by_header, numerical_headers, display_usedHeaders_list, filter_display_headers, get_category_select_type, filter_empty_in_headers, numerical_units, optional_features, compute_form } from "./headers";
import { predict, method_mapping, mean_std, range } from "../tool";
import { side_infos } from "./infomations";

const wheel_scroll_events = ["wheel", "mousewheel", "DOMMouseScroll"]
function stopScrollFun(evt: any) {
  if(evt.preventDefault) {  
    // Firefox  
    evt.preventDefault();  
    evt.stopPropagation();  
  } else {  
    // IE  
    evt.cancelBubble=true;  
    evt.returnValue=false;  
  }
  return false;  
}

export const AppWithStyles = ({ styles }: { styles: CSSModuleClasses }) => {
  useEffect(()=>{
    const eles = document.querySelectorAll(`input[type="number"]`)
    if (eles) {
      eles.forEach(el=>{
        const iel = el as HTMLInputElement;
        wheel_scroll_events.forEach(e=>iel.addEventListener(e, stopScrollFun, {passive: false}))
      })
    }
  })

  const states = range(0, display_usedHeaders_list.length).map(()=>{
    const [form, setForm] = useState<Record<string, number>>({});
    const [checked, setChecked] = useState(false);
    const [probas, setProbas] = useState<{
      method: string;
      probs_list: Array<Array<number>>;
    }[]>([]);
    const [error, setError] = useState("")
    const [progress, setProgress] = useState("")
    const [loading, setLoading] = useState(false);
    return {form, setForm, checked, setChecked, probas, setProbas, error, setError, progress, setProgress, loading, setLoading};
  })

  const actions = display_usedHeaders_list.map(({name, headers, used_headers}, idx)=>{
    const pre_forms = range(0, idx+1).map(i=>states[i].form);
    const pre_setCheckeds = range(0, idx+1).map(i=>states[i].setChecked);
    const {loading, setLoading, setProgress, setError, setProbas} = states[idx];

    const emptyHeaders = useMemo(
      () => filter_empty_in_headers(headers, pre_forms, optional_features),
      pre_forms
    );

    const submit = useCallback(()=>{
      if (loading) return;
      pre_setCheckeds.forEach(setChecked=>setChecked(true));
      if (emptyHeaders.length === 0) {
        setLoading(true);
        setProgress("")
        // merge pre_forms
        const form = compute_form(pre_forms.reduce((pre, form)=> ({...pre, ...form}), {}))
        predict([form], used_headers, name, [
          1,
          used_headers.length,
        ], (message, progress)=>{
          setProgress(`${message}-${(progress*100).toFixed(0)}%`)
        }).then((results) => {
          setError("")
          console.log(results)
          setProbas(results);
          setLoading(false);
          pre_setCheckeds.forEach(setChecked=>setChecked(false));
        }).catch((error: Error)=>{
          setError(error.message);
          setProbas([]);
          setLoading(false);
          pre_setCheckeds.forEach(setChecked=>setChecked(false));
        });
      }
    }, [loading, ...pre_forms])

    return {emptyHeaders, submit}
  })

  return (
    <>
      <h3 className={styles["title"]}>
        <img src={`${import.meta.env.VITE_BASE}/logo.png`} alt="logo" className={styles["logo"]}/>
        <span>Predictive Tool of Organ/Space Surgical Site Infection</span>
      </h3>
      <div className={styles["content"]}>
        <div className={styles["left-form"]}>
          {
            display_usedHeaders_list.map(({name, title, headers: ori_headers}, idx) => {
              const headers = filter_display_headers(idx, ori_headers)
              const {emptyHeaders, submit} = actions[idx]
              const {form, setForm, checked, error, probas, progress, loading} = states[idx]
              return <div className={styles["header-containers"]} key={name}>
                <h4 className={styles["title"]}>{title}</h4>
                <div className={styles["one-headers"]}>
                  {Object.keys(headers).map((k) => {
                    return <Fragment key={k}>
                      {k!="$" && <div className={styles["header-type"]}>{k}</div>}
                      <div className={styles["headers"]}>
                        {headers[k].map(h=><div
                          key={h}
                          className={cs(
                            styles["header-row"],
                            emptyHeaders.includes(h) && checked ? styles["empty"] : ""
                          )}
                        >
                          {optional_features.includes(h) && <i className={styles["unit"]}>(Optional)</i>}
                          {numerical_units[h] && <span className={styles["unit"]}>{numerical_units[h]}</span>}
                          {numerical_headers.includes(h) ?
                            <input
                              className={styles["ipt"]}
                              id={h}
                              type="number"
                              autoComplete="off"
                              onChange={({ target }: any) => {
                                setForm({
                                  ...form,
                                  [h]: (target as HTMLInputElement).valueAsNumber,
                                });
                              }}
                              value={(form[h] ?? NaN) + ""}
                            />
                          : (get_category_select_type(h)=="select" ? 
                              <select 
                                id={h} 
                                className={cs(styles["ipt"], styles["opts"], (form[h]??-1) == -1 ? styles["opts-not-selected"] : "")} 
                                onChange={({target})=>{
                                  setForm({
                                    ...form,
                                    [h]: parseInt((target as HTMLSelectElement).value),
                                  });
                                }} 
                              value={form[h]??-1}
                              >
                                <option value={-1} key={-1} className={styles["opt"]}>Select {header_mapping[h]}...</option>
                                {
                                  get_options_by_header(h)
                                    .map(((op, idx)=>!!op?<option className={styles["opt"]} value={idx} key={op}>{op}</option>:null))
                                }
                              </select> :
                              <button id={h} className={cs(styles["ipt"], styles["radios"])}>
                                {
                                  get_options_by_header(h)
                                    .map(((op, idx)=> !!op ? 
                                    <Fragment key={idx}>
                                      <input type="radio" id={`${h}-${op}`} name={h} className={styles["opt"]} value={idx} key={op} 
                                        onChange={({target})=>{
                                          setForm({
                                            ...form,
                                            [h]: parseInt((target as HTMLInputElement).value),
                                          });
                                        }}
                                        checked={(form[h]??-1)==idx}
                                      />
                                      <label htmlFor={`${h}-${op}`}>{op}</label>
                                    </Fragment>: null))
                                }
                              </button>
                          )}
                          <label htmlFor={h} className={styles["hname"]}>{header_mapping[h]}</label>
                        </div>)}
                      </div>
                    </Fragment>
                  })}
                </div>
                <div>
                  <button className={styles["btn"]} onClick={submit}>
                    Risk of Infection
                  </button>
                  <span>
                    {checked && emptyHeaders.length > 0 ? (
                      <span className={styles["warn"]}>
                        Please fill in the valid values within the red border before clicking on the prediction
                      </span>
                    ) : loading ? (
                      <>
                        <i className={styles["loading"]}></i>
                        <span>{progress}</span>
                      </>
                    ) : null}
                  </span>
                </div>
                {
                  !loading && (error != "" || probas.length>0) &&
                  <div className={styles["results"]}>
                    {error == "" ?
                      probas.map(({method, probs_list})=>
                        <div key={method} className={styles["method-result"]}>
                          <span className={styles["method"]}>{method_mapping[method]}：</span>
                          <span className={styles["result"]}>{mean_std(probs_list.map(nums=>nums[0])).join("±")}</span>
                        </div>
                      ) : <div className={styles["warn"]}>{error}</div>
                    }
                  </div>
                }
              </div>
            })
          }          
        </div>
        <div className={styles["right-side"]}>
          {
            side_infos.map(({title, text})=>
              <div className={styles["item"]} key={title}>
                <h3 className={styles["title"]}>{title}</h3>
                <div className={styles["text"]} dangerouslySetInnerHTML={{__html: typeof text=="function" ? text(styles) : text}}></div>
              </div>)
          }
        </div>
      </div>
    </>
  );
};

import { useCallback, useMemo, useState } from "react";
import cs from "classnames";
import { usedHeaders_list, header_mapping, now_methods, get_options_by_header, numerical_headers } from "./headers";
import { convertDictNumber, predict, method_mapping, mean_std, is_empty } from "../tool";

export const AppWithStyles = ({ styles }: { styles: CSSModuleClasses }) => {
  const [selectedId, setSelectedId] = useState("1");
  const [checked, setChecked] = useState(false);
  const [name, headers] = useMemo(
    () => usedHeaders_list.find((p) => p[0] === selectedId)!,
    [selectedId]
  );
  const [form, setForm] = useState<Record<string, string>>({});

  const emptyHeaders = useMemo(
    () => headers.filter((h) => is_empty(form, h)),
    [headers, form]
  );

  const [probas, setProbas] = useState<{
      method: string;
      probs_list: Array<Array<number>>;
  }[]>([]);
  const [error, setError] = useState("")
  const [progress, setProgress] = useState("")
  const [loading, setLoading] = useState(false);

  const submit = useCallback(() => {
    if (loading) return;
    setChecked(true);
    if (emptyHeaders.length === 0) {
      setLoading(true);
      setProgress("")
      predict(convertDictNumber([form]), headers, name, [
        1,
        headers.length,
      ], (message, progress)=>{
        setProgress(`${message}-${(progress*100).toFixed(0)}%`)
      }).then((results) => {
        setError("")
        console.log(results)
        setProbas(results);
        setLoading(false);
        setChecked(false);
        // scroll to end
        window.scrollTo({
          top: document.documentElement.scrollHeight,
          behavior: 'smooth'
        });
      }).catch((error: Error)=>{
        setError(error.message);
        setProbas([]);
        setLoading(false);
        setChecked(false);
      });
    }
  }, [emptyHeaders, form, loading]);

  return (
    <>
      <div className={styles["select"]}>
        {now_methods.map(({ id, text }) => (
          <div
            key={id}
            className={cs(
              styles["select-item"],
              selectedId == id ? styles["selected"] : ""
            )}
            onClick={() => setSelectedId(id)}
          >
            <span className={styles["select-item-mark"]}></span>
            <span className={styles["select-item-text"]}>{text}</span>
          </div>
        ))}
      </div>
      <div className={styles["headers"]}>
        {headers.map((h) => (
          <div
            key={h}
            className={cs(
              styles["header-row"],
              emptyHeaders.includes(h) && checked ? styles["empty"] : ""
            )}
          >
            {/* 验证输入是否合法，整数，小数，二分类 */}
            {numerical_headers.includes(h)?
              <input
                className={styles["ipt"]}
                id={h}
                type="text"
                autoComplete="off"
                onBeforeInput={({ target }: any) =>
                  (target.dataset.pre = target.value)
                }
                onInput={({ target }: any) => {
                  // 限制不合法输入
                  const pre = target.dataset.pre as string;
                  const now = target.value as string;
                  let value = now;
                  if (
                    pre.includes(".") &&
                    now.indexOf(".") !== now.lastIndexOf(".")
                  ) {
                    value = now
                      .split(".")
                      .reduce(
                        (pre: string, cur: string, idx: number) =>
                          `${pre}${cur}${idx === 0 ? "." : ""}`,
                        ""
                      );
                  } else {
                    value = target.value.replace(/[^(\d|\.)]/g, "");
                  }
                  setForm({
                    ...form,
                    [h]: value,
                  });
                }}
                value={form[h] ?? ""}
              />
            : 
              <select 
                id={h} 
                className={cs(styles["ipt"], styles["opts"], ((form[h] as any)??-1) == -1 ? styles["opts-not-selected"] : "")} 
                onChange={({target})=>{
                  setForm({
                    ...form,
                    [h]: target.value + "",
                });
              }} 
              value={form[h]??-1}
              title={get_options_by_header(h)[form[h] as any] ?? "==Select=="}
              >
                <option value={-1} key={-1} className={styles["opt"]}>==Select==</option>
                {
                  get_options_by_header(h)
                    .map(((op, idx)=>!!op?<option className={styles["opt"]} value={idx} key={op}>{op}</option>:null))
                }
              </select>
            }
            <label htmlFor={h} title={h}>{header_mapping[h]}</label>
          </div>
        ))}
      </div>
      <div>
        <button className={styles["btn"]} onClick={submit}>
          Predict Risk of infection
        </button>
        <span>
          {checked && emptyHeaders.length > 0 ? (
            <span className={styles["warn"]}>
              "Please fill in the valid values within the red border before
              clicking on the prediction"
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
        !loading &&
        <div className={styles["results"]}>
          {error == "" ?
            probas.map(({method, probs_list})=>
              <div key={method} className={styles["method-result"]}>
                <span className={styles["method"]}>{method_mapping[method]}：</span>
                <span className={styles["result"]}>{mean_std(probs_list.map(nums=>nums[0])).join("±")}</span>
              </div>
            ):<div className={styles["warn"]}>{error}</div>
          }
        </div>
      }
    </>
  );
};

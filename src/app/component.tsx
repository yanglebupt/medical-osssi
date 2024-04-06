import { useCallback, useMemo, useState } from "react";
import cs from "classnames";
import { usedHeaders_list, header_mapping, now_methods } from "./headers";
import { convertDictNumber, predict } from "../tool";

export const AppWithStyles = ({ styles }: { styles: CSSModuleClasses }) => {
  const [selectedId, setSelectedId] = useState("1");
  const [checked, setChecked] = useState(false);
  const [__, headers, _, model_path] = useMemo(
    () => usedHeaders_list.find((p) => p[0] === selectedId)!,
    [selectedId]
  );
  const [form, setForm] = useState<Record<string, string>>({});

  const emptyHeaders = useMemo(
    () => headers.filter((h) => !(h in form) || form[h] === ""),
    [headers, form]
  );

  const [proba, setProba] = useState("");
  const [loading, setLoading] = useState(false);

  const submit = useCallback(() => {
    if (loading) return;
    setChecked(true);
    if (emptyHeaders.length === 0) {
      setLoading(true);
      predict(convertDictNumber([form]), headers, model_path, [
        1,
        headers.length,
      ]).then((proba) => {
        setProba(proba[0] + "");
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
            <label htmlFor={h}>{header_mapping[h]}</label>
            {/* 验证输入是否合法，整数，小数，二分类 */}
            <input
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
                setProba("");
              }}
              value={form[h] || ""}
            />
          </div>
        ))}
      </div>
      <div>
        <button className={styles["btn"]} onClick={submit}>
          Predict
        </button>
        <span className={styles["res"]}>
          Risk of infection：
          {checked && emptyHeaders.length > 0 ? (
            <span className={styles["warn"]}>
              "Please fill in the valid values within the red border before
              clicking on the prediction"
            </span>
          ) : loading ? (
            <i className={styles["loading"]}></i>
          ) : proba === "" ? (
            ""
          ) : (
            Number(proba).toFixed(3)
          )}
        </span>
      </div>
    </>
  );
};

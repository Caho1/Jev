"""将独立 Jev 对照的进度与结果追加到现有训练看板，不更改训练证据。"""
import argparse
import json
import time
from pathlib import Path

from banking77_dashboard import APP,LOCAL,LAB,atomic_json,build,now
from banking77_case_rows import case_rows

DIRECTORY=LAB/"results/banking77-jev-v1"


def update(build_status=None):
    snapshot=json.loads((APP/"src/data.json").read_text())
    state=json.loads((DIRECTORY/"status.json").read_text())
    protocol=json.loads((DIRECTORY/"protocol.json").read_text())
    result=json.loads((DIRECTORY/"results.json").read_text()) if (DIRECTORY/"results.json").exists() else None
    complete=bool(result and result["complete"])
    previous=snapshot["queries"].get("jev_progress",{}).get("rows",[{}])[0]
    if previous.get("updatedAt")==state["updated_at"] and not build_status:return False
    progress={"phase":state["phase"],"completed":state["completed"],"total":state["total"],
              "success":state["success"],"errors":state["errors"],"progressRate":state["completed"]/state["total"],
              "updatedAt":state["updated_at"],"model":"jev-1.13.0","complete":complete}
    accuracy=[]
    class_rows=[]
    comparison=[]
    label_audit=[]
    labels=json.loads((LAB/"data/banking77-v1/labels.json").read_text())
    if complete:
        names={"baseline":"原始 Laya","tuned":"微调 Laya","jev":"Jev 1.13.0"}
        for key,name in names.items():
            m=result["models"][key]
            accuracy.append({"model":name,"accuracyRate":m["accuracy"],"macroF1Rate":m["macro_f1"],
                             "correct":m["correct"],"samples":m["records"],"failures":m["failures"],
                             "validSubsetAccuracyRate":result["valid_subset"][key]["accuracy"],"validSubsetRecords":result["valid_subset"][key]["records"]})
        m=result["paired"]["tuned"]
        comparison=[{"delta":m["accuracy_difference_b_minus_a"],"lower":m["group_bootstrap_95_percent"][0],"upper":m["group_bootstrap_95_percent"][1],
                     "jevOnly":m["b_only_correct"],"layaOnly":m["a_only_correct"],"bothCorrect":m["both_correct"],"bothWrong":m["both_wrong"]}]
        for i,label in enumerate(labels):
            l=result["models"]["tuned"];j=result["models"]["jev"]
            class_rows.append({"intent":label,"samples":l["support"][i],"layaRecallRate":l["per_class_recall"][i],"jevRecallRate":j["per_class_recall"][i],
                               "delta":j["per_class_recall"][i]-l["per_class_recall"][i]})
        original=[json.loads(line) for line in (LAB/"data/banking77-v1/test.jsonl").read_text().splitlines()]
        pin_rows=[row for row in original if row["label_name"]=="get_physical_card"]
        label_audit=[{"intent":"get_physical_card","samples":len(pin_rows),"pinMentions":sum("pin" in row["state"].casefold() for row in pin_rows),"example":pin_rows[0]["state"],"exampleId":pin_rows[0]["id"]}]
    def query(rows,definitions,files):
        return {"rows":rows,"source":{"label":"BANKING77 · Jev 与冻结 Laya 对照","executedAt":state["updated_at"],
                "files":files,"metricDefinitions":definitions,
                "caveats":["Jev 是开箱 API，Laya 已在 BANKING77 训练集微调；比较的是两种可用配置，而非相同训练条件。",
                           "BANKING77 是公开数据，Jev 是否在上游训练见过这些数据未知。",
                           "仅验证英文银行短句闭集分类，不代表中文、未知意图或生产流量表现。"],
                "evidenceFlow":[{"title":"固定比较协议","detail":json.dumps(protocol["protocol"],ensure_ascii=False)},
                                {"title":"请求 Jev","detail":"scripts/evaluate_banking77_jev.py；state 为原始客户文本，questions.intent 为同一指令及规范顺序 77 类，正确标签从未发送。API 输出仅保留为评测证据。"},
                                {"title":"配对计算","detail":"按测试 ID 对齐本次 Jev choice 与已保存的两份 Laya test.npz，分类错误及请求失败计入同一分母。"}]},
                "methods":[{"language":"text","code":"scripts/evaluate_banking77_jev.py 中 classification 和 paired；原始预测按冻结测试 ID 对齐。scripts/update_jev_dashboard.py 提取结果，scripts/banking77_case_rows.py 按 ID 连接保存的预测与原文，校验文件哈希并核对正确数，不重新调用模型。"}]}
    snapshot["queries"].update({
        "jev_progress":query([progress],[{"label":"请求进度","definition":"已记录请求 / 3,080；失败不会从分母剔除。一次独立调用处理一条消息。"}],["results/banking77-jev-v1/status.json"]),
        "jev_accuracy":query(accuracy,[{"label":"准确率","definition":"正确预测数 / 同一 3,080 条官方测试样本。请求失败计为错误。"},
                                             {"label":"Macro F1","definition":"77 个类别的 F1 等权平均。使用 Jev 返回的 choice；不会用舍入后并列概率改变预测。"},
                                             {"label":"有效回答子集","definition":"排除 Jev 的 1 条 choice 与概率最大值不一致的校验失败，三模型在同一剩余 3,079 条上再比较。主表仍保留完整 3,080 条。"}],
                             ["results/banking77-jev-v1/results.json","results/banking77-jev-v1/responses.jsonl","results/banking77-lora-v1/baseline-test.npz","results/banking77-lora-v1/tuned-test.npz"]),
        "jev_pair":query(comparison,[{"label":"准确率差","definition":"Jev 准确率减微调 Laya 准确率，正值为 Jev 更高。95% 区间按重复组配对 bootstrap 2,000 次。"}],
                         ["results/banking77-jev-v1/results.json"]),
        "jev_classes":query(class_rows,[{"label":"召回率差","definition":"该类 Jev 召回率减微调 Laya 召回率，每类固定 40 个测试样本。"}],
                            ["results/banking77-jev-v1/results.json"]),
        "jev_label_audit":query(label_audit,[{"label":"类别命名检查","definition":"本轮完成后的解释性检查，不改评分或重新请求。get_physical_card 类的 40 条测试文本全部含 PIN；类别名不足以说明真实标注语义。"}],
                                ["data/banking77-v1/test.jsonl","results/banking77-jev-v1/protocol.json"])
    })
    if complete:
        snapshot["queries"]["jev_cases"] = query(case_rows(), [
            {"label":"逐题预测", "definition":"完整的 3,080 条官方测试文本及标准标签。按冻结 ID 连接两份 Laya logits 和 Jev 已保存回答；正确表示预测类别与原始标准标签完全相同。"},
            {"label":"预测概率", "definition":"Laya 使用独立校准集拟合的温度计算 softmax；Jev 显示 API 原始舍入概率，未额外校准。两者数值不代表相同的正确率保证。Top 3 按各自概率排序；Jev 最终类别仍以 API choice 为准。"},
            {"label":"校验失败", "definition":"1 条 Jev 记录的 choice 与最大概率不一致，原始回答未保存，预测与概率保持缺失；按冻结协议计为错误。"},
        ], ["data/banking77-v1/test.jsonl", "results/banking77-lora-v1/baseline-test.npz",
            "results/banking77-lora-v1/tuned-test.npz", "results/banking77-lora-v1/results.json",
            "results/banking77-jev-v1/responses.jsonl", "results/banking77-jev-v1/protocol.json"])
        snapshot["queries"]["jev_cases"]["payloadColumns"] = ["baselineTop3", "tunedTop3", "jevTop3"]
    snapshot["generatedAt"]=now()
    if build_status:snapshot["buildStatus"]=build_status
    atomic_json(APP/"src/data.json",snapshot)
    build()
    atomic_json(LOCAL/"dashboard-version.json",{"generatedAt":snapshot["generatedAt"]})
    print(json.dumps(progress,ensure_ascii=False),flush=True)
    return True


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--watch",action="store_true");p.add_argument("--build-status",choices=["updating","complete"]);a=p.parse_args()
    while True:
        update(a.build_status)
        a.build_status=None
        state=json.loads((DIRECTORY/"status.json").read_text())
        if not a.watch or state["phase"] in ["complete","stopped_on_error"]:break
        time.sleep(15)

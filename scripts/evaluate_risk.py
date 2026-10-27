"""Evaluate existing risk scores against independent CSV labels (standard library only)."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

FIELDS = ['video_id', 'video_path', 'source_id', 'person_id', 'emotion_label', 'is_fake', 'is_fraud', 'label_source']


def indexed(rows, key):
    result = {}
    for row in rows:
        name = str(row.get(key, '')).strip()
        if not name or name in result:
            raise ValueError(f'Missing or duplicate {key}: {name}')
        result[name] = row
    return result


def proportion(k, n):
    if not n:
        return {'value': None, 'ci95_wilson': None, 'denominator': 0}
    p, z = k / n, 1.959963984540054
    center = (p + z*z/(2*n))/(1+z*z/n)
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return {'value': p, 'ci95_wilson': [max(0, center-radius), min(1, center+radius)], 'denominator': n}


def binary_metrics(pairs, threshold):
    counts = Counter(('TP' if y else 'FP') if s >= threshold else ('FN' if y else 'TN') for y, s in pairs)
    tp, fp, fn, tn = (counts[k] for k in ('TP', 'FP', 'FN', 'TN'))
    positive, negative = tp+fn, fp+tn
    roc, ap = None, None
    if positive and negative:
        # Group tied scores: no arbitrary ordering inside a tie.
        groups = {}
        for y, score in pairs:
            group = groups.setdefault(score, [0, 0])
            group[y] += 1
        seen_pos = seen_neg = 0
        area = average_precision = 0.0
        for score in sorted(groups, reverse=True):
            neg, pos = groups[score]
            area += neg * (seen_pos + pos/2)
            seen_pos += pos
            seen_neg += neg
            average_precision += (pos/positive)*(seen_pos/(seen_pos+seen_neg))
        roc, ap = area/(positive*negative), average_precision
    return {'n': len(pairs), 'positive': positive, 'negative': negative,
            'confusion': {k: counts[k] for k in ('TP', 'FP', 'FN', 'TN')},
            'recall': proportion(tp, positive), 'precision': proportion(tp, tp+fp),
            'false_positive_rate': proportion(fp, negative),
            'roc_auc': roc, 'pr_auc_average_precision': ap,
            'conclusion': '证据不足：缺少可评估的正类或负类。' if not (positive and negative) else
            '已计算标签区分指标；是否可用于预警需结合独立测试集、预设误报要求和样本代表性判断，不能自动认定有效。'}


def evaluate(labels, results, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and in [0, 1]')
    truth, predictions = indexed(labels, 'video_id'), indexed(results, 'id')
    rows = []
    for key, label in truth.items():
        row = {field: label.get(field, '') for field in FIELDS}
        for target in ('is_fake', 'is_fraud'):
            value = str(row[target]).strip()
            if value not in ('', '0', '1'):
                raise ValueError(f'{key}: {target} must be blank, 0 or 1')
            row[target] = value
        result = predictions.get(key)
        status, score = 'missing_result', None
        if result is not None:
            status = str(result.get('status', 'missing_status'))
            if status == 'success':
                candidate = result.get('risk_score')
                if isinstance(candidate, (int, float)) and not isinstance(candidate, bool) and math.isfinite(candidate) and 0 <= candidate <= 1:
                    score = float(candidate)
                else:
                    status = 'invalid_score'
        row.update(processing_status=status, risk_score=score, threshold=threshold,
                   alert='' if score is None else int(score >= threshold),
                   original_prediction='' if result is None else result.get('prediction', ''),
                   error='' if result is None else result.get('error', ''))
        for target in ('is_fake', 'is_fraud'):
            row[target+'_outcome'] = '' if score is None or not row[target] else (
                ('TP' if row[target] == '1' else 'FP') if score >= threshold else
                ('FN' if row[target] == '1' else 'TN'))
        rows.append(row)
    valid = [r for r in rows if r['risk_score'] is not None]
    metrics = {'threshold': threshold, 'labels_count': len(rows), 'results_count': len(results),
               'status_counts': dict(Counter(r['processing_status'] for r in rows)),
               'unmatched_result_ids': sorted(set(predictions)-set(truth)),
               'valid_scores': len(valid), 'high_risk_count': sum(r['alert'] for r in valid),
               'mean_risk_score': sum(r['risk_score'] for r in valid)/len(valid) if valid else None}
    for target in ('is_fake', 'is_fraud'):
        metrics[target] = binary_metrics([(int(r[target]), r['risk_score']) for r in valid if r[target] != ''], threshold)
        metrics[target]['missing_label_count'] = sum(r[target] == '' for r in rows)
        metrics[target]['labeled_without_score'] = sum(r[target] != '' and r['risk_score'] is None for r in rows)
    return rows, metrics


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render(metrics, rows):
    lines = ['# 风险分数与视频标签对照评估', '',
             f"固定阈值：{metrics['threshold']}；标签样本：{metrics['labels_count']}；有效风险结果：{metrics['valid_scores']}；高风险：{metrics['high_risk_count']}。", '',
             f"处理状态：{metrics['status_counts']}；无标签清单匹配的结果：{len(metrics['unmatched_result_ids'])}。", '',
             '本报告不调整阈值。原始录制不自动标为非诈骗；情感标签不转换为真假或诈骗标签。', '']
    for target, title in [('is_fake', '深度合成检测'), ('is_fraud', '诈骗预警')]:
        m = metrics[target]
        lines += [f'## {title}', '', m['conclusion'], '',
                  f"有效标签样本 {m['n']}（正类 {m['positive']}、负类 {m['negative']}）；缺标签 {m['missing_label_count']}；有标签无有效分数 {m['labeled_without_score']}。", '',
                  '| 实际 / 预测 | 预警 | 不预警 |', '|---|---:|---:|',
                  f"| 正类 | {m['confusion']['TP']} | {m['confusion']['FN']} |",
                  f"| 负类 | {m['confusion']['FP']} | {m['confusion']['TN']} |", '',
                  '| 指标 | 数值 | 95% Wilson 区间 |', '|---|---|---|']
        for key, name in [('recall', '召回率'), ('precision', '精确率'), ('false_positive_rate', '误报率')]:
            item = m[key]
            value = '不可计算' if item['value'] is None else f"{item['value']:.2%}"
            ci = '不可计算' if item['ci95_wilson'] is None else ' – '.join(f'{x:.2%}' for x in item['ci95_wilson'])
            lines.append(f'| {name} | {value} | {ci} |')
        lines += ['', f"ROC-AUC：{m['roc_auc'] if m['roc_auc'] is not None else '不可计算'}；PR-AUC（采用 Average Precision 定义）：{m['pr_auc_average_precision'] if m['pr_auc_average_precision'] is not None else '不可计算'}。", '']
        errors = [r for r in rows if r[target+'_outcome'] in ('FP', 'FN')]
        lines += ['错误案例（最多前 10 项；全部见 comparison.csv）：']
        lines += [f"- {r['video_id']}：{r[target+'_outcome']}，风险 {r['risk_score']:.4f}" for r in errors[:10]] or ['无可列出的误报／漏报案例。']
        lines.append('')
    lines += ['## 解释限制', '',
              '- 失败、缺失或无效分数不计入检测指标，已单独统计；指标仅代表成功处理的样本。',
              '- Wilson 区间按独立视频近似计算；同人物、同源片段相关性会使该区间过于乐观。AUC 未提供置信区间。',
              '- 精确率依赖样本中的正类占比；试验集结果不能直接解释为部署效果。',
              '- 必须按人物或原始片段分组划分开发／测试数据；本工具不自动验证数据独立性。',
              '- 当前 RAVDESS pilot 没有真假及诈骗真值时，只能描述高风险比例，不能证明诈骗检测有效。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True, help='run_batch results.json')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing directory is refused')
    args = parser.parse_args()
    with args.labels.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not set(FIELDS) <= set(reader.fieldnames):
            parser.error('Labels CSV must contain: '+', '.join(FIELDS))
        labels = list(reader)
    results = json.loads(args.results.read_text(encoding='utf-8-sig'))
    rows, metrics = evaluate(labels, results, args.threshold)
    metrics['inputs'] = {name: {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                         for name, path in [('labels', args.labels), ('results', args.results)]}
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output/'comparison.csv', rows, list(rows[0]) if rows else FIELDS)
    (args.output/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (args.output/'evaluation.md').write_text(render(metrics, rows), encoding='utf-8')
    print(f"Evaluation written to {args.output.resolve()}")


if __name__ == '__main__':
    main()

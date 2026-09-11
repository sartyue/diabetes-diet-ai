# -*- coding: utf-8 -*-
"""
main.py — diabetes_project（DiaDiet AI · Render 部署版）
=========================================================
糖尿病风险评估 + 个性化一日三餐饮食推荐，单文件 FastAPI 后端。

设计约束（Render 免费实例友好）：
  * 业务代码 100% 原生标准库：
      - 不使用 pydantic BaseModel，请求体用 json.loads 手动解析 + 手动校验
        （避免 pydantic 版本差异导致的部署报错）；
      - 数据模型使用 dataclass / 普通类。
  * 食谱生成采用确定性规则选餐，不依赖任何外部 LLM API，
    结果可复现、断网可用，适合课程演示与论文实验。
  * 风险规则依据《中国2型糖尿病防治指南（2020年版）》；
    营养目标依据《中国糖尿病医学营养治疗指南》原则。

运行方式：
  本地：  python main.py
  Render 启动命令： uvicorn main:app --host 0.0.0.0 --port $PORT
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")


# ==========================================================
# 第一部分：糖尿病风险规则引擎
# （原 diabetes_risk_engine.py 整体移植，无 pydantic）
# ==========================================================

class Thresholds:
    """《中国2型糖尿病防治指南（2020年版）》核心切点（唯一事实来源）。"""
    GUIDELINE_VERSION = "CDS-2020"
    # --- 血糖指标（静脉血浆葡萄糖，mmol/L）---
    FPG_NORMAL_MAX = 6.1        # 空腹血糖正常上限；6.1~7.0 为空腹血糖受损(IFG)
    FPG_DM = 7.0                # 空腹血糖糖尿病诊断切点（需复测确认）
    PG2H_NORMAL_MAX = 7.8       # OGTT 2h 正常上限；7.8~11.1 为糖耐量异常(IGT)
    PG2H_DM = 11.1              # OGTT 2h 糖尿病诊断切点
    RANDOM_GLUCOSE_DM = 11.1    # 随机血糖糖尿病切点（须伴典型症状才可诊断）
    # --- 糖化血红蛋白（%，须在标准化检测条件下）---
    HBA1C_NORMAL_MAX = 6.0      # 正常上限；6.0~6.5 为糖尿病前期
    HBA1C_DM = 6.5              # 2020 版指南新增诊断切点之一
    # --- 体格指标（中国标准）---
    BMI_NORMAL_MIN = 18.5       # BMI 正常下限
    BMI_NORMAL_MAX = 24.0       # 中国超重标准
    BMI_OBESE = 28.0            # 中国肥胖标准
    WAIST_MALE_ABDOMINAL = 90.0    # 男性腹型肥胖腰围切点（cm）
    WAIST_FEMALE_ABDOMINAL = 85.0  # 女性腹型肥胖腰围切点（cm）


class RiskLevel(str, Enum):
    """风险等级（枚举成员顺序即严重程度，便于取 max）。"""
    LOW = "低风险"      # 指标处于正常范围
    MEDIUM = "中风险"   # 糖尿病前期（IFG / IGT / HbA1c 边缘升高）
    HIGH = "高风险"     # 符合糖尿病诊断切点（提示就医复测确认）


class LabMetrics:
    """
    检验指标输入模型（普通类 + 手动校验，替代 pydantic BaseModel）。
    所有字段可选（就诊时不一定每项都查），缺失指标不参与评估，报告中会提示。
    """
    # 供前端展示用的中文名映射
    FIELD_LABELS = {
        "fpg": "空腹血糖", "ogtt_2h": "OGTT 2小时血糖",
        "hba1c": "糖化血红蛋白", "random_glucose": "随机血糖",
        "bmi": "BMI", "waist": "腰围",
    }

    def __init__(self, data: dict):
        self.age: Optional[int] = self._to_int(data.get("age"))
        self.sex: Optional[str] = data.get("sex")
        self.fpg: Optional[float] = self._to_float(data.get("fpg"))
        self.ogtt_2h: Optional[float] = self._to_float(data.get("ogtt_2h"))
        self.hba1c: Optional[float] = self._to_float(data.get("hba1c"))
        self.random_glucose: Optional[float] = self._to_float(data.get("random_glucose"))
        self.bmi: Optional[float] = self._to_float(data.get("bmi"))
        self.waist: Optional[float] = self._to_float(data.get("waist"))
        self._check_all_range()

    # ---------- 类型转换工具（空字符串/None 一律归一化为 None）----------
    @staticmethod
    def _to_float(v) -> Optional[float]:
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise ValueError(f"数值字段 {v!r} 无法解析为数字，请检查输入")

    @staticmethod
    def _to_int(v) -> Optional[int]:
        if v is None or v == "":
            return None
        try:
            return int(float(v))
        except (TypeError, ValueError):
            raise ValueError(f"整数字段 {v!r} 无法解析为整数，请检查输入")

    # ---------- 范围校验（防止填错单位，如 mg/dL 当 mmol/L 填入）----------
    def _check_all_range(self):
        for name in ("fpg", "ogtt_2h", "random_glucose"):
            val = getattr(self, name)
            if val is not None and not (1.0 <= val <= 45.0):
                raise ValueError(f"血糖数值 {val} mmol/L 超出合理范围 [1.0, 45.0]，请检查单位")
        if self.hba1c is not None and not (3.0 <= self.hba1c <= 20.0):
            raise ValueError(f"HbA1c 数值 {self.hba1c}% 超出合理范围 [3.0, 20.0]（应为百分比 %）")
        if self.bmi is not None and not (10.0 <= self.bmi <= 60.0):
            raise ValueError(f"BMI 数值 {self.bmi} 超出合理范围 [10.0, 60.0]")
        if self.waist is not None and not (40.0 <= self.waist <= 200.0):
            raise ValueError(f"腰围 {self.waist} cm 超出合理范围 [40.0, 200.0]")


@dataclass(frozen=True)
class Rule:
    """规则三要素：等级贡献 / 触发谓词 / 人话版原因生成器。"""
    rule_id: str
    name: str
    level: Optional[RiskLevel]        # None 表示"仅提示、不改分级"
    triggered: callable
    reason: callable


def _build_rules() -> list[Rule]:
    """集中注册全部规则（顺序不影响结果，最终取最高等级）。"""
    T = Thresholds
    return [
        # ---------- 高风险规则（满足任一糖尿病诊断切点）----------
        Rule("DM-01", "空腹血糖达糖尿病切点", RiskLevel.HIGH,
             lambda m: m.fpg is not None and m.fpg >= T.FPG_DM,
             lambda m: f"空腹血糖 {m.fpg} mmol/L，达到糖尿病诊断切点 "
                       f"{T.FPG_DM} mmol/L（正常参考 <{T.FPG_NORMAL_MAX}）"),
        Rule("DM-02", "OGTT 2h 血糖达糖尿病切点", RiskLevel.HIGH,
             lambda m: m.ogtt_2h is not None and m.ogtt_2h >= T.PG2H_DM,
             lambda m: f"餐后(OGTT)2小时血糖 {m.ogtt_2h} mmol/L，达到糖尿病诊断切点 "
                       f"{T.PG2H_DM} mmol/L（正常参考 <{T.PG2H_NORMAL_MAX}）"),
        Rule("DM-03", "糖化血红蛋白达糖尿病切点", RiskLevel.HIGH,
             lambda m: m.hba1c is not None and m.hba1c >= T.HBA1C_DM,
             lambda m: f"糖化血红蛋白 HbA1c {m.hba1c}%，达到糖尿病诊断切点 "
                       f"{T.HBA1C_DM}%（正常参考 <{T.HBA1C_NORMAL_MAX}%）"),
        Rule("DM-04", "随机血糖达糖尿病切点", RiskLevel.HIGH,
             lambda m: m.random_glucose is not None and m.random_glucose >= T.RANDOM_GLUCOSE_DM,
             lambda m: f"随机血糖 {m.random_glucose} mmol/L ≥ {T.RANDOM_GLUCOSE_DM} mmol/L；"
                       f"若伴典型症状（多饮多尿多食、体重下降）需高度怀疑糖尿病，建议尽快就医复测"),
        # ---------- 中风险规则（糖尿病前期）----------
        Rule("PRE-01", "空腹血糖受损(IFG)", RiskLevel.MEDIUM,
             lambda m: m.fpg is not None and T.FPG_NORMAL_MAX <= m.fpg < T.FPG_DM,
             lambda m: f"空腹血糖 {m.fpg} mmol/L，超过正常值 {T.FPG_NORMAL_MAX} "
                       f"但未达糖尿病切点，属于空腹血糖受损（糖尿病前期）"),
        Rule("PRE-02", "糖耐量异常(IGT)", RiskLevel.MEDIUM,
             lambda m: m.ogtt_2h is not None and T.PG2H_NORMAL_MAX <= m.ogtt_2h < T.PG2H_DM,
             lambda m: f"餐后(OGTT)2小时血糖 {m.ogtt_2h} mmol/L，超过正常值 "
                       f"{T.PG2H_NORMAL_MAX} 但未达糖尿病切点，属于糖耐量异常（糖尿病前期）"),
        Rule("PRE-03", "HbA1c 边缘升高", RiskLevel.MEDIUM,
             lambda m: m.hba1c is not None and T.HBA1C_NORMAL_MAX <= m.hba1c < T.HBA1C_DM,
             lambda m: f"糖化血红蛋白 HbA1c {m.hba1c}%，处于 {T.HBA1C_NORMAL_MAX}%~"
                       f"{T.HBA1C_DM}% 之间，提示血糖控制边缘状态（糖尿病前期）"),
        # ---------- 代谢风险提示（不改变分级，只追加原因）----------
        Rule("BMI-01", "超重肥胖", None,
             lambda m: m.bmi is not None and m.bmi >= T.BMI_NORMAL_MAX,
             lambda m: f"BMI {m.bmi} 超过中国超重标准 {T.BMI_NORMAL_MAX}"
                       + (f"，已达肥胖标准 {T.BMI_OBESE}" if m.bmi >= T.BMI_OBESE else "")
                       + "，是胰岛素抵抗与2型糖尿病的重要危险因素"),
        Rule("BMI-02", "体重过低", None,
             lambda m: m.bmi is not None and m.bmi < T.BMI_NORMAL_MIN,
             lambda m: f"BMI {m.bmi} 低于正常下限 {T.BMI_NORMAL_MIN}，需警惕营养不良或其他消耗性疾病"),
        Rule("WC-01", "腹型肥胖", None,
             lambda m: (m.waist is not None and m.sex == "male" and m.waist >= T.WAIST_MALE_ABDOMINAL)
                       or (m.waist is not None and m.sex == "female" and m.waist >= T.WAIST_FEMALE_ABDOMINAL),
             lambda m: f"腰围 {m.waist} cm，达到腹型肥胖标准"
                       f"（男 ≥{T.WAIST_MALE_ABDOMINAL:.0f} / 女 ≥{T.WAIST_FEMALE_ABDOMINAL:.0f} cm），"
                       f"内脏脂肪堆积与糖代谢异常密切相关"),
    ]


@dataclass
class RiskReport:
    """结构化评估报告：等级 + 触发原因 + 数据缺失提示。"""
    level: RiskLevel
    reasons: list[str] = field(default_factory=list)
    triggered_rules: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    guideline: str = Thresholds.GUIDELINE_VERSION
    disclaimer: str = ("本结果为课程项目的教育性风险评估，不构成医学诊断；"
                       "达到高风险切点者应到医疗机构复测确认。")

    def to_dict(self) -> dict:
        return {
            "risk_level": self.level.value,
            "reasons": self.reasons,
            "triggered_rules": self.triggered_rules,
            "missing_fields": self.missing_fields,
            "guideline": self.guideline,
            "disclaimer": self.disclaimer,
        }


class DiabetesRiskEvaluator:
    """糖尿病风险规则引擎（对外唯一评估入口）。"""
    # 决定分级的"核心血糖字段"，用于缺失提示
    CORE_FIELDS = {"fpg": "空腹血糖", "ogtt_2h": "OGTT 2小时血糖",
                   "hba1c": "糖化血红蛋白", "random_glucose": "随机血糖"}

    def __init__(self, rules: Optional[list[Rule]] = None):
        self.rules = rules if rules is not None else _build_rules()

    def evaluate(self, metrics: LabMetrics) -> RiskReport:
        reasons, triggered_ids, levels = [], [], []
        for rule in self.rules:
            if rule.triggered(metrics):
                reasons.append(f"[{rule.rule_id}] {rule.reason(metrics)}")
                triggered_ids.append(rule.rule_id)
                if rule.level is not None:
                    levels.append(rule.level)
        final_level = max(levels) if levels else RiskLevel.LOW
        missing = [cn for key, cn in self.CORE_FIELDS.items()
                   if getattr(metrics, key) is None]
        return RiskReport(level=final_level, reasons=reasons,
                          triggered_rules=triggered_ids, missing_fields=missing)


# ==========================================================
# 第二部分：食物营养数据库（数据内嵌，免去 foods.json 文件）
# ==========================================================

@dataclass(frozen=True)
class FoodItem:
    """一条食物记录：所有营养值均为"每100g可食部"近似值。"""
    id: str
    name: str
    category: str          # staple主食/vegetable蔬菜/fruit水果/protein蛋白/dairy奶制品/nut坚果
    kcal: float
    carbs: float
    protein: float
    fat: float
    fiber: float
    gi: int                # 升糖指数（0 = 不适用，如纯蛋白质食物）
    gl: int                # 血糖负荷（近似值）
    purine_mg: float       # 嘌呤 mg/100g
    note: str


# 数据来源：《中国食物成分表》（杨月欣主编）、公开食物 GI/GL 表（演示用精简版）
FOODS_RAW: list[dict] = [
    {"id": "F01", "name": "燕麦片", "category": "staple", "kcal": 367, "carbs": 61.6, "protein": 15.0, "fat": 6.7, "fiber": 5.3, "gi": 55, "gl": 34, "purine_mg": 24, "note": "高膳食纤维，β-葡聚糖有助于延缓葡萄糖吸收"},
    {"id": "F02", "name": "糙米", "category": "staple", "kcal": 348, "carbs": 75.0, "protein": 7.5, "fat": 2.7, "fiber": 3.4, "gi": 70, "gl": 52, "purine_mg": 35, "note": "保留麸皮，纤维高于精白米"},
    {"id": "F03", "name": "荞麦面条", "category": "staple", "kcal": 340, "carbs": 70.0, "protein": 12.0, "fat": 2.2, "fiber": 3.5, "gi": 59, "gl": 41, "purine_mg": 40, "note": "低GI主食，含芦丁"},
    {"id": "F04", "name": "全麦面包", "category": "staple", "kcal": 246, "carbs": 45.0, "protein": 9.0, "fat": 3.4, "fiber": 6.0, "gi": 69, "gl": 31, "purine_mg": 20, "note": "注意选择无添加糖的产品"},
    {"id": "F05", "name": "鲜玉米", "category": "staple", "kcal": 112, "carbs": 22.8, "protein": 4.0, "fat": 1.2, "fiber": 2.9, "gi": 55, "gl": 13, "purine_mg": 9, "note": "整粒食用，咀嚼慢，升糖相对平缓"},
    {"id": "F06", "name": "红薯", "category": "staple", "kcal": 90, "carbs": 20.7, "protein": 1.6, "fat": 0.2, "fiber": 2.0, "gi": 77, "gl": 16, "purine_mg": 4, "note": "高GI，高风险用户应控制单次食用量"},
    {"id": "F07", "name": "白米饭", "category": "staple", "kcal": 116, "carbs": 25.9, "protein": 2.6, "fat": 0.3, "fiber": 0.3, "gi": 83, "gl": 21, "purine_mg": 18, "note": "高GI精制主食"},
    {"id": "F08", "name": "白馒头", "category": "staple", "kcal": 223, "carbs": 47.0, "protein": 7.0, "fat": 1.1, "fiber": 1.3, "gi": 88, "gl": 41, "purine_mg": 17, "note": "高GI精制主食"},
    {"id": "F09", "name": "油条", "category": "staple", "kcal": 388, "carbs": 51.0, "protein": 6.9, "fat": 17.6, "fiber": 0.9, "gi": 75, "gl": 38, "purine_mg": 19, "note": "高GI+高脂肪油炸食品"},
    {"id": "F10", "name": "西兰花", "category": "vegetable", "kcal": 34, "carbs": 4.3, "protein": 4.1, "fat": 0.6, "fiber": 1.6, "gi": 15, "gl": 1, "purine_mg": 29, "note": "高纤维低热量蔬菜"},
    {"id": "F11", "name": "菠菜", "category": "vegetable", "kcal": 24, "carbs": 2.8, "protein": 2.6, "fat": 0.3, "fiber": 1.7, "gi": 15, "gl": 1, "purine_mg": 13, "note": "深绿叶菜，富含镁"},
    {"id": "F12", "name": "番茄", "category": "vegetable", "kcal": 20, "carbs": 4.0, "protein": 0.9, "fat": 0.2, "fiber": 0.5, "gi": 15, "gl": 1, "purine_mg": 4, "note": "低热量，可生食"},
    {"id": "F13", "name": "黄瓜", "category": "vegetable", "kcal": 16, "carbs": 2.9, "protein": 0.8, "fat": 0.2, "fiber": 0.5, "gi": 15, "gl": 1, "purine_mg": 3, "note": "加餐可选，热量极低"},
    {"id": "F14", "name": "苹果", "category": "fruit", "kcal": 53, "carbs": 12.3, "protein": 0.2, "fat": 0.2, "fiber": 1.2, "gi": 36, "gl": 5, "purine_mg": 1, "note": "低GI水果，建议带皮食用"},
    {"id": "F15", "name": "橙子", "category": "fruit", "kcal": 48, "carbs": 11.1, "protein": 0.8, "fat": 0.2, "fiber": 0.6, "gi": 43, "gl": 5, "purine_mg": 2, "note": "整果食用，不要榨汁"},
    {"id": "F16", "name": "香蕉", "category": "fruit", "kcal": 93, "carbs": 22.0, "protein": 1.4, "fat": 0.2, "fiber": 1.2, "gi": 52, "gl": 11, "purine_mg": 3, "note": "中低GI，单次半根为宜"},
    {"id": "F17", "name": "西瓜", "category": "fruit", "kcal": 26, "carbs": 5.8, "protein": 0.6, "fat": 0.1, "fiber": 0.3, "gi": 72, "gl": 4, "purine_mg": 1, "note": "高GI水果，高风险用户应避免"},
    {"id": "F18", "name": "鸡蛋", "category": "protein", "kcal": 144, "carbs": 2.8, "protein": 13.3, "fat": 8.8, "fiber": 0.0, "gi": 0, "gl": 0, "purine_mg": 1, "note": "优质蛋白，几乎不含嘌呤"},
    {"id": "F19", "name": "鸡胸肉", "category": "protein", "kcal": 133, "carbs": 1.3, "protein": 19.4, "fat": 5.0, "fiber": 0.0, "gi": 0, "gl": 0, "purine_mg": 140, "note": "低脂优质蛋白"},
    {"id": "F20", "name": "猪里脊", "category": "protein", "kcal": 150, "carbs": 1.5, "protein": 20.2, "fat": 6.0, "fiber": 0.0, "gi": 0, "gl": 0, "purine_mg": 150, "note": "畜肉中脂肪较低的部位"},
    {"id": "F21", "name": "三文鱼", "category": "protein", "kcal": 180, "carbs": 0.0, "protein": 20.0, "fat": 10.5, "fiber": 0.0, "gi": 0, "gl": 0, "purine_mg": 83, "note": "富含ω-3脂肪酸"},
    {"id": "F22", "name": "带鱼", "category": "protein", "kcal": 127, "carbs": 3.1, "protein": 17.7, "fat": 4.9, "fiber": 0.0, "gi": 0, "gl": 0, "purine_mg": 190, "note": "中高嘌呤，高尿酸者慎食"},
    {"id": "F23", "name": "猪肝", "category": "protein", "kcal": 129, "carbs": 5.0, "protein": 19.3, "fat": 3.5, "fiber": 0.0, "gi": 0, "gl": 0, "purine_mg": 227, "note": "高嘌呤+高胆固醇内脏"},
    {"id": "F24", "name": "豆腐", "category": "protein", "kcal": 84, "carbs": 4.2, "protein": 8.1, "fat": 3.7, "fiber": 0.4, "gi": 15, "gl": 1, "purine_mg": 55, "note": "植物蛋白，低嘌呤"},
    {"id": "F25", "name": "牛奶", "category": "dairy", "kcal": 54, "carbs": 3.4, "protein": 3.0, "fat": 3.2, "fiber": 0.0, "gi": 27, "gl": 1, "purine_mg": 1, "note": "建议选纯牛奶而非含乳饮料"},
    {"id": "F26", "name": "无糖酸奶", "category": "dairy", "kcal": 60, "carbs": 4.5, "protein": 3.0, "fat": 3.0, "fiber": 0.0, "gi": 30, "gl": 1, "purine_mg": 1, "note": "配料表应无蔗糖/果葡糖浆"},
    {"id": "F27", "name": "核桃", "category": "nut", "kcal": 646, "carbs": 9.6, "protein": 14.9, "fat": 58.8, "fiber": 9.5, "gi": 14, "gl": 1, "purine_mg": 25, "note": "坚果热量高，每日一小把(约15g)即可"},
    {"id": "F28", "name": "花生", "category": "nut", "kcal": 574, "carbs": 16.0, "protein": 24.8, "fat": 44.0, "fiber": 5.5, "gi": 14, "gl": 2, "purine_mg": 79, "note": "坚果热量高，需控制总量"},
]


class FoodDatabase:
    """内存食物库：启动时从内嵌数据构建，提供按名称/类别检索。"""

    def __init__(self, raw: Optional[list[dict]] = None):
        source = raw if raw is not None else FOODS_RAW
        self.foods: list[FoodItem] = [
            FoodItem(
                id=d["id"], name=d["name"], category=d["category"],
                kcal=d["kcal"], carbs=d["carbs"], protein=d["protein"],
                fat=d["fat"], fiber=d["fiber"], gi=d["gi"], gl=d["gl"],
                purine_mg=d["purine_mg"], note=d["note"],
            ) for d in source
        ]
        self._by_name = {f.name: f for f in self.foods}

    def get(self, name: str) -> Optional[FoodItem]:
        return self._by_name.get(name)

    def all(self) -> list[FoodItem]:
        return list(self.foods)


# ==========================================================
# 第三部分：营养目标计算（Mifflin-St Jeor 公式）
# ==========================================================

class NutritionGoalsCalculator:
    """
    依据《中国糖尿病医学营养治疗指南》原则计算每日营养目标：
      BMR (Mifflin-St Jeor)：男 = 10W + 6.25H - 5A + 5；女 = ... - 161
      TDEE = BMR × 活动系数
      超重/肥胖（BMI≥24）时制造 500 kcal 热量缺口（减重目标）
      供能比：碳水 50% / 蛋白 18% / 脂肪 32%；三餐 30% / 40% / 30%
    """
    ACTIVITY_FACTORS = {"sedentary": 1.2, "light": 1.375,
                        "moderate": 1.55, "active": 1.725}
    MEAL_SPLIT = {"breakfast": 0.30, "lunch": 0.40, "dinner": 0.30}

    def calc(self, profile: dict) -> dict:
        sex = profile.get("sex", "male")
        age, height, weight = profile["age"], profile["height_cm"], profile["weight_kg"]
        activity = profile.get("activity", "light")

        # --- BMR（Mifflin-St Jeor, 1990）---
        if sex == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        tdee = bmr * self.ACTIVITY_FACTORS.get(activity, 1.375)

        # --- 减重调整：BMI≥24 超重者每日缺口 500 kcal ---
        bmi = profile.get("bmi") or weight / (height / 100) ** 2
        weight_loss_goal = bmi >= 24.0
        target_kcal = tdee - (500 if weight_loss_goal else 0)
        # 女性安全下限 1200 kcal、男性 1500 kcal（防止极端节食）
        floor = 1500 if sex == "male" else 1200
        target_kcal = max(round(target_kcal), floor)

        return {
            "bmr": round(bmr), "tdee": round(tdee), "target_kcal": target_kcal,
            "carbs_g": round(target_kcal * 0.50 / 4),    # 碳水 4 kcal/g
            "protein_g": round(target_kcal * 0.18 / 4),  # 蛋白 4 kcal/g
            "fat_g": round(target_kcal * 0.32 / 9),      # 脂肪 9 kcal/g
            "weight_loss_goal": weight_loss_goal,
            "meal_kcal": {k: round(target_kcal * v) for k, v in self.MEAL_SPLIT.items()},
        }


# ==========================================================
# 第四部分：硬规则过滤（确定性决策层，LLM/规则选餐无权更改）
# ==========================================================

class HardRuleFilter:
    """
    按风险等级对食物库做硬性过滤，先于选餐执行；过滤结果即"白名单"。
      高风险：剔除 GI > 70 的高GI食物（白米饭/馒头/西瓜/油条/红薯等）
      中风险：剔除 GI > 75；55 < GI ≤ 70 保留但标注"中GI提示"
      低风险：不做GI硬性剔除，高GI食物仅提示
      附加约束：高尿酸 -> 剔除嘌呤 > 150 mg/100g 的食物
    """
    LEVEL_GI_LIMITS = {RiskLevel.HIGH: 70, RiskLevel.MEDIUM: 75,
                       RiskLevel.LOW: 10_000}   # 哨兵值 = 不设上限

    def filter(self, foods: list[FoodItem], level: RiskLevel,
               extra: Optional[dict] = None) -> dict:
        extra = extra or {}
        limit_purine = extra.get("limit_purine", False)
        purine_cap = 150.0
        gi_limit = self.LEVEL_GI_LIMITS[level]

        allowed, excluded = [], []
        for f in foods:
            warnings = []
            # --- 规则1：GI 硬性上限 ---
            if f.gi > gi_limit:
                excluded.append({
                    "name": f.name, "gi": f.gi,
                    "reason": f"GI {f.gi} 超过{level.value}允许上限 "
                              f"{gi_limit if gi_limit < 1000 else '无'}"
                              + ("，属于高升糖指数食物" if f.gi > 70 else ""),
                })
                continue
            # --- 规则2：嘌呤约束（合并症：高尿酸血症/痛风）---
            if limit_purine and f.purine_mg > purine_cap:
                excluded.append({
                    "name": f.name, "purine_mg": f.purine_mg,
                    "reason": f"嘌呤 {f.purine_mg:.0f} mg/100g 超过高尿酸上限 {purine_cap:.0f}",
                })
                continue
            # --- 软标签：中GI 食物（不剔除，但给出提示）---
            if 55 < f.gi <= 70:
                warnings.append(f"中GI({f.gi})，建议控制单次份量并搭配高纤维蔬菜")
            if f.category == "nut":
                warnings.append("坚果热量高，每日不超过 15~20g")
            allowed.append({"food": f, "warnings": warnings})

        return {"allowed": allowed, "excluded": excluded}


# ==========================================================
# 第五部分：确定性规则选餐（三餐食谱生成，无外部 API 依赖）
# ==========================================================

class RuleBasedMealPlanner:
    """
    从白名单中按规则选餐，保证"无 LLM 也能出基础食谱"（可靠性设计）。
    选餐策略：
      * 各餐结构固定：早=主食+奶+水果；午/晚=主食+蛋白+蔬菜
      * 同类食物按"GI 升序 + 纤维降序"排序，随三餐轮换 offset 保证多样性
      * 份量 = 该餐目标热量 × 供能占比 ÷ 食物热量密度，取 10g 整数，
        并受各类别克数上限约束（低热量蔬菜按热量反推会出现 500g+ 的失真份量）
    """
    STRUCTURE = {
        "breakfast": [("staple", 0.55), ("dairy", 0.30), ("fruit", 0.15)],
        "lunch":     [("staple", 0.56), ("protein", 0.24), ("vegetable", 0.20)],
        "dinner":    [("staple", 0.52), ("protein", 0.24), ("vegetable", 0.24)],
    }
    # 各类别单次份量上限（g）：参考《中国糖尿病医学营养治疗指南》食物交换份法
    CATEGORY_CAPS = {"staple": 400, "vegetable": 300, "fruit": 250,
                     "dairy": 300, "protein": 250, "nut": 20}

    def generate(self, goals: dict, allowed_foods: list[FoodItem],
                 seed: int = 0) -> dict:
        """
        seed：轮换种子（由用户档案派生，如 age % 9）。
          * 同一用户 -> 相同 seed -> 相同食谱（确定性、可复现，便于论文实验）；
          * 不同用户 -> 不同 seed -> 食谱有差异化（体现"个性化"）。
        """
        def pick(category: str, offset: int = 0) -> Optional[FoodItem]:
            cands = [f for f in allowed_foods if f.category == category]
            if not cands:
                return None
            # 主食：仅在 GI≤60 的候选中轮换（若存在），把整餐加权GI控制在低GI区间
            if category == "staple":
                low_gi = [f for f in cands if f.gi <= 60]
                if low_gi:
                    cands = low_gi
            # 蛋白类：默认排除高嘌呤（>150 mg/100g，如猪肝/带鱼）候选，
            # 降低普通用户的嘌呤负担；合并高尿酸时由硬过滤进一步收紧
            if category == "protein":
                low_purine = [f for f in cands if f.purine_mg <= 150]
                if low_purine:
                    cands = low_purine
            cands.sort(key=lambda f: (f.gi if f.gi > 0 else 99, -f.fiber))
            return cands[offset % len(cands)]

        meals = {}
        for meal_idx, (meal, parts) in enumerate(self.STRUCTURE.items()):
            meal_kcal = goals["meal_kcal"][meal]
            items = []
            for cat, share in parts:
                # 各类别轮换 offset = 餐序 + 用户种子，保证早/午/晚不重样且用户间有差异
                f = pick(cat, offset=meal_idx + seed)
                if f is None:
                    continue
                portion = round(meal_kcal * share / f.kcal * 100 / 10) * 10
                portion = max(30, min(portion, self.CATEGORY_CAPS.get(cat, 300)))
                items.append({
                    "name": f.name, "portion_g": portion,
                    "reason": f"GI {f.gi}，每100g约 {f.kcal:.0f} kcal；{f.note}",
                })
            meals[meal] = {"items": items, "notes": f"本餐目标约 {meal_kcal} kcal"}

        return {
            "meals": meals,
            "general_advice": [
                "主食粗细搭配，优先选择低GI（GI≤55）的全谷物",
                "先吃蔬菜再吃主食，有助于平缓餐后血糖上升",
                "定时定量进餐，避免饥一顿饱一顿",
                "烹调以蒸煮炖为主，避免油炸和高糖调味",
            ],
            "disclaimer": "本食谱由算法生成，仅供教育参考，不构成医疗建议；"
                          "具体饮食方案请咨询医生或注册营养师。",
        }


# ==========================================================
# 第六部分：营养达标率闭环校验（独立于生成的确定性核算）
# ==========================================================

class ComplianceChecker:
    """
    按食物库营养数据核算食谱的实际总热量/三大营养素/加权GI，
    与营养目标逐项对比，输出达标率与预警（生成-校验闭环）。
    """
    # 各营养素判定容差（实测/目标比值区间），按指南推荐供能比区间换算：
    #   碳水目标 50% 供能 x (0.90, 1.20) -> 45%~60%，与指南一致（对血糖最关键）
    #   蛋白目标 18% 供能 x (0.85, 1.30) -> 15%~23%，指南 15~20% 略放宽
    #   脂肪目标 32% 供能 x (0.55, 1.35) -> 18%~43%；下限较宽是因为演示
    #   食物库未计入烹调用油，食谱脂肪天然偏低（另有 flags 提示补油）
    TOLERANCE = {
        "kcal": (0.90, 1.10),
        "carbs": (0.90, 1.20),
        "protein": (0.85, 1.30),
        "fat": (0.55, 1.35),
    }

    def check(self, meal_plan: dict, goals: dict, db: FoodDatabase) -> dict:
        totals = {"kcal": 0.0, "carbs": 0.0, "protein": 0.0, "fat": 0.0}
        gi_num, gi_den = 0.0, 0.0   # 加权GI：按各食物碳水克数加权

        for meal in meal_plan.get("meals", {}).values():
            for item in meal.get("items", []):
                f = db.get(item.get("name", ""))
                if f is None:
                    continue
                r = item.get("portion_g", 0) / 100.0
                totals["kcal"] += f.kcal * r
                totals["carbs"] += f.carbs * r
                totals["protein"] += f.protein * r
                totals["fat"] += f.fat * r
                if f.carbs > 0 and f.gi > 0:
                    gi_num += f.gi * f.carbs * r
                    gi_den += f.carbs * r

        weighted_gi = round(gi_num / gi_den, 1) if gi_den > 0 else 0.0

        def row(key: str, label: str, target: float, actual: float, unit: str) -> dict:
            ratio = actual / target if target > 0 else 0.0
            lo, hi = self.TOLERANCE[key]
            ok = lo <= ratio <= hi
            return {"label": label, "target": round(target), "actual": round(actual),
                    "unit": unit, "ratio_pct": round(ratio * 100, 1),
                    "status": "达标" if ok else "偏差"}

        rows = [
            row("kcal", "总热量", goals["target_kcal"], totals["kcal"], "kcal"),
            row("carbs", "碳水化合物", goals["carbs_g"], totals["carbs"], "g"),
            row("protein", "蛋白质", goals["protein_g"], totals["protein"], "g"),
            row("fat", "脂肪", goals["fat_g"], totals["fat"], "g"),
        ]

        # 加权GI 判定（独立行，目标为"低GI食谱 ≤55"）
        if weighted_gi <= 55:
            gi_verdict, gi_ok = "低GI食谱（≤55）", True
        elif weighted_gi <= 70:
            gi_verdict, gi_ok = "中GI食谱（56~70），建议减少中GI主食份量", False
        else:
            gi_verdict, gi_ok = "高GI食谱（>70），建议替换主食", False
        rows.append({"label": "加权GI", "target": 55, "actual": weighted_gi,
                     "unit": "", "ratio_pct": None,
                     "status": "达标" if gi_ok else "偏差"})

        # 预警信息（前端黄条提示）
        flags = []
        kcal_ratio = totals["kcal"] / goals["target_kcal"] if goals["target_kcal"] else 1
        if kcal_ratio > 1.10:
            flags.append("食谱总热量超出目标 10% 以上，建议适当减少主食份量")
        elif kcal_ratio < 0.90:
            flags.append("食谱总热量低于目标 10% 以上，可通过加餐（如黄瓜/番茄）补足")
        fat_ratio = totals["fat"] / goals["fat_g"] if goals["fat_g"] else 1
        if fat_ratio < 0.85:
            flags.append("脂肪供能偏低：演示食物库未计入烹调用油，"
                         "实际烹饪时每餐可添加植物油 8~10g（全天约 25g）")
        protein_ratio = totals["protein"] / goals["protein_g"] if goals["protein_g"] else 1
        if protein_ratio > 1.25:
            flags.append("蛋白质高于目标：优质蛋白略充裕对一般人群无碍，"
                         "合并肾功能异常者应咨询医生后调整")

        return {
            "rows": rows,
            "weighted_gi": weighted_gi,
            "gi_verdict": gi_verdict,
            "flags": flags,
            # 实际供能占比（前端饼图数据）
            "energy_breakdown": {
                "carbs_kcal": round(totals["carbs"] * 4),
                "protein_kcal": round(totals["protein"] * 4),
                "fat_kcal": round(totals["fat"] * 9),
            },
        }


# ==========================================================
# 第七部分：业务编排（输入解析 -> 评估 -> 过滤 -> 选餐 -> 校验）
# ==========================================================

# 身高缺失时按性别使用的默认身高（cm），用于从 BMI 反推体重
DEFAULT_HEIGHT = {"male": 170.0, "female": 158.0}


class AssessmentService:
    """无状态业务服务：一次调用完成"评估 + 推荐 + 校验"全流程。"""

    def __init__(self):
        self.db = FoodDatabase()
        self.evaluator = DiabetesRiskEvaluator()

    # ---------- 入参规整：身高/体重/BMI 三者互补 ----------
    @staticmethod
    def _build_profile(data: dict) -> tuple[dict, list[str]]:
        notes: list[str] = []
        sex = data.get("sex") or "male"
        if sex not in ("male", "female"):
            raise ValueError("性别(sex)只能是 male 或 female")
        if data.get("age") in (None, ""):
            raise ValueError("年龄(age)为必填项")
        age = int(float(data["age"]))
        if not (1 <= age <= 120):
            raise ValueError(f"年龄 {age} 超出合理范围 [1, 120]")

        height = LabMetrics._to_float(data.get("height_cm"))
        weight = LabMetrics._to_float(data.get("weight_kg"))
        bmi = LabMetrics._to_float(data.get("bmi"))

        if height and weight:
            bmi = bmi or round(weight / (height / 100) ** 2, 1)
        elif bmi:
            # 只填了 BMI：按默认身高反推体重，保证营养目标可计算
            height = height or DEFAULT_HEIGHT[sex]
            weight = round(bmi * (height / 100) ** 2, 1)
            notes.append(f"未填写身高体重，按默认身高 {height:.0f} cm 与 BMI {bmi} 反推体重 {weight} kg")
        else:
            height = DEFAULT_HEIGHT[sex]
            weight = 65.0 if sex == "male" else 55.0
            notes.append("未填写身高/体重/BMI，营养目标按默认体格参数估算")

        if not (100.0 <= height <= 230.0):
            raise ValueError(f"身高 {height} cm 超出合理范围 [100, 230]")
        if not (25.0 <= weight <= 250.0):
            raise ValueError(f"体重 {weight} kg 超出合理范围 [25, 250]")

        activity = data.get("activity") or "light"
        if activity not in NutritionGoalsCalculator.ACTIVITY_FACTORS:
            activity = "light"

        profile = {"age": age, "sex": sex, "height_cm": height,
                   "weight_kg": weight, "bmi": bmi, "activity": activity}
        return profile, notes

    # ---------- 主流程 ----------
    def run(self, data: dict) -> dict:
        # ① 入参规整（含 BMI 派生）
        profile, notes = self._build_profile(data)

        # ② 风险评估（规则引擎）
        metrics = LabMetrics(data)
        report = self.evaluator.evaluate(metrics)

        # ③ 营养目标（BMR/TDEE/三大营养素/三餐分配）
        goals = NutritionGoalsCalculator().calc(profile)

        # ④ 硬规则过滤（确定性白名单，附"为什么不能吃"的剔除原因）
        limit_purine = bool(data.get("limit_purine", False))
        flt = HardRuleFilter().filter(self.db.all(), report.level,
                                      {"limit_purine": limit_purine})
        allowed_foods = [a["food"] for a in flt["allowed"]]

        # ⑤ 规则选餐生成三餐食谱（seed 由用户档案派生：同用户可复现，异用户有差异）
        seed = profile["age"] % 9
        plan = RuleBasedMealPlanner().generate(goals, allowed_foods, seed=seed)

        # ⑥ 营养达标率闭环校验（独立核算，前端展示表格）
        compliance = ComplianceChecker().check(plan, goals, self.db)

        return {
            "profile": {**profile, "derived_notes": notes},
            "report": report.to_dict(),
            "nutrition_goals": goals,
            "excluded_foods": flt["excluded"],
            "meal_plan": plan,
            "compliance": compliance,
            "engine": {
                "risk_engine": "rule-based (CDS-2020 guideline)",
                "meal_planner": "rule-based deterministic (rotation seed=%d)" % seed,
                "llm_mode": False,
            },
        }


# 单例服务（启动时构建食物库，避免每次请求重复解析）
SERVICE = AssessmentService()


# ==========================================================
# 第八部分：FastAPI 应用与路由
# ==========================================================

app = FastAPI(title="Diabetes Diet Advisor",
              description="糖尿病风险评估 + 个性化饮食推荐（课程项目，非医疗器械）",
              version="1.0.0")

# 挂载静态目录（index.html 及静态资源）
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """根路径直接返回前端页面。"""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health() -> dict:
    """健康检查（Render 部署观测用）。"""
    return {"status": "ok", "food_count": len(SERVICE.db.foods),
            "guideline": Thresholds.GUIDELINE_VERSION}


@app.post("/api/assess")
async def assess(request: Request):
    """
    核心接口：接收体检指标 JSON，返回"风险评估 + 营养目标 + 三餐食谱 + 达标率"。
    注意：请求体用原生 json.loads 解析（不使用 pydantic 模型），
          参数校验由 LabMetrics / AssessmentService 手动完成。
    """
    raw = await request.body()
    if not raw:
        return JSONResponse(status_code=400, content={"error": "请求体为空，请提交 JSON 数据"})
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
    except (json.JSONDecodeError, ValueError) as exc:
        return JSONResponse(status_code=400, content={"error": f"JSON 解析失败: {exc}"})

    try:
        result = SERVICE.run(data)
    except ValueError as exc:      # 业务校验错误 -> 400 + 中文提示
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:       # 兜底：500 且不泄露堆栈
        return JSONResponse(status_code=500,
                            content={"error": f"服务器内部错误: {exc}"})
    return result


# ==========================================================
# 启动入口：适配 Render 的 $PORT 环境变量（本地默认 8000）
# ==========================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

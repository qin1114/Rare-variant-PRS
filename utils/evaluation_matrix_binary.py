import numpy as np
import torch
import pandas as pd
import os

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, rankdata
from scipy import sparse



# ========== R on the liability scale ==========
from scipy.stats import norm

def calculate_liability_r2(final_data, score_name="Probability", use_common=False, K=0.01):
    """
    计算R² on the liability scale
    
    Parameters:
    restricted_model: 限制模型（不包含主要预测变量）
    full_model: 完整模型（包含所有预测变量）
    score_name: PRS列名称，支持输入Probability Common_PRS Combined_PRS
    K: prevalence 
    
    Returns:
    dict: 包含各种R²指标的字典
    """
    try:
        # 尝试包含性别变量的完整模型
        if use_common:
            formula_full = f'disease ~ Q("{score_name}") + Q("Common_PRS") + Q("21003-0.0") + Q("31-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
            formula_null = 'disease ~ Q("21003-0.0") + Q("Common_PRS") + Q("31-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
        else:
            formula_full = f'disease ~ Q("{score_name}") + Q("21003-0.0") + Q("31-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
            formula_null = 'disease ~ Q("21003-0.0") + Q("31-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
        
        full_model = smf.logit(formula_full, data=final_data).fit(disp=False)
        restricted_model = smf.logit(formula_null, data=final_data).fit(disp=False)
        
    except Exception as e:
        # 处理可能的错误（如单性别疾病导致的完美分离问题）
        print(f"完整模型拟合失败: {e}, 尝试不含性别变量的模型")
        
        # 不含性别变量的模型
        if use_common:
            formula_full = f'disease ~ Q("{score_name}") + Q("Common_PRS") + Q("21003-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
            formula_null = 'disease ~ Q("21003-0.0") + Q("Common_PRS") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
        else:
            formula_full = f'disease ~ Q("{score_name}") + Q("21003-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
            formula_null = 'disease ~ Q("21003-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
        
        full_model = smf.logit(formula_full, data=final_data).fit(disp=False)
        restricted_model = smf.logit(formula_null, data=final_data).fit(disp=False)

    # formula_full = f'disease ~ Q("{score_name}") + Q("Common_PRS")'
    # formula_null = 'disease ~ Q("Common_PRS")'
    
    # full_model = smf.logit(formula_full, data=final_data).fit(disp=False)
    # restricted_model = smf.logit(formula_null, data=final_data).fit(disp=False)


    # 提取似然值
    L_res = restricted_model.llf  # 限制模型的对数似然值
    L_full = full_model.llf       # 完整模型的似然值
    N = full_model.nobs           # 样本量
    
    # 计算 R²
    R2_obs = 1 - np.exp((L_res - L_full) * 2 / N)
    
    # 计算 z
    z = norm.pdf(norm.ppf(1 - K))
    
    # 计算 liability R²
    R2_lia = R2_obs * (K * (1 - K)) / (z ** 2)
    
    
    return {
        'L_res': L_res,
        'L_full': L_full,
        'N': N,
        'R2': R2_obs,
        'R2_lia': R2_lia
    }



# ========== Nagelkerke's R ==========
def calculate_nagelkerke_r2(final_data, score_name="Probability", use_common=False):
    """
    计算Nagelkerke R²
    
    Parameters:
    restricted_model: 限制模型（不包含主要预测变量）
    full_model: 完整模型（包含所有预测变量）
    score_name: PRS列名称，支持输入Probability Common_PRS Combined_PRS
    
    Returns:
    dict: 包含各种R²指标的字典
    """
    try:
        # 尝试包含性别变量的完整模型
        if use_common:
            formula_full = f'disease ~ Q("{score_name}") + Q("Common_PRS") + Q("21003-0.0") + Q("31-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
            formula_null = 'disease ~ Q("21003-0.0") + Q("Common_PRS") + Q("31-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
        else:
            formula_full = f'disease ~ Q("{score_name}") + Q("21003-0.0") + Q("31-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
            formula_null = 'disease ~ Q("21003-0.0") + Q("31-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
        
        full_model = smf.logit(formula_full, data=final_data).fit(disp=False)
        restricted_model = smf.logit(formula_null, data=final_data).fit(disp=False)
        
    except Exception as e:
        # 处理可能的错误（如单性别疾病导致的完美分离问题）
        print(f"完整模型拟合失败: {e}, 尝试不含性别变量的模型")
        
        # 不含性别变量的模型
        if use_common:
            formula_full = f'disease ~ Q("{score_name}") + Q("Common_PRS") + Q("21003-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
            formula_null = 'disease ~ Q("21003-0.0") + Q("Common_PRS") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
        else:
            formula_full = f'disease ~ Q("{score_name}") + Q("21003-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
            formula_null = 'disease ~ Q("21003-0.0") + ' + ' + '.join([f'Q("22009-0.{i}")' for i in range(1, 11)])
        
        full_model = smf.logit(formula_full, data=final_data).fit(disp=False)
        restricted_model = smf.logit(formula_null, data=final_data).fit(disp=False)

    # formula_full = f'disease ~ Q("{score_name}") + Q("Common_PRS")'
    # formula_null = 'disease ~ Q("Common_PRS")'
    
    # full_model = smf.logit(formula_full, data=final_data).fit(disp=False)
    # restricted_model = smf.logit(formula_null, data=final_data).fit(disp=False)


    # 提取似然值
    L_res = restricted_model.llf  # 限制模型的对数似然值
    L_full = full_model.llf       # 完整模型的似然值
    N = full_model.nobs           # 样本量
    
    # 计算 R²
    R2 = 1 - np.exp((L_res - L_full) * 2 / N)
    
    # 计算 R²_max
    R2_max = 1 - np.exp(L_res * 2 / N)
    
    # 计算 Nagelkerke R²
    R2_nag = R2 / R2_max if R2_max != 0 else 0
    
    # 计算McFadden's R² (statsmodels默认提供的)
    mcfadden_r2 = full_model.prsquared
    
    return {
        'L_res': L_res,
        'L_full': L_full,
        'N': N,
        'R2': R2,
        'R2_max': R2_max,
        'R2_nag': R2_nag,
        'mcfadden_R2': mcfadden_r2
    }



"""
IDI: Integrated Discrimination Improvement
"""
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

def calculate_IDI_with_inference(df, y_col="disease", pred_old_col="pred_old", pred_new_col="pred_new"):
    """
    计算IDI及其p值，严格按照Pencina et al. (2008)公式(15)
    
    参数:
    df: 包含预测值和真实标签的DataFrame
    y_col: 真实标签列名
    pred_old_col: 旧模型预测概率列名
    pred_new_col: 新模型预测概率列名
    
    返回:
    IDI值, p值, 置信区间
    """
    # 分离病例和对照
    events = df[df[y_col] == 1].copy()
    non_events = df[df[y_col] == 0].copy()
    
    n_events = len(events)
    n_non_events = len(non_events)
    
    # ========== 公式(13)：计算IDI点估计 ==========
    # P_new,events - P_old,events
    event_up = events[pred_new_col].mean() - events[pred_old_col].mean()
    
    # P_old,nonevents - P_new,nonevents
    non_event_down = non_events[pred_old_col].mean() - non_events[pred_new_col].mean()
    
    # IDI = (P_new,events - P_old,events) - (P_new,nonevents - P_old,nonevents)
    IDI = event_up + non_event_down
    
    # ========== 公式(15)：计算标准误和z检验 ==========
    # 计算 S_events（病例中配对差异的标准误）
    if n_events > 1:
        # 每个病例的新旧模型预测概率差
        event_differences = events[pred_new_col].values - events[pred_old_col].values
        # 论文中的 S_events = 配对差异的标准误
        # 标准误 = 标准差 / sqrt(n)
        s_events = np.std(event_differences, ddof=1) / np.sqrt(n_events)
    else:
        s_events = 0
    
    # 计算 S_nonevents（对照中配对差异的标准误）
    if n_non_events > 1:
        # 每个对照的新旧模型预测概率差（注意：公式中是P_old - P_new）
        non_event_differences = non_events[pred_old_col].values - non_events[pred_new_col].values
        s_nonevents = np.std(non_event_differences, ddof=1) / np.sqrt(n_non_events)
    else:
        s_nonevents = 0
    
    # IDI的标准误（假设病例和对照独立）
    se_idi = np.sqrt(s_events**2 + s_nonevents**2)
    
    # z统计量（公式15）
    if se_idi > 0:
        z_score = IDI / se_idi
        # 双侧p值（使用标准正态分布，不是t分布！）
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    else:
        z_score = 0
        p_value = 1.0
    
    # ========== 95%置信区间（正态近似） ==========
    if se_idi > 0:
        ci_lower = IDI - 1.96 * se_idi
        ci_upper = IDI + 1.96 * se_idi
    else:
        ci_lower = ci_upper = IDI
    
    # ========== 各组成部分的检验 ==========
    # 病例部分（event_up）的检验
    if s_events > 0:
        z_event = event_up / s_events
        p_event = 2 * (1 - stats.norm.cdf(abs(z_event)))
    else:
        z_event = 0
        p_event = 1.0
    
    # 对照部分（non_event_down）的检验
    if s_nonevents > 0:
        z_non_event = non_event_down / s_nonevents
        p_non_event = 2 * (1 - stats.norm.cdf(abs(z_non_event)))
    else:
        z_non_event = 0
        p_non_event = 1.0
    
    
    result = {
        # IDI点估计
        "IDI": IDI,
        "IDI_event_up": event_up,
        "IDI_non_event_down": non_event_down,
        
        # 标准误和检验统计量
        "IDI_se": se_idi,
        "z_score": z_score,
        
        # p值
        "IDI_p_value": p_value,           # IDI整体p值
        "event_up_p_value": p_event,      # 病例部分p值
        "non_event_down_p_value": p_non_event,  # 对照部分p值
        
        # 置信区间
        "IDI_CI_lower": ci_lower,
        "IDI_CI_upper": ci_upper,
        
        # 详细信息（用于调试）
        "s_events": s_events,
        "s_nonevents": s_nonevents,
        "n_events": n_events,
        "n_non_events": n_non_events,
        
        # 公式(14)的验证
        "discrimination_slope_new": events[pred_new_col].mean() - non_events[pred_new_col].mean(),
        "discrimination_slope_old": events[pred_old_col].mean() - non_events[pred_old_col].mean(),
        "IDI_slope": (events[pred_new_col].mean() - non_events[pred_new_col].mean()) - 
                     (events[pred_old_col].mean() - non_events[pred_old_col].mean())
    }
    
    return result


def calculate_IDI_value(res_df, use_common=False, thre=1):
    """
    use_common: True for rvPRS and False for cvPRS
    thre: threshold for extreme-risk individuals
    """

    # 构建逻辑回归模型计算预测概率
    # 标准化PRS
    res_df["Common_PRS_std"] = (res_df["Common_PRS"] - res_df["Common_PRS"].mean()) / res_df["Common_PRS"].std()
    res_df["Rare_PRS_std"] = (res_df["Probability"] - res_df["Probability"].mean()) / res_df["Probability"].std()
    
    y_col = "disease"
    y = res_df[y_col].values
    

    # 使用statsmodels进行逻辑回归
    try:
        cov_ls = ["21003-0.0", "31-0.0"] + [f"22009-0.{i}" for i in range(1, 11)]
        # 尝试包含性别变量的完整模型
        if use_common:
            # 旧模型：Ccovariat-only
            X_old = sm.add_constant(res_df[cov_ls].values)
            model_old = sm.Logit(y, X_old).fit(disp=0, method='lbfgs', maxiter=1000)
            pred_old = model_old.predict(X_old)
            # 新模型：Common PRS + covariat
            X_new = sm.add_constant(res_df[["Common_PRS_std"]+cov_ls].values)
            model_new = sm.Logit(y, X_new).fit(disp=0, method='lbfgs', maxiter=1000)
            pred_new = model_new.predict(X_new)
        
        else:
            # 旧模型：Common PRS + covariat
            X_old = sm.add_constant(res_df[["Common_PRS_std"]+cov_ls].values)
            model_old = sm.Logit(y, X_old).fit(disp=0, method='lbfgs', maxiter=1000)
            pred_old = model_old.predict(X_old)
            # 新模型：Common + Rare PRS + covariat
            X_new = sm.add_constant(res_df[["Common_PRS_std", "Rare_PRS_std"]+cov_ls].values)
            model_new = sm.Logit(y, X_new).fit(disp=0, method='lbfgs', maxiter=1000)
            pred_new = model_new.predict(X_new)
    
    except:
        cov_ls = ["21003-0.0"] + [f"22009-0.{i}" for i in range(1, 11)]
        # 移除性别变量的模型
        if use_common:
            # 旧模型：Covariat-only
            X_old = sm.add_constant(res_df[cov_ls].values)
            model_old = sm.Logit(y, X_old).fit(disp=0, method='lbfgs', maxiter=1000)
            pred_old = model_old.predict(X_old)
            # 新模型：Common PRS + covariat
            X_new = sm.add_constant(res_df[["Common_PRS_std"]+cov_ls].values)
            model_new = sm.Logit(y, X_new).fit(disp=0, method='lbfgs', maxiter=1000)
            pred_new = model_new.predict(X_new)
        
        else:
            # 旧模型：Common PRS + covariat
            X_old = sm.add_constant(res_df[["Common_PRS_std"]+cov_ls].values)
            model_old = sm.Logit(y, X_old).fit(disp=0, method='lbfgs', maxiter=1000)
            pred_old = model_old.predict(X_old)
            # 新模型：Common + Rare PRS + covariat
            X_new = sm.add_constant(res_df[["Common_PRS_std", "Rare_PRS_std"]+cov_ls].values)
            model_new = sm.Logit(y, X_new).fit(disp=0, method='lbfgs', maxiter=1000)
            pred_new = model_new.predict(X_new)


    # 将预测概率添加到DataFrame
    res_df["pred_old"] = pred_old
    res_df["pred_new"] = pred_new
    
    # 计算IDI
    idi_results = calculate_IDI_with_inference(
        df=res_df,
        y_col=y_col,
        pred_old_col="pred_old",
        pred_new_col="pred_new"
    )
        
    return idi_results





"""
OR: Odds Ratio
"""
import statsmodels.api as sm
from scipy import stats


def calculate_OR(X, Y, cov=None, cov_nosex=None, OR_thre=1):
    """
    执行逻辑回归并计算OR和置信区间
    X: rare PRS，可以回归协变量(common PRS)；不回归时即为默认的(a/b)/(c/d)
    Y: 事件是否发生
    """
    X = X.flatten()

    ## 判断是否位于PRS前1%
    top_1_percent = np.percentile(X, 100-OR_thre)  # 前1%的阈值
    binary_label = np.where(X > top_1_percent, 1,  0)

    # 处理重复值
    need_from_threshold = int(len(X)*OR_thre/100) - np.sum(binary_label)
    if need_from_threshold >= OR_thre:
        at_threshold_indices = np.where((X == top_1_percent))[0]
        selected_from_threshold = np.random.choice(
            at_threshold_indices, 
            size=need_from_threshold, 
            replace=False
        )
        binary_label[selected_from_threshold] = 1


    if cov is None:
        X = binary_label
    else:
        X = np.hstack([np.expand_dims(binary_label, axis=1), cov])

    X = sm.add_constant(X)  # 添加常数项
    try:
        # 拟合逻辑回归模型
        model = sm.GLM(Y, X, family=sm.families.Binomial())
        results = model.fit()
    except:
        ## 去除sex作为协变量
        X = np.hstack([np.expand_dims(binary_label, axis=1), cov_nosex])
        X = sm.add_constant(X) 

        model = sm.GLM(Y, X, family=sm.families.Binomial())
        results = model.fit()
    
    sample_rate = np.sum(binary_label)/len(binary_label)
    if sample_rate<0.005:
        ## 放弃当前表型
        OR=1
        p_value = 1
    else:
        # 使用summary_frame获取置信区间
        conf_int = results.conf_int(alpha=0.05)  # 95% CI
        OR = np.exp(results.params[1])
        CI_lower = np.exp(conf_int[1, 0])
        CI_upper = np.exp(conf_int[1, 1])

        # 计算p值（Wald检验）
        beta = results.params[1]  # log(OR)
        se_beta = results.bse[1]
        z_statistic = beta / se_beta
        p_value = 2 * stats.norm.sf(np.abs(z_statistic))
    
    return OR, (CI_lower, CI_upper), p_value



def calculate_OR_value(res_df, thre=1, OR_thre=1):
    """
    PRS前1(thre=1)% vs 剩余99%患病率(位于Top 1%)比例

    计算OR_combined时：使用training set权重生成combined PRS
    """    
    ## 加载test
    binary_label = res_df["disease"].values
    rare_pred = res_df["Probability"].values
    common_PRS = np.expand_dims(res_df["Common_PRS"].values, axis=1) 
    
    # OR for common PRS
    cov_ls = ["21003-0.0", "31-0.0"] + [f"22009-0.{i}" for i in range(1, 11)]
    cov = res_df[cov_ls].values
    cov_nosex = res_df[["21003-0.0"] + [f"22009-0.{i}" for i in range(1, 11)]].values
    OR_common, (CI_lower_common, CI_upper_common), p_value_common = calculate_OR(common_PRS, binary_label.flatten(), OR_thre=OR_thre, cov=cov, cov_nosex=cov_nosex)
    
    # OR for rare PRS
    cov_ls = ['Common_PRS', "21003-0.0", "31-0.0"] + [f"22009-0.{i}" for i in range(1, 11)]
    cov = res_df[cov_ls].values
    cov_nosex = res_df[['Common_PRS', "21003-0.0"] + [f"22009-0.{i}" for i in range(1, 11)]].values
    OR, (CI_lower, CI_upper), p_value = calculate_OR(rare_pred, binary_label.flatten(), OR_thre=OR_thre, cov=np.concatenate([cov, common_PRS], axis=1), cov_nosex=np.concatenate([cov_nosex, common_PRS], axis=1))

    return OR, (CI_lower, CI_upper), p_value, OR_common, (CI_lower_common, CI_upper_common), p_value_common




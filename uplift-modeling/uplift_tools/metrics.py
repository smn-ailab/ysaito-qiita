from operator import itemgetter

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from plotly.graph_objs import Bar, Box, Figure, Histogram, Layout, Scatter
from plotly.offline import init_notebook_mode, iplot, plot


def uplift_frame_reg(outcome, treat, score):
    """create a DataFrame that is used to evaluate uplift-model for regression settongs

    Parameters

    ----------
    outcome : integer
      The Target Variable List for each suject in the test data

    treat : bool
      Whether a sunject is in the treatment group or note

    score : integer
      The "Uplift-Score" for each subject

    ----------

    """

    # sorting by uplift socre
    result = list(zip(outcome, treat, score))
    result.sort(key=itemgetter(2), reverse=True)

    # initializing
    treat_uu = 0  # the number of subjects in the treatment group at each step
    control_uu = 0  # the number of subjects in the control group at each step
    y_treat = 0  # the sum of the outcome variable in the treatment group at each step
    y_control = 0  # the sum of the outcome variable in the control group at each step
    y_treat_avg = 0  # the average of the outcome variable in the treatment group at each step
    y_control_avg = 0  # the average of the outcome variable in the control group at each step
    lift = 0.0  # the calicurated "uplift" at each step

    stat_data = []

    # calc num of subjects, total outcome, avg outcome
    for y, is_treat, score in result:
        if is_treat:
            treat_uu += 1
            y_treat += y
            y_treat_avg = y_treat / treat_uu
        else:
            control_uu += 1
            y_control += y
            y_control_avg = y_control / control_uu

        # calc "lift" at each step
        lift = (y_treat_avg - y_control_avg) * treat_uu

        stat_data.append([y, is_treat, score, treat_uu, control_uu,
                          y_treat, y_control, y_treat_avg, y_control_avg, lift])

    # convert stat_data to DataFrame
    df = DataFrame(stat_data)
    df.columns = [["y", "is_treat", "score", "treat_uu", "control_uu",
                   "y_treat", "y_control", "y_treat_avg", "y_control_avg", "lift"]]

    # calc base_line at each step
    df["base_line"] = df.index * df.loc[len(df.index) - 1, "lift"].values[0] / len(df.index)

    return df


def uplift_frame_clf(outcome, treat, score):
    """create a DataFrame that is used to evaluate uplift-model for classification settongs

    Parameters

    ----------
    outcome : bool
      The Target Variable List for each suject in the test data

    treat : bool
      Whether a sunject is in the treatment group or not

    score : integer
      The "Uplift-Score" for each subject

    ----------

    """

    # sorting by uplift socre
    result = list(zip(outcome, treat, score))
    result.sort(key=itemgetter(2), reverse=True)

    # initializing
    treat_uu = 0  # the number of subjects in the treatment group at each step
    control_uu = 0  # the number of subjects in the control group at each step
    y_treat = 0  # the total of the outcome variable in the treatment group at each step
    y_control = 0  # the total of the outcome variable in the control group at each step
    y_treat_avg = 0  # the probability of the outcome variable in the treatment group at each step
    y_control_avg = 0  # the probability of the outcome variable in the control group at each step
    lift = 0.0  # the calicurated "uplift" at each step

    stat_data = []

    # calc num of subjects, total outcome, rate outcome
    for y, is_treat, score in result:
        if is_treat:
            treat_uu += 1
            if y:
                y_treat += 1
            y_treat_avg = y_treat / treat_uu
        else:
            control_uu += 1
            if y:
                y_control += 1
            y_control_avg = y_control / control_uu

        # calc "lift" at each step
        lift = (y_treat_avg - y_control_avg) * treat_uu

        stat_data.append([y, is_treat, score, treat_uu, control_uu,
                          y_treat, y_control, y_treat_avg, y_control_avg, lift])

    # convert stat_data to DataFrame
    df = DataFrame(stat_data)
    df.columns = [["y", "is_treat", "score", "treat_uu", "control_uu",
                   "y_treat", "y_control", "y_treat_avg", "y_control_avg", "lift"]]

    # calc base_line at each step
    df["base_line"] = df.index * df.loc[len(df.index) - 1, "lift"].values[0] / len(df.index)

    return df


def uplift_bar(outcome, treat, score, task="classification"):

    # sorting by uplift socre
    test_list = list(zip(outcome, treat, score))
    test_list.sort(key=itemgetter(2), reverse=True)
    qdf = DataFrame(columns=("y_treat", "y_control"))

    # sort by uplift-score and divid into 10 groups
    for n in range(10):
        start = int(n * len(test_list) / 10)
        end = int((n + 1) * len(test_list) / 10) - 1
        quantiled_result = test_list[start:end]

        # count the num of subjects in treatment and control group at each decile
        treat_uu = list(map(lambda item: item[1], quantiled_result)).count(True)
        control_uu = list(map(lambda item: item[1], quantiled_result)).count(False)

        # calc the avg outcome for treatment and control group at each decile
        if task == "classification":
            treat_cv_list = []
            control_cv_list = []

            for item in quantiled_result:
                if item[1]:
                    treat_cv_list.append(item[0])
                else:
                    control_cv_list.append(item[0])

            treat_cv = treat_cv_list.count(True)
            control_cv = control_cv_list.count(True)

            y_treat = treat_cv / treat_uu
            y_control = control_cv / control_uu

        elif task == "regression":
            y_treat_list = []
            y_control_list = []

            for item in quantiled_result:
                if item[1]:
                    y_treat_list.append(item[0])
                else:
                    y_control_list.append(item[0])

            y_treat = mean(y_treat_list)
            y_control = mean(y_control_list)

        label = "{}%~{}%".format(n * 10, (n + 1) * 10)
        qdf.loc[label] = [y_treat, y_control]

    trace1 = Bar(x=qdf.index.tolist(),
                 y=qdf.y_treat.values.tolist(), name="treat")
    trace2 = Bar(x=qdf.index.tolist(),
                 y=qdf.y_control.values.tolist(), name="control")

    layout = Layout(barmode="group", yaxis={"title": "Mean Outcome"}, xaxis={"title": "Uplift Score Percentile"})
    fig = Figure(data=[trace1, trace2], layout=layout)
    iplot(fig)

    # calc Area Under Uplift Curve
    auuc = round(((np.array(df.lift) - np.array(df.base_line)).sum()) / len(df.lift), 3)

    # Total Outcome sorted by Uplift Rank
    trace1 = Scatter(x=np.arange(df.shape[0]), y=df.y_treat.T.values[0], name="treat")
    trace2 = Scatter(x=np.arange(df.shape[0]), y=df.y_control.T.values[0], name="control")
    data = [trace1, trace2]b
    layout = Layout(title="AUUC = {}".format(auuc), yaxis={"title": "Total Outcome"}, xaxis={"title": "{} Rank".format(score_name)})
    fig = Figure(data=data, layout=layout)
    iplot(fig)

    # Avg Outcome sorted by Uplift Rank
    trace1 = Scatter(x=np.arange(df.shape[0]), y=df.y_treat_avg.T.values[0], name="treat")
    trace2 = Scatter(x=np.arange(df.shape[0]), y=df.y_control_avg.T.values[0], name="control")
    data = [trace1, trace2]
    layout = Layout(title="AUUC = {}".format(auuc), yaxis={"title": "Mean Outcome"}, xaxis={"title": "{} Rank".format(score_name)})
    fig = Figure(data=data, layout=layout)
    iplot(fig)

    # Lift Curve sorted by Uplift Rank
    trace1 = Scatter(x=np.arange(df.shape[0]),
                     y=df.lift.T.values[0],
                     name="treat")
    trace2 = Scatter(x=np.arange(df.shape[0]), y=df.base_line.T.values[0], name="baseline")
    data = [trace1, trace2]
    layout = Layout(title="AUUC = {}".format(auuc), yaxis={"title": "Uplift"}, xaxis={"title": "{} Rank".format(score_name)})
    fig = Figure(data=data, layout=layout)
    iplot(fig)

    # Lift Curve sorted by Uplift Score
    trace1 = Scatter(x=df.score.T.values[0], y=df.lift.T.values[0], name="treat")
    trace2 = Scatter(x=df.score.T.values[0], y=df.base_line.T.values[0], name="baseline")
    data = [trace1, trace2]
    layout = Layout(title="AUUC = {}".format(auuc), yaxis={"title": "{}-Score".format(score_name)},
                    xaxis={"title": "{}".format(score_name), "autorange": "reversed"})
    fig = Figure(data=data, layout=layout)
    iplot(fig)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import seaborn as sns
import plotly.graph_objects as go
import numpy as np

def _read_csv(filename):
    fl_results_df=pd.read_csv(filename, sep=";", header=None)
    fl_results_df.columns=["fl_round","client_id","learning_rate","direction","mse","best_mse","mae","r2","lookback","num_epochs","hidden_size","batch_size"]
    return fl_results_df

def draw_plot(filename: str, save_fig=False):
    fl_results_df = _read_csv(filename)

    plt.figure(figsize=(10,5))
    sns.lineplot(fl_results_df,x="fl_round",y="mse_cluster_0",hue="K")
    plt.yscale("log")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("FL Round")

    plt.legend()
    if save_fig:
        plt.savefig("start_with_c0_baseline_test_on_c0.png",dpi=200,format="png")

    plt.figure(figsize=(10,5))
    sns.lineplot(fl_results_df,x="fl_round",y="mse_cluster_5",hue="K")
    plt.yscale("log")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("FL Round")
    plt.legend()
    if save_fig:
        plt.savefig("start_with_c5_baseline_test_on_c5.png",dpi=200,format="png")

def preprocessing_results(filenames, client_id = None, mse_column='mse_cluster_0'):
    if client_id != None:
        n_rounds, y = __preprocessing_results_a_client(filenames, client_id, mse_column)
    else:
        n_rounds, y = __preprocessing_results_all_clients(filenames, mse_column)
    return n_rounds, y

def __preprocessing_results_all_clients(filenames, mse_column):
    dfs,y = [], []
    for filename in filenames:
        if filename != None:
            dfs.append(_read_csv(filename))
        else:
            dfs.append(pd.DataFrame())

    for index, df in enumerate(dfs):
        if not df.empty:
            n_rounds = len(df.fl_round.unique())
            df_group = df[['fl_round', mse_column]].groupby(by=['fl_round']).mean()
            df_group = pd.DataFrame(df_group.reset_index())
            y.append(df_group[mse_column].values)
        else:
            y.append([])

    return n_rounds, y

def preprocessing_results_all_clients2(filename1, filename2, filename3, filename4, n_clients, mse_column):
    df = _read_csv(filename1)  if filename1 != None else pd.DataFrame()
    df1 = _read_csv(filename2) if filename2 != None else pd.DataFrame()
    df2 = _read_csv(filename3) if filename3 != None else pd.DataFrame()
    df3 = _read_csv(filename4) if filename4 != None else pd.DataFrame()
    if filename1 != None:
        n_rounds = len(df.fl_round.unique())
    elif filename2 != None:
        n_rounds = len(df1.fl_round.unique())
    elif filename3 != None:
        n_rounds = len(df2.fl_round.unique())
    elif filename4 != None:
        n_rounds = len(df3.fl_round.unique())
    else:
        n_rounds = len(df.fl_round.unique())

    y_df_list = []
    y_df1_list = []
    y_df2_list = []
    y_df3_list = []

    for client in range(n_clients):
        y_df_list.append(df[df['client_id']==client][mse_column].values) if not df.empty else []
        y_df1_list.append(df1[df1['client_id']==client][mse_column].values) if not df1.empty else []
        y_df2_list.append(df2[df2['client_id']==client][mse_column].values) if not df2.empty else []
        y_df3_list.append(df3[df3['client_id']==client][mse_column].values) if not df3.empty else []
   
    return n_rounds, y_df_list, y_df1_list, y_df2_list, y_df3_list

def plot_plotly(n_rounds, y, 
                title='Mean Squared Error over FL Rounds', 
                y_axis_title='Mean Squared Error', 
                y_axis_min=0, 
                y_axis_max=1000,
                algo_name1 = 'FedCluLearn',
                # algo_name1 = 'FedCluLearn-totals',
                # algo_name2 = 'FedCluLearn-recent',
                # algo_name3 = 'FedCluLearn-percentage',  
                algo_name4='FedAvg', 
                algo_name5='FedAtt', 
                algo_name6='FedProx',
                algo_name7='FedCluLearn-Prox',
                # algo_name7='FedCluLearn-Prox-totals',
                # algo_name8='FedCluLearn-Prox-recent',
                # algo_name9='FedCluLearn-Prox-percentage',
                name='plot_name'
            ):
    # Create a figure
    fig = go.Figure()
    x = np.arange(0, n_rounds+1)

    # Add traces for both algorithms with different line styles
  
    fig.add_trace(go.Scatter(x=x, y=y[0], mode='lines', name=algo_name1, line=dict(dash='solid', width=4)))
    # fig.add_trace(go.Scatter(x=x, y=y[1], mode='lines', name=algo_name2, line=dict(dash='solid')))
    # fig.add_trace(go.Scatter(x=x, y=y[2], mode='lines', name=algo_name3, line=dict(dash='solid')))
    fig.add_trace(go.Scatter(x=x, y=y[1], mode='lines', name=algo_name4, line=dict(dash='dash', width=4)))
    fig.add_trace(go.Scatter(x=x, y=y[2], mode='lines', name=algo_name5, line=dict(dash='dot', width=4)))
    fig.add_trace(go.Scatter(x=x, y=y[3], mode='lines', name=algo_name6, line=dict(dash='dashdot', width=4)))
    fig.add_trace(go.Scatter(x=x, y=y[4], mode='lines', name=algo_name7, line=dict(dash='solid', width=4)))
    # fig.add_trace(go.Scatter(x=x, y=y[7], mode='lines', name=algo_name8, line=dict(dash='solid')))
    # fig.add_trace(go.Scatter(x=x, y=y[8], mode='lines', name=algo_name9, line=dict(dash='solid')))


    # Update layout for dropdowns and interactivity
    fig.update_layout(
        title=title,
        xaxis_title="FL Rounds",
        yaxis_title=y_axis_title,
        # yaxis_range=[y_axis_min,y_axis_max],
        width=1200,
        height=600,
        legend=dict(
            orientation="h",  # Horizontal legend
            x=0.5,            # Center the legend horizontally
            y=1.05,           # Move the legend above the plot
            xanchor="center", # Center align the legend
            yanchor="bottom"  # Ensure the bottom of the legend aligns with y=1.15
        ),
        template='simple_white'
    )

    # fig.update_layout(
    #     title=title,
    #     xaxis_title="FL Rounds",
    #     yaxis_title=y_axis_title,
    #     # yaxis_range=[y_axis_min,y_axis_max],
    #     # width=600,
    #     # height=450,
    #     width=600,
    #     height=450,
    #     showlegend=True,
    #     legend=dict(
    #         font=dict(size=30),
    #         orientation="h",  # Horizontal legend
    #         traceorder="normal",
    #         x=0.1,            # Center the legend horizontally
    #         y=1.15,           # Move the legend above the plot
    #         xanchor="left", # Center align the legend
    #         yanchor="top",  # Ensure the bottom of the legend aligns with y=1.15
    #         tracegroupgap=5,
    #     ),
    #     xaxis=dict(
    #         titlefont=dict(size=30),  # X-axis label font size
    #         tickfont=dict(size=30),  # X-axis tick labels font size
    #     ),
    #     yaxis=dict(
    #         titlefont=dict(size=30),  # Y-axis label font size
    #         tickfont=dict(size=30)  # Y-axis tick labels font size
    #     ),
    #     template='simple_white',
    #     margin=dict(l=5, r=5, t=5, b=5),
    # )

    # pio.write_image(fig, f"{name}.pdf", format="pdf", scale=6)
    # Show the plot
    fig.show(render_mode='png')


def plot_plotly_real(n_rounds, y, 
                title='Mean Squared Error over FL Rounds', 
                y_axis_title='Mean Squared Error', 
                y_axis_min=0, 
                y_axis_max=1000,
                algo_name1 = 'totals',
                algo_name2 = 'recent',
                algo_name3 = '50%',  
                algo_name4='FedAvg', 
                algo_name5='FedAtt', 
                algo_name6='FedProx',
                algo_name7='totals',
                algo_name8='recent',
                algo_name9='50%',
                algo_name10 = '25%',
                algo_name11 = '75%',
                algo_name12='25%',
                algo_name13='75%',
                name='plot_name'
            ):
    # Create a figure
    fig = go.Figure()
    x = np.arange(0, n_rounds+1)

    # Add traces for both algorithms with different line styles
    fig.add_trace(go.Scatter(x=x, y=y[0], mode='lines', name=algo_name1, line=dict(dash='solid', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[1], mode='lines', name=algo_name2, line=dict(dash='solid', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[2], mode='lines', name=algo_name3, line=dict(dash='solid', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[3], mode='lines', name=algo_name4, line=dict(dash='dash', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[4], mode='lines', name=algo_name5, line=dict(dash='dot', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[5], mode='lines', name=algo_name6, line=dict(dash='dashdot', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[6], mode='lines', name=algo_name7, line=dict(dash='solid', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[7], mode='lines', name=algo_name8, line=dict(dash='solid', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[8], mode='lines', name=algo_name9, line=dict(dash='solid', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[9], mode='lines', name=algo_name10, line=dict(dash='solid', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[10], mode='lines', name=algo_name11, line=dict(dash='solid', color='brown', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[11], mode='lines', name=algo_name12, line=dict(dash='solid', width=3)))
    fig.add_trace(go.Scatter(x=x, y=y[12], mode='lines', name=algo_name13, line=dict(dash='solid', width=3)))


    # Update layout for dropdowns and interactivity
    fig.update_layout(
        title=title,
        xaxis_title="FL Rounds",
        yaxis_title=y_axis_title,
        # yaxis_range=[y_axis_min,y_axis_max],
        width=1200,
        height=600,
        legend=dict(
            orientation="h",  # Horizontal legend
            x=0.5,            # Center the legend horizontally
            y=0.8,           # Move the legend above the plot
            xanchor="center", # Center align the legend
            yanchor="bottom"  # Ensure the bottom of the legend aligns with y=1.15
        ),
        template='simple_white'
    )

    # fig.update_layout(
    #     title=title,
    #     xaxis_title="FL Rounds",
    #     yaxis_title=y_axis_title,
    #     # yaxis_range=[y_axis_min,y_axis_max],
    #     width=600,
    #     height=450,
    #     legend=dict(
    #         font=dict(size=25),
    #         orientation="h",  # Horizontal legend
    #         x=0.5,            # Center the legend horizontally
    #         y=0.9,           # Move the legend above the plot
    #         xanchor="center", # Center align the legend
    #         yanchor="bottom"  # Ensure the bottom of the legend aligns with y=1.15
    #     ),
    #     xaxis=dict(
    #         titlefont=dict(size=30),  # X-axis label font size
    #         tickfont=dict(size=30)  # X-axis tick labels font size
    #     ),
    #     yaxis=dict(
    #         titlefont=dict(size=30),  # Y-axis label font size
    #         tickfont=dict(size=30)  # Y-axis tick labels font size
    #     ),
    #     template='simple_white',
    #     margin=dict(l=5, r=5, t=5, b=5),
    # )

    # pio.write_image(fig, f"{name}.pdf", format="pdf", scale=6)
    # Show the plot
    fig.show(render_mode='png')

def plot_plotly2(n_rounds, y_df_list, y_df1_list, y_df2_list, n_clients = 3,title='Mean Squared Error over FL Rounds', y_axis_title='Mean Squared Error', algo_name1='FedCluLearn', algo_name2='FedAvg', algo_name3='FedAtt'):
    # Create a figure
    fig = go.Figure()
    x = np.arange(0, n_rounds+1)

    # Add traces for both algorithms with different line styles
    for y1, y2, y3, client_id in zip(y_df_list, y_df1_list, y_df2_list, range(n_clients)):
        fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=f'{algo_name1} Client {client_id}', line=dict(dash='solid')))
        fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name=f'{algo_name2} Client {client_id}', line=dict(dash='dash')))
        fig.add_trace(go.Scatter(x=x, y=y3, mode='lines', name=f'{algo_name3} Client {client_id}', line=dict(dash='dot')))

    # Update layout for dropdowns and interactivity
    fig.update_layout(
        title=title,
        xaxis_title="FL Round",
        yaxis_title=y_axis_title,
        width=1200,
        height=600,
        legend=dict(
            orientation="v",
            xanchor="right",
            yanchor="middle",
        ),
        template='simple_white'
    )

    # Show the plot
    fig.show(render_mode='png')

def plot_plotly3(n_rounds, y_df_list, y_df1_list, y_df2_list, y_df3_list, n_clients = 3, model='FedCluLearn', title='Mean Squared Error over FL Rounds', y_axis_title='Mean Squared Error', y_axis_min=0, y_axis_max=1000):
    # Create a figure
    fig = go.Figure()
    x = np.arange(0, n_rounds+1)

    if model == 'FedCluLearn':
        y_list = y_df_list
    
    elif model == 'FedAvg':
        y_list = y_df1_list
    elif model == 'FedAtt':
        y_list = y_df2_list
    elif model == 'FedProx':
        y_list = y_df3_list

    # Add traces for both algorithms with different line styles
    for y1, client_id in zip(y_list, range(n_clients)):
        fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name=f'{model} Client {client_id}', line=dict(dash='solid')))

    # Update layout for dropdowns and interactivity
    fig.update_layout(
        # title=f'{title} - {model}',
        xaxis_title="FL Round",
        yaxis_title=y_axis_title,
        # yaxis_range=[y_axis_min,y_axis_max],
        width=1200,
        height=600,
        legend=dict(
            orientation="v",
            xanchor="right",
            yanchor="middle",
        ),
        template='simple_white'
    )

    # Show the plot
    fig.show(render_mode='png')


def __preprocessing_results_a_client(filenames, client_id=0, mse_column='mse_cluster_0'):
    dfs,y = [], []
    for filename in filenames:
        if filename != None:
            dfs.append(_read_csv(filename))
        else:
            dfs.append(pd.DataFrame())
    n_rounds = len(dfs[0].fl_round.unique())

    for index, df in enumerate(dfs):
        if not df.empty:
            dfs[index] = df[['fl_round', 'client_id', mse_column]]
            y.append(df[df['client_id'] == client_id][mse_column].values)
        else:
            y.append([])
  
    return n_rounds, y
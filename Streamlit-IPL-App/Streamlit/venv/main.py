import streamlit as st 
import pandas as pd 
import numpy as np 
import altair as alt 
import plotly.express as px


ipl = pd.read_csv('Streamlit-IPL-App\Streamlit\venv\Files\IPL_Matches_2008_2022.csv')
l = list(ipl.Season.unique())
batter = pd.read_csv('Streamlit-IPL-App\Streamlit\venv\Files\ipl_deliveries.csv')
ball = pd.read_csv('Streamlit-IPL-App\Streamlit\venv\Files\IPL_Ball_by_Ball_2008_2022.csv')
l = pd.Series(l)

def matches_played(team, df):
    return df[(df['Team1'] == team) | (df['Team2'] == team)].shape[0]

def winning_team(team, df):
    return df[df['WinningTeam'] == team].shape[0]

def matches_no_result(team, df):
    return df[((df['Team1'] == team) | (df['Team2'] == team)) & (df['WonBy'] == 'NoResults')].shape[0]

def highlight_winner_runner(val):
    if val == 'Winner' or val == 'Runner' or val == 'Third' or val == 'Fourth':
        return 'color: black; background-color: #EC7063' if val == 'Winner' else 'color: black; background-color: #808B96' if val == 'Runner' else 'color: black; background-color: #F1948A' if val == 'Third' else 'color: black; background-color: #ABB2B9'
    else:
        return ''

def season(seasons):
    all_season_data = []

    for i in seasons:
        df = ipl[ipl['Season'].isin([i])]
        new_df = pd.DataFrame()
        new_df['Team'] = df.Team1.unique()
        new_df['Matches Played'] = new_df['Team'].apply(lambda x: matches_played(x, df))
        new_df['Won'] = new_df['Team'].apply(lambda x: winning_team(x, df))
        new_df['Lost'] = new_df['Matches Played'] - new_df['Won']
        new_df['No Result'] = new_df['Team'].apply(lambda x: matches_no_result(x, df))
        new_df['Points'] = new_df['Won'] * 2 + new_df['No Result']
        new_df['Season Position'] = new_df['Points'].rank(ascending=False, method='first').astype('object')
        
        df['LossingTeam'] = pd.concat([df[df.WinningTeam == df.Team1]['Team2'], df[df.WinningTeam == df.Team2]['Team1']])
        
        final = df[df.MatchNumber == 'Final']
        winner = final.WinningTeam.values[0]
        runner = final.LossingTeam.values[0]
        new_df.loc[new_df['Team'] == winner, 'Season Position'] = 'Winner'
        new_df.loc[new_df['Team'] == runner, 'Season Position'] = 'Runner'

        qualifier = df[df.MatchNumber == 'Qualifier 2']
        eliminator = df[df.MatchNumber == 'Eliminator']

        if not qualifier.empty: 
            third = qualifier.LossingTeam.values[0]
            new_df.loc[new_df['Team'] == third, 'Season Position'] = 'Third'

        if not eliminator.empty:
            fourth = eliminator.LossingTeam.values[0]
            new_df.loc[new_df['Team'] == fourth, 'Season Position'] = 'Fourth'
        
        new_df.sort_values('Points', ascending=False, inplace=True)
        new_df.set_index('Team', inplace=True)
        all_season_data.append(new_df)

    if all_season_data:
        result_df = pd.concat(all_season_data, axis=0, keys=seasons)
        styled_result_df = result_df.style.applymap(highlight_winner_runner, subset=['Season Position'])
        return styled_result_df
    else:
        return pd.DataFrame()

def choice(season, team):
    all_season = []

    for i in season:
        df = ipl[ipl['Season'].isin([i])]
        a = df.groupby('Team1')['Team2'].value_counts().reset_index()
        b = df.groupby('Team2')['Team1'].value_counts().reset_index().rename(columns={'Team2': 'Team1', 'Team1': 'Team2'})

        df = pd.concat([a, b], ignore_index=True)
        df = df[df.Team1 == team].drop(columns={'Team1'})
        df = df.rename(columns={'Team2': 'Teams', 'count': 'Matches Played'}).set_index('Teams')
        all_season.append(df)

    all_season = [df.reset_index() for df in all_season]

    if all_season:
        result_df = pd.concat(all_season, axis=0,keys=season)
        return result_df.reset_index().drop(columns={'level_1'}).rename(columns={'level_0':'Season'}).groupby(['Season','Teams']).sum()
    else:
        return pd.DataFrame()

def Players(seasons, team):
    all_seasons = []

    for season in seasons:
        l = set()
        df_team1 = ipl[(ipl['Season'].isin([season])) & (ipl['Team1'] == team)]['Team1Players']
        df_team2 = ipl[(ipl['Season'].isin([season])) & (ipl['Team2'] == team)]['Team2Players']
        
        df = pd.concat([df_team1, df_team2], ignore_index=True)
        
        for i in df:
            player_names = i.strip("[]").split(', ')
            for player in player_names:
                l.add(player.replace("'", ""))
        
        player_list = pd.DataFrame(list(l), columns=['Players'])
        all_seasons.append(player_list)

    result_df = pd.concat(all_seasons, axis=1, keys=seasons).fillna('-')
    return result_df

import pandas as pd

def toss_win(seasons, team):
    all_season = []
    d = {}
    for season in seasons:
        df = ipl[ipl['Season'] == season]
        toss_win = df['TossWinner'].value_counts()
        all_season.append(toss_win)
    
    result_df = pd.concat(all_season, axis=1, keys=seasons).fillna(0)
    df = ipl[ipl['Season'].isin(seasons)]  # Use isin to select multiple seasons
    matches_by_season = df.groupby('Season').apply(lambda x: x[(x['Team1'] == team) | (x['Team2'] == team)].shape[0])
    
    r = result_df.reset_index()
    
    for season in seasons:
        toss = r.loc[r['TossWinner'] == team, str(season)]
        toss_numeric = pd.to_numeric(toss, errors='coerce').fillna(0)
        
        if matches_by_season[season] > 0: 
            per = toss_numeric / matches_by_season[season] * 100
            d[season] = round(float(per.mean()), 2)  # Calculate the mean percentage for each season
        else:
            d[season] = 0.0 
    
    return d


def decision(season, team):
    toss_decision_counts = ipl[ipl.Season == season].groupby(['TossWinner', 'TossDecision'])['MatchNumber'].count().reset_index()
    
    bat_decisions = toss_decision_counts.pivot_table(index='TossWinner', columns='TossDecision', values='MatchNumber', fill_value=0)['bat']
    field_decisions = toss_decision_counts.pivot_table(index='TossWinner', columns='TossDecision', values='MatchNumber', fill_value=0)['field']
    
    if team in bat_decisions.index:
        c_b = bat_decisions.loc[team]
    else:
        c_b = 0
    
    if team in field_decisions.index:
        c_f = field_decisions.loc[team]
    else:
        c_f = 0
    
    b = round((int(c_b) / (int(c_b) + int(c_f))) * 100, 2) if (int(c_b) + int(c_f)) > 0 else 0
    f = round((int(c_f) / (int(c_b) + int(c_f))) * 100, 2) if (int(c_b) + int(c_f)) > 0 else 0
    
    return b, f

def batting(season,team):
    df = batter.merge(ipl[['ID', 'Season']], on='ID')
    df = df[df.batter.isin(Players([season], team)[season].squeeze().tolist())]
    df['isBatsmanBall'] = df.extra_type.apply(lambda x: 1 if x != 'wides' else 0)
    df['isBatsmanOut'] = df.batter == df.player_out
    b_df = df.groupby(['batter', 'ID'], as_index=False)[['batsman_run', 'isBatsmanBall', 'isBatsmanOut']].sum(numeric_only=True)
    innings = b_df.groupby('batter').ID.count()
    if not b_df.empty:
        b_df = b_df.groupby('batter').agg(
            {
                'batsman_run': ['sum', 'max'],
                'isBatsmanBall': 'sum',
                'isBatsmanOut': 'sum'
            })
        b_df['Innings'] = innings
        b_df['TotalRuns'] = b_df[('batsman_run', 'sum')]
        b_df['Avg'] = round(b_df['TotalRuns'] / b_df[('isBatsmanOut', 'sum')],2)
        b_df['Avg'] = b_df['Avg'].replace([np.inf, -np.inf], 0)
        b_df['HighestScore'] = b_df[('batsman_run', 'max')]
        b_df['StrikeRate'] = round((b_df['TotalRuns'] / b_df[('isBatsmanBall', 'sum')]) * 100,2)
        return b_df[['Innings', 'TotalRuns', 'Avg', 'HighestScore', 'StrikeRate']]
    else:
        return pd.DataFrame()

def heatmap(season,team,run):
    df1 = ball.merge(ipl[['ID', 'Season']], on='ID')

    df = df1[(df1['ballnumber'].isin([1,2,3,4,5,6]))]
    df = df[df['BattingTeam'] == team]
    df = df[df['batsman_run'] == run]
    df = df[df['Season'] == season]
    df = df.pivot_table(index='ballnumber',columns='overs',values='batsman_run',aggfunc='count').fillna(0)
    
    custom_x_labels = list(range(1, 21))

    fig = px.imshow(df, color_continuous_scale='reds')

    fig.update_layout(
        xaxis=dict(tickvals=list(range(0, len(custom_x_labels))), ticktext=custom_x_labels, title_text="Overs", title=dict(font=dict(color="black"))),
        yaxis=dict(title_text="Ball Number", title=dict(font=dict(color="black"))),
        title='Boundaries Heat Map'
    )

    return fig

def bowler(season,team):

    df = ball.merge(ipl[['ID', 'Season']], on='ID')
    df = df[df.bowler.isin(Players([season],team)[season].squeeze().tolist())]
    df['Bowler Wicket'] = df.kind.apply(lambda x: 1 if x in ['caught',\
    'caught and bowled', 'bowled', 'stumped','lbw', 'hit wicket'] else 0)
    df['Bowler Runs'] = df.extra_type.apply(lambda x: 0 if x in ['legbyes','byes'] else\
    1) * df['total_run']
    df['Legal Ball'] = df.extra_type.apply(lambda x: 0 if x in ['wides','noballs'] else 1)

    df = df.groupby(['bowler'],as_index=False)[['Bowler Wicket','Bowler Runs',\
    'Legal Ball']].sum(numeric_only=True)

    if not df.empty:
        df['Economy'] = df['Bowler Runs']/df['Legal Ball'] * 6
        df = df.sort_values(['Bowler Wicket','Economy'],ascending = [False,True])
        return df
    else:
        return pd.DataFrame()
    
def b_heatmap(season,team):

    df1 = ball.merge(ipl[['ID', 'Season']], on='ID')
    df1 = df1[(df1['ballnumber'].isin([1, 2, 3, 4, 5, 6]))]
    df1['Bowler Wicket'] = df1.kind.apply(lambda x: 1 if x in ['caught', 'caught and bowled', 'bowled', 'stumped', 'lbw', 'hit wicket'] else 0)
    df1 = df1[df1['BattingTeam'] == team]
    df1 = df1[df1['Season'] == season]
    heatmap_data = df1.pivot_table(index='ballnumber', columns='overs', values='Bowler Wicket', aggfunc='sum').fillna(0)

    fig = px.imshow(heatmap_data, color_continuous_scale='reds')

    custom_x_labels = list(range(1, 21))

    fig.update_layout(
        xaxis=dict(tickvals=list(range(0, len(custom_x_labels))), ticktext=custom_x_labels, title_text="Overs", title=dict(font=dict(color="black"))),
        yaxis=dict(title_text="Ball Number", title=dict(font=dict(color="black"))),
        title="Bowler Wickets Heatmap",
    )

    return fig

st.set_page_config(
    page_title='INDIAN PREMIER LEAUGE[2008 - 2022]',
    page_icon="🏆",
    layout='wide',
    initial_sidebar_state='expanded')
alt.themes.enable('dark')

st.title('INDIAN PREMIER LEAUGE[2008 - 2022] :trophy:')

choose = ['Select Team']
for i in sorted(ipl.Team1.unique()):
    choose.append(i)

option = st.selectbox(
        label = "Select Team",
        options = choose)
    
multiselect = st.multiselect(
    label = 'Select Season',
    options = l)

if option != 'Select Team' and len(multiselect)>0:
        st.header(f'{option} Analysis')
        st.subheader('Score Board')

        df = season(multiselect)
        st.dataframe(df,use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader('Matches Played Against')
            if multiselect:
                choice_df = choice(multiselect,option).pivot_table(index='Teams', columns='Season', values='Matches Played', fill_value='-')
                st.dataframe(choice_df,use_container_width=True)
            else:
                st.dataframe(pd.DataFrame())

        with col2:
            st.subheader('Players')
            if multiselect:
                p_df = Players(multiselect, option)
                st.dataframe(p_df,use_container_width=True ,hide_index=True)
            else:
                st.dataframe(pd.DataFrame())
        
        col3, col4, col5, col6 = st.columns(4)
        
        with col3:
            select = st.radio(
                'Select Season:',
                multiselect,
                index=None
            )
        with col4:
            if select:
                st.write('Selection:', select)
                d = toss_win(multiselect, option)
                st.metric('Toss Winning %',f'{d[select]}%')
            else:
                st.write('Selection:', None)

        with col5:
            if select:
                d = decision(select, option)
                st.metric('Toss Decision',f'{d[0]}% to Bat')
            else:
                st.write('Selection:', None)

        with col6:
            if select:
                d = decision(select, option)
                st.metric('Toss Decision',f'{d[1]}% to Field')
            else:
                st.write('Selection:', None)
        
        st.header(f'{option} Batting Analysis')

        if select:
            df = batting(select,option)
            st.dataframe(df,use_container_width=True)

            if not df.empty:
                col = st.color_picker('Select a plot colour (Auto Scale the plot)')
                fig = px.scatter(df, x='Avg', y='StrikeRate', text=df.index, size=df['TotalRuns']*4, hover_name=df.index,title='Batsman Analysis')
                fig.update_traces(textposition='bottom right', textfont=dict(color='black'))
                fig.update_layout(
                    xaxis=dict(range=[df['Avg'].min(), df['Avg'].max()]),
                    yaxis=dict(range=[df['StrikeRate'].min(), df['StrikeRate'].max()])
                )

                avg_mean = df['Avg'].mean()
                fig.add_shape(
                    dict(
                        type='line',
                        x0=avg_mean,
                        x1=avg_mean,
                        y0=df['StrikeRate'].min(),
                        y1=df['StrikeRate'].max(),
                        line=dict(color='red', dash='dash'),
                    )
                )

                strike_rate_mean = df['StrikeRate'].mean()
                fig.add_shape(
                    dict(
                        type='line',
                        x0=df['Avg'].min(),
                        x1=df['Avg'].max(),
                        y0=strike_rate_mean,
                        y1=strike_rate_mean,
                        line=dict(color='red', dash='dash'),
                    )
                )
                
                fig.update_layout(title=dict(x=0.5, y=0.92, xanchor='center', yanchor='top'), showlegend=True)
                fig.update_traces(marker=dict(color=col))

                st.plotly_chart(fig,use_container_width=True)

                st.header('No of Fours & Sixes:')
                
                fas = st.radio(
                    'select',
                    [4,6],
                    index=None
                )
                
                if fas:
                    st.plotly_chart(heatmap(select,option,fas),use_container_width=True)
                else:
                    st.dataframe(pd.DataFrame())
            else:
                st.dataframe(pd.DataFrame())
        else:
            st.dataframe(pd.DataFrame())

        st.header(f'{option} Bowling Analysis')

        if select:
            df = bowler(select,option)
            st.dataframe(df,use_container_width=True)

            if not df.empty:
                fig = px.scatter(df, x='Economy', y='Bowler Wicket', text='bowler', size='Legal Ball', hover_name='bowler',title='Bowler Analysis')
                fig.update_traces(textposition='bottom right', textfont=dict(color='black'))
                fig.update_layout(
                    xaxis=dict(range=[df['Economy'].min(), df['Economy'].max()]),
                    yaxis=dict(range=[df['Bowler Wicket'].min(), df['Bowler Wicket'].max()])
                )

                avg_mean = df['Economy'].mean()
                fig.add_shape(
                    dict(
                        type='line',
                        x0=avg_mean,
                        x1=avg_mean,
                        y0=df['Bowler Wicket'].min(),
                        y1=df['Bowler Wicket'].max(),
                        line=dict(color='red', dash='dash'),
                    )
                )

                strike_rate_mean = df['Bowler Wicket'].mean()
                fig.add_shape(
                    dict(
                        type='line',
                        x0=df['Economy'].min(),
                        x1=df['Economy'].max(),
                        y0=strike_rate_mean,
                        y1=strike_rate_mean,
                        line=dict(color='red', dash='dash'),
                    )
                )
                
                fig.update_layout(title=dict(x=0.5, y=0.92, xanchor='center', yanchor='top'), showlegend=True)
                fig.update_traces(marker=dict(color=col))

                st.plotly_chart(fig,use_container_width=True)

                st.header('No of Wickets:')

                if not df.empty:
                    st.plotly_chart(b_heatmap(select,option),use_container_width=True)
                else:
                    st.dataframe(pd.DataFrame())
                
                col7, col8 = st.columns(2)

                b_df = batting(select, option).reset_index()

                with col7:
                    if not b_df.empty:
                        fig = px.pie(b_df, values='StrikeRate', names='batter',title='Batting Performance Pie Chart - StrikeRate')
                        st.plotly_chart(fig)
                
                with col8:
                    if not b_df.empty:
                        fig2 = px.pie(b_df, values='Avg', names='batter',title='Batting Performance Pie Chart - Avg')
                        st.plotly_chart(fig2)
                
                col9, col10 = st.columns(2)

                b_df = bowler(select, option)

                with col9:
                    if not b_df.empty:
                        fig = px.pie(b_df, values='Economy', names='bowler',title='Bowling Performance Pie Chart - Economy')
                        st.plotly_chart(fig)
                
                with col10:
                    if not b_df.empty:
                        fig2 = px.pie(b_df, values='Bowler Wicket', names='bowler',title='BOwling Performance Pie Chart - Bowler Wicket')
                        st.plotly_chart(fig2)
                

else:
    st.header('Select The Team & The Seasons')

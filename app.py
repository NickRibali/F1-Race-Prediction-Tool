import pandas as pd
import streamlit as st

from model import Predictor, walk_forward

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏁", layout="wide")


@st.cache_data
def load_data():
    return pd.read_csv('f1_clean.csv')


@st.cache_resource
def load_predictor(df):
    return Predictor(df)


@st.cache_data
def load_check(df):
    return walk_forward(df, last_k=5)


data = load_data()
predictor = load_predictor(data)

st.title("🏁 F1 Race Predictor")
st.write("Enter a starting grid and the model predicts the finishing order. "
         "It uses grid position plus each driver's and team's recent results.")

# Start from the most recent race in the data so there is always something to edit
last = data[data['race_key'] == data['race_key'].max()]
default_grid = (last[['driver_abbr', 'team_name', 'grid_position']]
                .sort_values('grid_position').reset_index(drop=True))

st.subheader("1. Starting grid")
st.caption("Edit any cell, or add and remove rows. Driver codes are three letters.")
grid_input = st.data_editor(default_grid, num_rows="dynamic", width="stretch",
                            hide_index=True, key="grid")

st.subheader("2. Predicted finish")
if st.button("Predict finishing order", type="primary"):
    grid, errors, warnings = predictor.check_grid(grid_input)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)
    if grid is not None:
        result = predictor.predict(grid)
        result['places_gained'] = result['grid_position'].astype(int) - result['predicted_finish']
        st.dataframe(
            result.rename(columns={'driver_abbr': 'Driver', 'team_name': 'Team',
                                   'grid_position': 'Starts', 'predicted_finish': 'Predicted finish',
                                   'recent_dnf_rate': 'Recent DNF rate',
                                   'places_gained': 'Places gained'}),
            width="stretch", hide_index=True)
        st.bar_chart(result.set_index('driver_abbr')['places_gained'])
        st.caption("Bars above zero are drivers the model expects to gain places.")

with st.expander("How good is this model?"):
    st.write("For each of the last 5 races in the data, the model was trained only on earlier "
             "races and scored on that race. Lower is better.")
    st.dataframe(load_check(data), width="stretch")
    st.write("The model does not beat the simple rule 'finish where you start' on average miss. "
             "Its advantage is on big misses. Races with crashes and breakdowns are the hardest to predict.")

with st.expander("About the data"):
    st.write(f"Trained on {predictor.n_rows:,} driver results from {predictor.n_races} races. "
             f"The most recent race in the data has the key {predictor.latest_race_key}.")

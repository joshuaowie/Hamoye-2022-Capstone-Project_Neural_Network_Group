import pandas as pd
import numpy as np
import streamlit as st
import pickle


st.write(""" 
# Injury Prediction App

This app predicts injuries of runners
""")
st.write("-----")

df_day = pd.read_csv('day_approach_maskedID_timeseries.csv')


tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["1st day", "2nd day", "3rd day", "4th day", "5th day", "6th day", "7th day"])


def user_input():
    nr_session = tab1.selectbox('number of sessions',
                                df_day['nr. sessions'].unique())
    total_km = tab1.number_input('total km', min_value=min(df_day['total km']))
    km_Z3_4 = tab1.number_input('km Z3-4', min_value=min(df_day['km Z3-4']))
    km_Z5_T1_T2 = tab1.number_input(
        'km Z5-T1-T2', min_value=min(df_day['km Z5-T1-T2']))
    km_sprinting = tab1.number_input(
        'km sprinting', min_value=min(df_day['km sprinting']))
    strength_training = tab1.selectbox(
        'strength training', df_day['strength training'].unique())
    hours_alternative = tab1.number_input(
        'hours alternative', min_value=min(df_day['hours alternative']))
    perceived_exertion = tab1.number_input(
        'perceived exertion', min_value=min(df_day['perceived exertion']))
    perceived_trainingSuccess = tab1.number_input(
        'perceived trainingSuccess', min_value=min(df_day['perceived trainingSuccess']))
    perceived_recovery = tab1.number_input(
        'perceived recovery', min_value=min(df_day['perceived recovery']))

    nr_session1 = tab2.selectbox('number of sessions.1',
                                 df_day['nr. sessions.1'].unique())
    total_km1 = tab2.number_input(
        'total km.1', min_value=min(df_day['total km.1']))
    km_Z3_41 = tab2.number_input(
        'km Z3-4.1', min_value=min(df_day['km Z3-4.1']))
    km_Z5_T1_T21 = tab2.number_input(
        'km Z5-T1-T2.1', min_value=min(df_day['km Z5-T1-T2.1']))
    km_sprinting1 = tab2.number_input(
        'km sprinting.1', min_value=min(df_day['km sprinting.1']))
    strength_training1 = tab2.selectbox(
        'strength training.1', df_day['strength training.1'].unique())
    hours_alternative1 = tab2.number_input(
        'hours alternative.1', min_value=min(df_day['hours alternative.1']))
    perceived_exertion1 = tab2.number_input(
        'perceived exertion.1', min_value=min(df_day['perceived exertion.1']))
    perceived_trainingSuccess1 = tab2.number_input(
        'perceived trainingSuccess.1', min_value=min(df_day['perceived trainingSuccess.1']))
    perceived_recovery1 = tab2.number_input(
        'perceived recovery.1', min_value=min(df_day['perceived recovery.1']))

    nr_session2 = tab3.selectbox('number of sessions.2',
                                 df_day['nr. sessions.2'].unique())
    total_km2 = tab3.number_input(
        'total km.2', min_value=min(df_day['total km.2']))
    km_Z3_42 = tab3.number_input(
        'km Z3-4.2', min_value=min(df_day['km Z3-4.2']))
    km_Z5_T1_T22 = tab3.number_input(
        'km Z5-T1-T2.2', min_value=min(df_day['km Z5-T1-T2.2']))
    km_sprinting2 = tab3.number_input(
        'km sprinting.2', min_value=min(df_day['km sprinting.2']))
    strength_training2 = tab3.selectbox(
        'strength training.2', df_day['strength training.2'].unique())
    hours_alternative2 = tab3.number_input(
        'hours alternative.2', min_value=min(df_day['hours alternative.2']))
    perceived_exertion2 = tab3.number_input(
        'perceived exertion.2', min_value=min(df_day['perceived exertion.2']))
    perceived_trainingSuccess2 = tab3.number_input(
        'perceived trainingSuccess.2', min_value=min(df_day['perceived trainingSuccess.2']))
    perceived_recovery2 = tab3.number_input(
        'perceived recovery.2', min_value=min(df_day['perceived recovery.2']))

    nr_session3 = tab4.selectbox('number of sessions.3',
                                 df_day['nr. sessions.3'].unique())
    total_km3 = tab4.number_input(
        'total km.3', min_value=min(df_day['total km.3']))
    km_Z3_43 = tab4.number_input(
        'km Z3-4.3', min_value=min(df_day['km Z3-4.3']))
    km_Z5_T1_T23 = tab4.number_input(
        'km Z5-T1-T2.3', min_value=min(df_day['km Z5-T1-T2.3']))
    km_sprinting3 = tab4.number_input(
        'km sprinting.3', min_value=min(df_day['km sprinting.3']))
    strength_training3 = tab4.selectbox(
        'strength training.3', df_day['strength training.3'].unique())
    hours_alternative3 = tab4.number_input(
        'hours alternative.3', min_value=min(df_day['hours alternative.3']))
    perceived_exertion3 = tab4.number_input(
        'perceived exertion.3', min_value=min(df_day['perceived exertion.3']))
    perceived_trainingSuccess3 = tab4.number_input(
        'perceived trainingSuccess.3', min_value=min(df_day['perceived trainingSuccess.3']))
    perceived_recovery3 = tab4.number_input(
        'perceived recovery.3', min_value=min(df_day['perceived recovery.3']))

    nr_session4 = tab5.selectbox('number of sessions.4',
                                 df_day['nr. sessions.4'].unique())
    total_km4 = tab5.number_input(
        'total km.4', min_value=min(df_day['total km.4']))
    km_Z3_44 = tab5.number_input(
        'km Z3-4.4', min_value=min(df_day['km Z3-4.4']))
    km_Z5_T1_T24 = tab5.number_input(
        'km Z5-T1-T2.4', min_value=min(df_day['km Z5-T1-T2.4']))
    km_sprinting4 = tab5.number_input(
        'km sprinting.4', min_value=min(df_day['km sprinting.4']))
    strength_training4 = tab5.selectbox(
        'strength training.4', df_day['strength training.4'].unique())
    hours_alternative4 = tab5.number_input(
        'hours alternative.4', min_value=min(df_day['hours alternative.4']))
    perceived_exertion4 = tab5.number_input(
        'perceived exertion.4', min_value=min(df_day['perceived exertion.4']))
    perceived_trainingSuccess4 = tab5.number_input(
        'perceived trainingSuccess.4', min_value=min(df_day['perceived trainingSuccess.4']))
    perceived_recovery4 = tab5.number_input(
        'perceived recovery.4', min_value=min(df_day['perceived recovery.4']))

    nr_session5 = tab6.selectbox('number of sessions.5',
                                 df_day['nr. sessions.5'].unique())
    total_km5 = tab6.number_input(
        'total km.5', min_value=min(df_day['total km.5']))
    km_Z3_45 = tab6.number_input(
        'km Z3-4.5', min_value=min(df_day['km Z3-4.5']))
    km_Z5_T1_T25 = tab6.number_input(
        'km Z5-T1-T2.5', min_value=min(df_day['km Z5-T1-T2.5']))
    km_sprinting5 = tab6.number_input(
        'km sprinting.5', min_value=min(df_day['km sprinting.5']))
    strength_training5 = tab6.selectbox(
        'strength training.5', df_day['strength training.5'].unique())
    hours_alternative5 = tab6.number_input(
        'hours alternative.5', min_value=min(df_day['hours alternative.5']))
    perceived_exertion5 = tab6.number_input(
        'perceived exertion.5', min_value=min(df_day['perceived exertion.5']))
    perceived_trainingSuccess5 = tab6.number_input(
        'perceived trainingSuccess.5', min_value=min(df_day['perceived trainingSuccess.5']))
    perceived_recovery5 = tab6.number_input(
        'perceived recovery.5', min_value=min(df_day['perceived recovery.5']))

    nr_session6 = tab7.selectbox('number of sessions.6',
                                 df_day['nr. sessions.6'].unique())
    total_km6 = tab7.number_input(
        'total km.6', min_value=min(df_day['total km.6']))
    km_Z3_46 = tab7.number_input(
        'km Z3-4.6', min_value=min(df_day['km Z3-4.6']))
    km_Z5_T1_T26 = tab7.number_input(
        'km Z5-T1-T2.6', min_value=min(df_day['km Z5-T1-T2.6']))
    km_sprinting6 = tab7.number_input(
        'km sprinting.6', min_value=min(df_day['km sprinting.6']))
    strength_training6 = tab7.selectbox(
        'strength training.6', df_day['strength training.6'].unique())
    hours_alternative6 = tab7.number_input(
        'hours alternative.6', min_value=min(df_day['hours alternative.6']))
    perceived_exertion6 = tab7.number_input(
        'perceived exertion.6', min_value=min(df_day['perceived exertion.6']))
    perceived_trainingSuccess6 = tab7.number_input(
        'perceived trainingSuccess.6', min_value=min(df_day['perceived trainingSuccess.6']))
    perceived_recovery6 = tab7.number_input(
        'perceived recovery.6', min_value=min(df_day['perceived recovery.6']))

    data = {
        'nr. sessions': nr_session,
        'total km': total_km,
        'km Z3-4': km_Z3_4,
        'km Z5-T1-T2': km_Z5_T1_T2,
        'km sprinting': km_sprinting,
        'strength training': strength_training,
        'hours alternative': hours_alternative,
        'perceived exertion': perceived_exertion,
        'perceived trainingSuccess': perceived_trainingSuccess,
        'perceived recovery': perceived_recovery,

        'nr. sessions.1': nr_session1,
        'total km.1': total_km1,
        'km Z3-4.1': km_Z3_41,
        'km Z5-T1-T2.1': km_Z5_T1_T21,
        'km sprinting.1': km_sprinting1,
        'strength training.1': strength_training1,
        'hours alternative.1': hours_alternative1,
        'perceived exertion.1': perceived_exertion1,
        'perceived trainingSuccess.1': perceived_trainingSuccess1,
        'perceived recovery.1': perceived_recovery1,

        'nr. sessions.2': nr_session2,
        'total km.2': total_km2,
        'km Z3-4.2': km_Z3_42,
        'km Z5-T1-T2.2': km_Z5_T1_T22,
        'km sprinting.2': km_sprinting2,
        'strength training.2': strength_training2,
        'hours alternative.2': hours_alternative2,
        'perceived exertion.2': perceived_exertion2,
        'perceived trainingSuccess.2': perceived_trainingSuccess2,
        'perceived recovery.2': perceived_recovery2,

        'nr. sessions.3': nr_session3,
        'total km.3': total_km3,
        'km Z3-4.3': km_Z3_43,
        'km Z5-T1-T2.3': km_Z5_T1_T23,
        'km sprinting.3': km_sprinting3,
        'strength training.3': strength_training3,
        'hours alternative.3': hours_alternative3,
        'perceived exertion.3': perceived_exertion3,
        'perceived trainingSuccess.3': perceived_trainingSuccess3,
        'perceived recovery.3': perceived_recovery3,

        'nr. sessions.4': nr_session4,
        'total km.4': total_km4,
        'km Z3-4.4': km_Z3_44,
        'km Z5-T1-T2.4': km_Z5_T1_T24,
        'km sprinting.4': km_sprinting4,
        'strength training.4': strength_training4,
        'hours alternative.4': hours_alternative4,
        'perceived exertion.4': perceived_exertion4,
        'perceived trainingSuccess.4': perceived_trainingSuccess4,
        'perceived recovery.4': perceived_recovery4,

        'nr. sessions.5': nr_session5,
        'total km.5': total_km5,
        'km Z3-4.5': km_Z3_45,
        'km Z5-T1-T2.5': km_Z5_T1_T25,
        'km sprinting.5': km_sprinting5,
        'strength training.5': strength_training5,
        'hours alternative.5': hours_alternative5,
        'perceived exertion.5': perceived_exertion5,
        'perceived trainingSuccess.5': perceived_trainingSuccess5,
        'perceived recovery.5': perceived_recovery5,

        'nr. sessions.6': nr_session6,
        'total km.6': total_km6,
        'km Z3-4.6': km_Z3_46,
        'km Z5-T1-T2.6': km_Z5_T1_T26,
        'km sprinting.6': km_sprinting6,
        'strength training.6': strength_training6,
        'hours alternative.6': hours_alternative6,
        'perceived exertion.6': perceived_exertion6,
        'perceived trainingSuccess.6': perceived_trainingSuccess6,
        'perceived recovery.6': perceived_recovery6,



    }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input()


# display user input details
st.header('User Input Features')
st.write(input_df)
st.write("-----")


if st.button('Predict Injury', key=1):
    load_clf = pickle.load(open('knn_model_day.pkl', 'rb'))

    prediction = load_clf.predict(input_df)
    # prediction_proba = load_clf.predict_proba(input_df)

    st.subheader("Prediction whether Non-injury(0) or Injury(1)")
    # st.write(prediction)
    injury_prediction = np.array(['Non-Injury', 'Injury'])
    st.write(injury_prediction[prediction])

    # st.subheader("Prediction Probability")
    # st.write(prediction_proba)

else:
    st.write("Press to predict Injury!")

import streamlit as st
from urllib.parse import quote_plus
import pandas as pd
import numpy as np
from datetime import datetime
import holidays
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings

# Ignorar os FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Adicionar link para o site de origem dos dados
st.markdown("""
# Previsão de Preços do Petróleo Brent - 15/05/2024 a 15/05/2025

[Dados Históricos - Petróleo Brent Futuros](https://www.investing.com/commodities/brent-oil-historical-data)
""")

# URL do arquivo CSV no GitHub
csv_url = 'https://raw.githubusercontent.com/Henitz/projeto2/master/Dados%20Hist%C3%B3ricos%20-%20Petr%C3%B3leo%20Brent%20Futuros%20(8).csv'

# Carregar dados do Brent
df = pd.read_csv(csv_url)

# Renomear colunas
df = df.rename(columns={'Data': 'ds', 'Último': 'y'})

# Substituir vírgulas por pontos na coluna 'y' e converter para numérico
df['y'] = df['y'].str.replace(',', '.').astype(float)

df['ds'] = pd.to_datetime(df['ds'], format='%d.%m.%Y')

# Remover colunas desnecessárias
colunas_para_remover = ['Abertura', 'Máxima', 'Mínima', 'Vol.', 'Var%']
df = df.drop(columns=colunas_para_remover)

# Obter feriados do Reino Unido de 1970 a 2025
uk_holidays = holidays.UK(years=range(1970, 2026))
holiday_dates = list(uk_holidays.keys())

# Criar DataFrame de feriados
feriados_uk = pd.DataFrame({
    'holiday': 'feriados_uk',
    'ds': pd.to_datetime(holiday_dates),
    'lower_window': 0,
    'upper_window': 1,
})

# Função para prever usando Prophet
def prevendo(df, data, flag):
    m = Prophet(holidays=feriados_uk)
    m.fit(df)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    if flag:
        return m, forecast
    else:
        data_proxima = pd.to_datetime(data, format='%d-%m-%Y')
        if data_proxima.weekday() >= 5 or data_proxima in uk_holidays:
            return m, None
        data_formatada = data_proxima.strftime('%Y-%m-%d')
        previsao = forecast.loc[forecast['ds'] == data_formatada, 'yhat'].values
        if previsao.size == 0:
            return m, None
        return m, previsao[0]

# Função para validar o formato da data
def validar_data(data):
    try:
        datetime.strptime(data, '%d-%m-%Y')
        return True
    except ValueError:
        return False

# Entrada do usuário para a data
data_input = st.text_input("Insira a data para previsão (formato DD-MM-AAAA):")

if data_input:
    if validar_data(data_input):
        model, previsao = prevendo(df, data_input, False)
        data_formatada = pd.to_datetime(data_input, format='%d-%m-%Y').strftime('%d-%m-%Y')
        if previsao is None:
            st.write(f"A data {data_formatada} é um final de semana ou feriado. Não há previsões disponíveis para esta data.")
        else:
            st.write(f"Valor previsto para {data_formatada}: {previsao:.2f}")
    else:
        st.write("Data inválida. Por favor, insira a data no formato DD-MM-AAAA.")

# Gráficos de Previsão
st.write("### Gráfico de Previsão")
model, forecast = prevendo(df, datetime.now().strftime('%d-%m-%Y'), True)

if forecast is not None:
    fig1 = model.plot(forecast)
    st.pyplot(fig1)

    st.write("### Gráfico de Componentes da Previsão")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

    # Filtrar y_true para corresponder às datas em forecast
    df_filtered = df[df['ds'].isin(forecast['ds'])]
    y_true = df_filtered['y'].values
    y_pred = forecast.loc[forecast['ds'].isin(df_filtered['ds']), 'yhat'].values

    # Métricas de Avaliação
    st.write("### Avaliação do Modelo")

    st.write("""
    - **MAE (Mean Absolute Error):** Mede a média dos erros absolutos entre as previsões e os valores reais. Quanto menor o valor, melhor o modelo.
    - **MSE (Mean Squared Error):** Mede a média dos erros quadrados entre as previsões e os valores reais. Dá mais peso aos grandes erros. Quanto menor o valor, melhor o modelo.
    - **RMSE (Root Mean Squared Error):** É a raiz quadrada do MSE. Mantém as unidades dos dados originais e é interpretado da mesma forma que o MSE.
    - **MAPE (Mean Absolute Percentage Error):** Mede a média dos erros percentuais absolutos entre as previsões e os valores reais. Quanto menor o valor, melhor o modelo.
    """)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MSE:** {mse:.2f}")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAPE:** {mape:.2f}")
else:
    st.write("As previsões ainda não estão disponíveis.")
